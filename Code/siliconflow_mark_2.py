import os
import re
import time
import threading
import warnings
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, List, ClassVar

import numpy as np
import pandas as pd
import requests
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate
from langchain_community.llms.utils import enforce_stop_tokens
from langchain.llms.base import LLM
from pydantic import BaseModel, Field
from retry import retry

# Suppress pandas FutureWarning warnings
warnings.simplefilter(action="ignore", category=FutureWarning)

print_lock = threading.Lock()


def safe_print(message: str) -> None:
    """Thread-safe print to avoid interleaved logs across threads."""
    with print_lock:
        print(message)


# Config (Secondary-topic whitelist)
# Requirement change: remove "Seeking help" secondary classification entirely.
# Keep only the mappings for non-help-seeking primary topics.
VALID_L2_MAP = {
    "分享": ["案例分享", "科普"],
    "广告": ["广告"],
    "其他": ["其他"],
    "表意不明": ["表意不明"],
    "数据缺失": ["数据缺失"],
    "一级缺失": ["一级缺失"],
}

# Letter mapping only applies to "Seeking help" in the original code.
# Since "Seeking help" is removed, we drop LETTER_MAP as well.


# Cleaning / validation helper
def clean_and_validate_label(raw_label, primary_topic):
    """
    Normalize and validate model output for secondary_topic under a given primary_topic.
    Returns a valid label from VALID_L2_MAP[primary_topic], or "INVALID".
    """
    if pd.isna(raw_label):
        return "INVALID"

    s_label = str(raw_label).strip()
    s_label = s_label.replace('"', "").replace("'", "").replace("。", "")

    valid_options = VALID_L2_MAP.get(primary_topic, [])
    if s_label in valid_options:
        return s_label

    cleaned = re.sub(r"^[\d\w]+\.|^Category:|^\w+\s*-\s*", "", s_label).strip()
    if cleaned in valid_options:
        return cleaned

    # Fuzzy containment (only if the output is short enough)
    for option in valid_options:
        if option in s_label and len(s_label) < len(option) + 10:
            return option

    return "INVALID"


# LLM output schema
class Classification(BaseModel):
    secondary_topic: str = Field(description="二级主题")


class TopicClassification(BaseModel):
    classification: Classification = Field(description="分类结果")


# SiliconFlow API wrapper (no hardcoded secrets)
class ModelChat(LLM):
    history: ClassVar[List] = []
    api_secret: str = ""
    model_name: str = "deepseek-ai/DeepSeek-V3.2"

    def __init__(self, api_secret: Optional[str] = None, model_name: Optional[str] = None):
        super().__init__()
        # Prefer explicit param; otherwise read from environment
        self.api_secret = api_secret or os.getenv("SILICONFLOW_API_KEY", "")
        if not self.api_secret:
            raise ValueError("Missing API key. Set SILICONFLOW_API_KEY or pass api_secret.")
        if model_name:
            self.model_name = model_name

    @property
    def _llm_type(self) -> str:
        return None

    @retry(tries=3, delay=2, backoff=2, jitter=(1, 3))
    def model_completion(self, prompt: str) -> str:
        """Call SiliconFlow chat completions endpoint and return assistant content."""
        url = "https://api.siliconflow.cn/v1/chat/completions"
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "max_tokens": 4060,
            "enable_thinking": False,  # change to True/False as needed
            "min_p": 0.05,
            "stop": None,
            "temperature": 0.1,
            "top_p": 0.7,
            "top_k": 50,
            "frequency_penalty": 0.5,
            "n": 1,
        }
        headers = {
            "Authorization": f"Bearer {self.api_secret}",
            "Content-Type": "application/json",
        }

        resp = requests.post(url, json=payload, headers=headers, timeout=180)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> Optional[str]:
        """Internal call with up to 3 attempts and incremental backoff."""
        for attempt in range(3):
            try:
                response = self.model_completion(prompt)
                if stop is not None:
                    response = enforce_stop_tokens(response, stop)
                return response
            except Exception as e:
                safe_print(f"[LLM Error] Attempt {attempt + 1} failed: {e}")
                if attempt < 2:
                    wait_time = 5 * (attempt + 1)
                    safe_print(f"[LLM Error] Waiting {wait_time} seconds before retry...")
                    time.sleep(wait_time)
                else:
                    safe_print("[LLM Error] Skipping this row after repeated failures.")
                    return None
        return None

    def __call__(self, prompt: str, stop: Optional[List[str]] = None) -> Optional[str]:
        return self._call(prompt, stop)


PARSER = PydanticOutputParser(pydantic_object=TopicClassification)

# Prompt template
# Requirement change: remove the "Seeking help" branch entirely from the prompt.
PROMPT_TEMPLATE = PromptTemplate(
    template="""角色：
你是一个严格遵循分类规范的中文文本分类引擎。你的目标是稳定、可复现、零随机地输出唯一正确标签。

任务：
给定一段中文帖子文本、发帖人身份（poster_role），以及已确定的一级主题 primary_topic，你必须从对应的二级主题集合中选择且仅选择一个 secondary_topic，并严格输出指定 JSON 结构，不得输出任何解释、注释或额外字段。

二级主题集合（按 primary_topic 限定，禁止创造新标签）：
当 primary_topic = 分享 时，secondary_topic 只能是：
1.案例分享
2.科普

当 primary_topic = 广告 时，secondary_topic 必须是：广告
当 primary_topic = 其他 时，secondary_topic 必须是：其他
当 primary_topic = 表意不明 时，secondary_topic 必须是：表意不明

二级主题操作化定义（用于 primary_topic = 分享 的细分）：
A. 分享-案例分享：以个人/家庭/孩子的具体经历为主线（诊断过程、干预经历、教育体验、生活记录），主要目的为经验交流与记录，不以提问求助为核心；无明确求助意图；有分享故事、交流或表达感想的意图。
B. 分享-科普：以知识解释、资料整理、链接/图片资源分享为主，语气相对客观中性，不以引流推广为目的。

特殊情况下的判定规则（必须执行，用于降低随机性）：
1.单标签原则：无论文本涉及多少方面，你只能输出一个 secondary_topic。
2.关键词只作线索，必须以“主要关注对象”和“核心意图”作为最终判定依据。

强制自检约束（内部执行，不得输出过程）：
1.你必须确保 secondary_topic 与 primary_topic 完全匹配；若不匹配必须重新选择，直到匹配为止。
2.禁止输出任何不在枚举中的标签、同义词、英文、拼音或新增字段。

输出约束（必须严格遵守）：
只输出一个 JSON 对象，不得包含任何多余内容。
输出字段必须完全一致：classification / secondary_topic

输出格式（严格照抄此结构）
{{
"classification": {{
"secondary_topic": "从对应子列表中选择的二级主题字符串"
}}
}}

任务开始：
请在给定一级主题的前提下，对以下文本进行二级主题分类：
发帖人身份（poster_role）：{poster_role}
已确定的一级主题（primary_topic）：{primary_topic}
文本：
{text}""",
    input_variables=["poster_role", "primary_topic", "text"],
)


# --- 3. Worker ---
def process_chunk_worker(args):
    """
    Worker function that processes one dataframe chunk.
    args: (df_chunk, api_secret, worker_id)
    """
    df_chunk, api_secret, worker_id = args
    llm = ModelChat(api_secret)
    df_chunk_processed = df_chunk.copy()

    for _, index in enumerate(df_chunk_processed.index):
        original_identity = df_chunk_processed.at[index, "identity_new"]
        original_title = df_chunk_processed.at[index, "Title"]
        original_content = df_chunk_processed.at[index, "Content"]
        primary_topic_val = df_chunk_processed.at[index, "primary_topic"]

        identity_str = str(original_identity) if pd.notna(original_identity) else "未知"
        title_str = str(original_title) if pd.notna(original_title) else ""
        content_str = str(original_content) if pd.notna(original_content) else ""

        # If primary_topic is missing, mark as "一级缺失"
        if pd.isna(primary_topic_val) or str(primary_topic_val).strip() == "" or str(primary_topic_val).lower() == "nan":
            df_chunk_processed.at[index, "secondary_topic"] = "一级缺失"
            continue

        primary_topic_str = str(primary_topic_val).strip()

        # Requirement change: skip rows where primary_topic == "寻求帮助"
        if primary_topic_str == "寻求帮助":
            df_chunk_processed.at[index, "secondary_topic"] = np.nan
            continue

        # If text is missing, mark as "数据缺失"
        if not title_str.strip() and not content_str.strip():
            df_chunk_processed.at[index, "secondary_topic"] = "数据缺失"
            continue

        try:
            combined_text = f"帖子标题：{title_str}\n帖子内容：{content_str}"
            prompt = PROMPT_TEMPLATE.format(
                poster_role=identity_str,
                primary_topic=primary_topic_str,
                text=combined_text,
            )
        except Exception as e:
            safe_print(f"[Worker {worker_id}] Prompt Error: {e}")
            continue

        retry_count = 0
        final_label = "INVALID"

        # Parse-level retry: at most 2 attempts (0 and 1) -> 2 API calls
        while retry_count <= 1:
            try:
                with ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(llm, prompt)
                    response_str = future.result(timeout=180)
                    if not response_str:
                        raise Exception("Empty Response")

                    cleaned_str = response_str.strip()
                    if cleaned_str.startswith("```json"):
                        cleaned_str = cleaned_str[7:]
                    if cleaned_str.endswith("```"):
                        cleaned_str = cleaned_str[:-3]
                    cleaned_str = cleaned_str.strip()

                    parsed_output = PARSER.parse(cleaned_str)
                    raw_secondary = parsed_output.classification.secondary_topic

                    validated = clean_and_validate_label(raw_secondary, primary_topic_str)
                    if validated != "INVALID":
                        final_label = validated
                        break
                    else:
                        safe_print(f"[Worker {worker_id}] Invalid label: '{raw_secondary}', retrying...")
                        retry_count += 1

            except Exception as e:
                safe_print(f"[Worker {worker_id}] Error: {e}")
                retry_count += 1

        df_chunk_processed.at[index, "secondary_topic"] = (np.nan if final_label == "INVALID" else final_label)

    return df_chunk_processed


def _load_progress(file_path, sheet_name):
    """Load existing progress from an output Excel file if present."""
    try:
        return pd.read_excel(file_path, sheet_name=sheet_name, engine="openpyxl")
    except Exception:
        return None


def _save_checkpoint(df, path, sheet):
    """Save progress to Excel; replaces the sheet if it already exists."""
    safe_print(f"Saving checkpoint to {path}...")
    try:
        if os.path.exists(path):
            with pd.ExcelWriter(path, engine="openpyxl", mode="a", if_sheet_exists="replace") as w:
                df.to_excel(w, sheet_name=sheet, index=False)
        else:
            with pd.ExcelWriter(path, engine="openpyxl", mode="w") as w:
                df.to_excel(w, sheet_name=sheet, index=False)
    except Exception as e:
        safe_print(f"Save failed: {e}")


def test_api_key(api_key: str) -> bool:
    """Test whether the SiliconFlow API key is valid."""
    safe_print("Testing API key...")
    url = "https://api.siliconflow.cn/v1/models"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    try:
        resp = requests.get(url, headers=headers, timeout=10)
        if resp.status_code == 200:
            safe_print("✓ API key is valid.")
            return True
        if resp.status_code == 403:
            safe_print("✗ API key is invalid or expired.")
            return False
        safe_print(f"? Unexpected response: HTTP {resp.status_code}")
        return False
    except Exception as e:
        safe_print(f"! Connection test failed: {e}")
        return False


#  Main
if __name__ == "__main__":
    # Use placeholder path as requested
    FILE_PATH = r"path/to/data.xlsx"
    SHEET_NAME = "Sheet1"
    CHECKPOINT_BATCH_SIZE = 50

    base, ext = os.path.splitext(FILE_PATH)
    OUTPUT_FILE_PATH = f"{base}_Level2{ext}"

    print(f"Input:  {FILE_PATH}")
    print(f"Output: {OUTPUT_FILE_PATH}")

    # Load API keys from environment (recommended).
    # Example: export SILICONFLOW_API_KEYS="key1,key2"
    api_keys_env = os.getenv("SILICONFLOW_API_KEYS", "").strip()
    if api_keys_env:
        api_secrets = [k.strip() for k in api_keys_env.split(",") if k.strip()]
    else:
        single = os.getenv("SILICONFLOW_API_KEY", "").strip()
        api_secrets = [single] if single else []

    # Optionally validate keys (quick check). If you trust your env, you can skip this block.
    valid_api_keys = [k for k in api_secrets if test_api_key(k)]
    if not valid_api_keys:
        raise ValueError("No valid API keys found. Set SILICONFLOW_API_KEY(S) correctly.")

    api_secrets = valid_api_keys
    num_workers = len(api_secrets)

    if not os.path.exists(FILE_PATH):
        raise FileNotFoundError("Level-1 file not found. Run Level-1 classification first.")

    df_source = pd.read_excel(FILE_PATH, sheet_name=SHEET_NAME, engine="openpyxl")
    df_target = df_source.copy()

    # Load previous progress if any
    if os.path.exists(OUTPUT_FILE_PATH):
        prog = _load_progress(OUTPUT_FILE_PATH, SHEET_NAME)
        if prog is not None and "secondary_topic" in prog.columns:
            df_target["secondary_topic"] = prog["secondary_topic"]
            print("Loaded previous progress.")

    if "secondary_topic" not in df_target.columns:
        df_target["secondary_topic"] = np.nan

    # Quick API connectivity smoke test
    safe_print("\nTesting API connectivity...")
    test_llm = ModelChat(api_secrets[0])
    try:
        test_result = test_llm("请回复'测试成功'四个字")
        safe_print("✓ API connectivity test succeeded." if test_result else "✗ API connectivity test returned empty.")
    except Exception as e:
        safe_print(f"! API connectivity test raised an error: {e}")

    print("\n" + "=" * 50)
    print("Processing remaining unlabeled rows (Level-2, excluding Seeking help)")
    print("=" * 50)

    loop_count = 0
    max_loops = 3  # strictly limit to 3 passes

    while loop_count < max_loops:
        loop_count += 1
        print(f"\n======== Pass {loop_count} / {max_loops} ========")

        def needs_rerun(row):
            p_val = row.get("primary_topic", "")
            if pd.isna(p_val) or str(p_val).strip() == "":
                return False

            # Requirement change: do not process Seeking help rows
            if str(p_val).strip() == "寻求帮助":
                return False

            val = row.get("secondary_topic", np.nan)
            if pd.isna(val) or str(val).lower() == "nan":
                return True
            if clean_and_validate_label(val, str(p_val).strip()) == "INVALID":
                return True
            return False

        target_indices = df_target[df_target.apply(needs_rerun, axis=1)].index
        count_rerun = len(target_indices)
        print(f">>> Remaining rows to process: {count_rerun}")

        if count_rerun == 0:
            print("All done!")
            break

        if loop_count > 1:
            time.sleep(2)

        index_batches = [
            target_indices[i : i + CHECKPOINT_BATCH_SIZE]
            for i in range(0, len(target_indices), CHECKPOINT_BATCH_SIZE)
        ]

        try:
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                for batch_idx in index_batches:
                    current_batch_df = df_target.loc[batch_idx]
                    chunks = np.array_split(current_batch_df, num_workers)
                    tasks = [
                        (chunks[i], api_secrets[i % num_workers], i + 1)
                        for i in range(num_workers)
                        if len(chunks[i]) > 0
                    ]

                    results = list(executor.map(process_chunk_worker, tasks))
                    for res in results:
                        df_target.update(res)
                    _save_checkpoint(df_target, OUTPUT_FILE_PATH, SHEET_NAME)

        except KeyboardInterrupt:
            safe_print("Interrupted by user.")
            break
        except Exception as e:
            safe_print(f"Error: {e}")
            break

    _save_checkpoint(df_target, OUTPUT_FILE_PATH, SHEET_NAME)

    # Final summary
    print("\n" + "=" * 50)
    print("Level-2 labeling completed! Summary:")
    print("=" * 50)
    print(f"Total rows: {len(df_target)}")
    print(f"Labeled rows: {df_target['secondary_topic'].notna().sum()}")
    print(f"Unlabeled rows: {df_target['secondary_topic'].isna().sum()}")

    print("Secondary-topic classification finished.")