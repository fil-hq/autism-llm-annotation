import os
import re
import time
import threading
import warnings
from concurrent.futures import ThreadPoolExecutor

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


# Config (Primary-topic whitelist)
VALID_L1_OPTIONS = ["分享", "寻求帮助", "广告", "其他", "表意不明", "数据缺失"]

LETTER_MAP_L1 = {
    "A": "表意不明",
    "B": "广告",
    "C": "寻求帮助",
    "D": "分享",
    "E": "其他",
}


# Cleaning / validation helper
def clean_and_validate_primary_topic(raw_label):
    """
    Normalize and validate model output for primary_topic.
    Returns a valid label from VALID_L1_OPTIONS, or "INVALID".
    """
    if pd.isna(raw_label):
        return "INVALID"

    s_label = str(raw_label).strip()
    s_label = s_label.replace('"', "").replace("'", "").replace("。", "")

    # Exact match
    if s_label in VALID_L1_OPTIONS:
        return s_label

    # Remove common prefixes like "1.", "Category:", "X -", "一级主题："
    cleaned = re.sub(r"^[\d\w]+\.|^Category:|^\w+\s*-\s*|^一级主题：", "", s_label).strip()
    if cleaned in VALID_L1_OPTIONS:
        return cleaned

    # Map single-letter choices to labels
    clean_letter = s_label.replace(".", "").strip().upper()
    if clean_letter in LETTER_MAP_L1:
        return LETTER_MAP_L1[clean_letter]

    # Fuzzy containment (only if the output is short enough)
    for option in VALID_L1_OPTIONS:
        if option in s_label and len(s_label) < len(option) + 10:
            return option

    return "INVALID"


# --- LLM output schema ---
class Classification(BaseModel):
    primary_topic: str = Field(description="一级主题")


class TopicClassification(BaseModel):
    classification: Classification = Field(description="分类结果")


# --- vLLM (OpenAI-compatible) client wrapper ---
class ModelChat(LLM):
    """
    A LangChain LLM wrapper that calls a local vLLM server via OpenAI-compatible endpoints.
    """
    history: ClassVar[list] = []
    api_base: str = "http://localhost:8080/v1"
    served_model_name: str = "DeepSeek-R1-Distill-Qwen-32B"

    def __init__(self, api_base: Optional[str] = None, served_model_name: Optional[str] = None):
        super().__init__()
        if api_base:
            self.api_base = api_base
        if served_model_name:
            self.served_model_name = served_model_name

    @property
    def _llm_type(self) -> str:
        return None

    # Network-level retry: keep 3 tries to handle transient issues
    @retry(tries=3, delay=2, backoff=2, jitter=(1, 3))
    def model_completion(self, prompt: str) -> str:
        """Call local vLLM chat completions endpoint and return assistant content."""
        url = f"{self.api_base}/chat/completions"
        payload = {
            "model": self.served_model_name,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "max_tokens": 4060,
            "temperature": 0.1,
            "top_p": 0.7,
            "frequency_penalty": 0.5,
            "n": 1,
        }
        headers = {"Content-Type": "application/json"}
        # Use a longer timeout for long contexts / slower generations
        resp = requests.post(url, json=payload, headers=headers, timeout=300)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]

    def _call(self, prompt: str, stop: Optional[list[str]] = None) -> Optional[str]:
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

    def __call__(self, prompt: str, stop: Optional[list[str]] = None) -> Optional[str]:
        return self._call(prompt, stop)


PARSER = PydanticOutputParser(pydantic_object=TopicClassification)

# Prompt template
PROMPT_TEMPLATE = PromptTemplate(
    template="""角色：
你是一个严格遵循分类规范的中文文本分类引擎。你的目标是稳定、可复现、零随机地输出唯一正确标签。

任务：
给定一段中文帖子文本及发帖人身份（poster_role），你必须从“一级主题集合”中选择且仅选择一个一级主题 primary_topic，并严格输出指定 JSON 结构，不得输出任何解释、注释或额外字段。

一级主题集合（只能从中选择一个，禁止创造新标签）：
1.分享
2.寻求帮助
3.广告
4.其他
5.表意不明

一级主题定义（必须按以下操作化定义判断）：
A. 表意不明：内容语义模糊、句子不完整、语法混乱、乱码或信息不足，无法明确理解作者意图。
B. 广告：明显带有推广、营销、引流目的的内容，包括软广和硬广；常含机构名称、课程宣传、联系方式（微信号/电话/二维码）、行动号召（报名、试听、加我等）。
C. 寻求帮助：以向他人寻求建议、答案或资源为主要目的，包括诊断、干预、医院/机构/医生/老师推荐评价、日常生活求助，以及无法细分的综合求助。
D. 分享：以主动分享信息或经验为主要目的，包括案例经历分享与科普资料分享，不以索取建议为核心目的。
E. 其他：无法归入上述任何类别，但文本可理解且非技术性缺失。

强制优先级规则（用于降低随机性，必须严格执行）:
优先级从高到低依次判定，满足任一条即停止并输出对应标签：
1.若文本包含明显广告引流线索（机构名 + 联系方式/加我/报名/试听/优惠/二维码/微信号/电话等），则 primary_topic = 广告。
2.若文本主要是提出问题、求建议、求推荐、求判断（出现“求”“请问”“帮忙看看/判断”“有没有推荐”等），则 primary_topic = 寻求帮助。
3.若文本主要在讲述经历、记录变化、分享资料/知识/链接，且不以求助为核心，则 primary_topic = 分享。
4.若文本可理解但与自闭症主题/上述功能不匹配，或属于边缘讨论，则 primary_topic = 其他。
5.若文本极短、乱码、难以理解或无法判断意图，则 primary_topic = 表意不明。

在“分享”与“寻求帮助”之间的判定中，请严格执行以下规则：
1.“寻求帮助”必须满足一个必要条件：发帖者明确向他人索取回应（如判断、建议、推荐或解决方案）。
2.仅描述困难、情绪、病情或经历，不构成“寻求帮助”。
3.当帖子主要结构为经历回顾、过程记录或经验总结，且未出现明确求助语言时，应优先判定为“分享”，即使内容中包含痛苦、挫折或问题描述。

与发帖人身份的结合规则（只能作为辅助，不得覆盖文本证据）:
1.poster_role 为“相关商业从业者”时，若文本出现引流或推广线索，优先判为“广告”；若无引流且内容为知识解释，可判为“分享”。
2.poster_role 为“患者/患者家属”时，若出现明确提问或求助语言，优先判为“寻求帮助”。

输出约束（必须严格遵守）:
只输出一个 JSON 对象，不得包含任何多余内容。
输出字段必须完全一致：classification / primary_topic
primary_topic 只能取以下之一：
["分享","寻求帮助","广告","其他","表意不明"]

输出格式（严格照抄此结构）
{{
"classification": {{
"primary_topic": "从一级主题集合中选择的字符串"
}}
}}

任务开始：
请对以下文本进行一级主题分类：
发帖人身份（poster_role）：{poster_role}
文本：
{text}""",
    input_variables=["poster_role", "text"],
)


# Worker
def process_chunk_worker(args):
    """
    Worker function that processes one dataframe chunk.
    args: (df_chunk, worker_id)
    """
    df_chunk, worker_id = args

    # Local vLLM server: no API key required
    llm = ModelChat(api_base="http://localhost:8080/v1", served_model_name="DeepSeek-R1-Distill-Qwen-32B")
    df_chunk_processed = df_chunk.copy()

    for _, index in enumerate(df_chunk_processed.index):
        # Pre-check: '标签' indicates missing data
        tag_val = df_chunk_processed.at[index, "标签"]
        if str(tag_val).strip() == "数据缺失":
            df_chunk_processed.at[index, "primary_topic"] = "数据缺失"
            continue

        original_identity = df_chunk_processed.at[index, "identity_new"]
        original_title = df_chunk_processed.at[index, "Title"]
        original_content = df_chunk_processed.at[index, "Content"]

        identity_str = str(original_identity) if pd.notna(original_identity) else "未知"
        title_str = str(original_title) if pd.notna(original_title) else ""
        content_str = str(original_content) if pd.notna(original_content) else ""

        if not title_str.strip() and not content_str.strip():
            df_chunk_processed.at[index, "primary_topic"] = "数据缺失"
            continue

        try:
            combined_text = f"帖子标题：{title_str}\n帖子内容：{content_str}"
            prompt = PROMPT_TEMPLATE.format(poster_role=identity_str, text=combined_text)
        except Exception as e:
            safe_print(f"[Worker {worker_id}] Prompt Error: {e}")
            continue

        retry_count = 0
        final_label = "INVALID"

        # Parse-level retry: at most 2 attempts (0 and 1) -> 2 generation requests
        while retry_count <= 1:
            try:
                with ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(llm, prompt)
                    response_str = future.result(timeout=300)
                    if not response_str:
                        raise Exception("Empty Response")

                    cleaned_str = response_str.strip()
                    if cleaned_str.startswith("```json"):
                        cleaned_str = cleaned_str[7:]
                    if cleaned_str.endswith("```"):
                        cleaned_str = cleaned_str[:-3]
                    cleaned_str = cleaned_str.strip()

                    parsed_output = PARSER.parse(cleaned_str)
                    raw_primary = parsed_output.classification.primary_topic

                    validated = clean_and_validate_primary_topic(raw_primary)
                    if validated != "INVALID":
                        final_label = validated
                        break
                    else:
                        safe_print(f"[Worker {worker_id}] Invalid label: '{raw_primary}', retrying...")
                        retry_count += 1

            except Exception as e:
                safe_print(f"[Worker {worker_id}] vLLM API Error: {e}")
                retry_count += 1

        df_chunk_processed.at[index, "primary_topic"] = (np.nan if final_label == "INVALID" else final_label)

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


# Main
if __name__ == "__main__":
    # Use placeholder paths as requested
    FILE_PATH = r"path/to/data.xlsx"
    SHEET_NAME = "Sheet1"
    CHECKPOINT_BATCH_SIZE = 80

    OUTPUT_FILE_PATH = r"path/to/output.xlsx"
    print(f"Output file: {OUTPUT_FILE_PATH}")

    # Configure worker count (adjust based on available GPUs / throughput)
    num_workers = 2

    # Check whether local vLLM API is reachable
    try:
        response = requests.get("http://localhost:8080/v1/models", timeout=5)
        if response.status_code == 200:
            print("✅ Connected to local vLLM API server.")
            models = response.json()
            print(f"Available models: {models}")
        else:
            print(f"❌ Local API responded with status: {response.status_code}")
            raise RuntimeError("Local vLLM API unhealthy.")
    except Exception as e:
        print(f"❌ Cannot connect to local vLLM API server: {e}")
        print("Please make sure the vLLM server is running, e.g.:")
        print("python -m vllm.entrypoints.openai.api_server ...")
        raise

    if not os.path.exists(FILE_PATH):
        raise FileNotFoundError("Source file does not exist.")

    df_source = pd.read_excel(FILE_PATH, sheet_name=SHEET_NAME, engine="openpyxl")
    df_target = df_source.copy()

    # Ensure required columns exist
    if "标签" not in df_target.columns:
        raise ValueError("Missing required column: 标签")

    # Load previous progress if any
    if os.path.exists(OUTPUT_FILE_PATH):
        prog = _load_progress(OUTPUT_FILE_PATH, SHEET_NAME)
        if prog is not None and "primary_topic" in prog.columns:
            df_target["primary_topic"] = prog["primary_topic"]
            print("Loaded previous progress.")

    if "primary_topic" not in df_target.columns:
        df_target["primary_topic"] = np.nan

    # Process remaining unlabeled rows
    print("\n" + "=" * 50)
    print("Processing remaining unlabeled rows")
    print("=" * 50)

    loop_count = 0
    max_loops = 3  # strictly limit to 3 passes

    while loop_count < max_loops:
        loop_count += 1
        print(f"\n======== Pass {loop_count} / {max_loops} ========")

        def needs_rerun(row):
            val = row.get("primary_topic", np.nan)
            s_val = str(val).strip()
            if pd.isna(val) or s_val.lower() == "nan" or s_val == "":
                return True
            if clean_and_validate_primary_topic(val) == "INVALID":
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
                    tasks = [(chunks[i], i + 1) for i in range(num_workers) if len(chunks[i]) > 0]

                    results = list(executor.map(process_chunk_worker, tasks))
                    for res in results:
                        df_target.update(res)
                    _save_checkpoint(df_target, OUTPUT_FILE_PATH, SHEET_NAME)

        except KeyboardInterrupt:
            safe_print("Interrupted by user.")
            break
        except Exception as e:
            safe_print(f"Error: {e}")

    _save_checkpoint(df_target, OUTPUT_FILE_PATH, SHEET_NAME)

    # Final summary
    print("\n" + "=" * 50)
    print("Labeling completed! Summary:")
    print("=" * 50)
    print(f"Total rows: {len(df_target)}")
    print(f"Labeled rows: {df_target['primary_topic'].notna().sum()}")
    print(f"Unlabeled rows: {df_target['primary_topic'].isna().sum()}")

    print("Primary-topic classification finished.")