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


#  Whitelist and mappings
# Keep whitelist aligned with the prompt's topic set
VALID_OPTIONS = [
    "自闭症症状",
    "自闭症检查",
    "自闭症诊断",
    "病因/诱发因素咨询",
    "费用与经济负担咨询",
    "自闭症日常管理",
    "自闭症干预",
    "自闭症资源推荐（评价）",
    "其他求助",
]

# Letter-to-label mapping (A-H)
LETTER_MAP = {
    "A": "自闭症症状",
    "B": "自闭症检查",
    "C": "自闭症诊断",
    "D": "病因/诱发因素咨询",
    "E": "费用与经济负担咨询",
    "F": "自闭症日常管理",
    "G": "自闭症干预",
    "H": "自闭症资源推荐（评价）",
    "I": "其他求助",
}


#  Cleaning / validation helper
def clean_and_validate_label(raw_label):
    """
    Normalize and validate model output for topic.
    Returns a valid label from VALID_OPTIONS, or "INVALID".
    """
    if pd.isna(raw_label):
        return "INVALID"

    s_label = str(raw_label).strip()
    s_label = s_label.replace('"', "").replace("'", "").replace("。", "")

    # 1) Direct match
    if s_label in VALID_OPTIONS:
        return s_label

    # 2) Prefix stripping (e.g., "1. 综合", "Topic: 干预措施")
    cleaned = re.sub(r"^[\d\w]+\.|^Category:|^\w+\s*-\s*|^Topic:", "", s_label).strip()
    if cleaned in VALID_OPTIONS:
        return cleaned

    # 3) Substring match (for verbose outputs like "主题：综合")
    for option in VALID_OPTIONS:
        if option in s_label and len(s_label) < len(option) + 10:
            return option

    # 4) Letter mapping (A -> label)
    key = s_label.upper().strip()
    if len(key) <= 2 and key in LETTER_MAP:
        return LETTER_MAP[key]

    return "INVALID"


# Pydantic schema for parsing
class ClassificationContent(BaseModel):
    topic: str = Field(description="主题")


class TopicClassification(BaseModel):
    classification: ClassificationContent = Field(description="分类结果")


# vLLM (OpenAI-compatible) client wrapper
class ModelChat(LLM):
    """
    A LangChain LLM wrapper that calls a local vLLM server via OpenAI-compatible endpoints.
    """
    history: ClassVar[List] = []
    api_base: str = "http://localhost:8080/v1"
    model_name: str = "DeepSeek-R1-Distill-Qwen-32B"

    def __init__(self, api_base: Optional[str] = None, model_name: Optional[str] = None):
        super().__init__()
        if api_base:
            self.api_base = api_base
        if model_name:
            self.model_name = model_name

    @property
    def _llm_type(self) -> str:
        return "vllm"

    @retry(tries=3, delay=2, backoff=2, jitter=(1, 3))
    def model_completion(self, prompt: str) -> str:
        """Call local vLLM chat completions endpoint and return assistant content."""
        url = f"{self.api_base}/chat/completions"
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "max_tokens": 4060,
            "temperature": 0.1,
            "top_p": 0.7,
            "top_k": 50,
            "frequency_penalty": 0.5,
            "n": 1,
        }
        headers = {"Content-Type": "application/json"}
        resp = requests.post(url, json=payload, headers=headers, timeout=180)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> Optional[str]:
        """Internal call with up to 3 attempts."""
        for _ in range(3):
            try:
                response = self.model_completion(prompt)
                if stop is not None:
                    response = enforce_stop_tokens(response, stop)
                return response
            except Exception as e:
                safe_print(f"[LLM Error] {e}, retrying...")
                time.sleep(5)
        safe_print("[LLM Error] Skipping due to repeated errors.")
        return None

    def __call__(self, prompt: str, stop: Optional[List[str]] = None) -> Optional[str]:
        return self._call(prompt, stop)


PARSER = PydanticOutputParser(pydantic_object=TopicClassification)

# Prompt template
# Requirement change: replace this prompt with the one you provided in the previous code (the rule-driven 9-topic prompt using QA + text).
PROMPT_TEMPLATE = PromptTemplate(
    template="""角色：
你是一个用于在线平台的
【严格规则驱动的主题分类引擎】。

你的任务是：仅依据患者需求文本（text），
按照给定主题定义与“混淆类区分规则”，
选择一个且仅一个最恰当的主题标签。

你不是在理解医学，而是在执行分类规则。

────────────────
【重要总规则】
1. 只依据 text 中患者“最核心、最想被医生回答的问题”分类。
2. QA 仅用于补全上下文，不得作为分类依据。
3. 单标签原则：无论涉及多少内容，只输出一个主题。
4. 若某类别的“混淆类区分规则”被触发，必须服从，不得自行权衡。
5. 若多个类别看似符合，优先选择规则限制最明确的那个。

────────────────
【主题集合（只能选一个）】
1. 自闭症症状
2. 自闭症检查
3. 自闭症诊断
4. 病因/诱发因素咨询
5. 费用与经济负担咨询
6. 自闭症干预
7. 自闭症资源推荐（评价）
8. 其他求助

────────────────
【主题定义 + 混淆类区分（必须逐条遵守）】

1【自闭症症状】
简述：描述行为或发育表现，询问这些表现是否异常或自闭症有哪些症状。
规则限制（必须全部满足）：
- 仅包含行为/发育表现的描述或提问；
- 询问自闭症有什么症状/该症状是否是自闭症
- 未询问“是否为自闭症 / 是否确诊 / 如何诊断”；
- 未提及任何检查、量表、评估或测试结果。
若出现诊断判断或检查信息禁止使用本类。

────────────────
2【自闭症检查】
简述：围绕检查、量表、评估工具或测试结果本身提问。
规则限制：
- 核心问题是“检查/量表/视频/结果如何理解、是否需要做、是否可靠”；
- 未直接要求判断“是不是自闭症”。
若检查仅作为依据用于判断是否自闭症转为【自闭症诊断】。
若是检查去哪里做比较好等重点不在检查的转为【自闭症资源推荐（评价）】

────────────────
3【自闭症诊断】
简述：围绕“是否为自闭症”、诊断流程、严重程度或分型进行判断。
规则限制（强制优先）：
- 只要文本出现以下任一情形，必须归入本类：
  · 是否为自闭症 / 是否属于自闭症 / 是否确诊 / 担心是自闭症；
  · 诊断流程、诊断标准、分型或严重程度；
  · 使用症状或检查结果来判断是否自闭症。
一旦触发关键规则关键词等，本类优先于所有其他类别。

────────────────
4【病因/诱发因素咨询】
简述：询问自闭症“为什么会发生”，关注成因或风险因素。
规则限制：
- 核心问题指向年龄、遗传、孕期、围产期、环境或责任归因,如询问由于xx原因是否会得自闭症;
- 不以诊断、检查或干预为主要问题。

────────────────
5【费用与经济负担咨询】
简述：围绕费用、经济压力、支付能力提问。
规则限制：
- 费用/报销/经济承受力是核心问题；
- 非“哪家机构好”等资源选择问题。

────────────────
6【自闭症干预】
简述：咨询日常或专业干预或训练方式、路径、效果或实施策略，包括预后效果。
规则限制：
- 明确出现具体干预方法或训练方案（如 ABA、语言训练、感统等）或药物治疗、日常生活中的引导等；
- 或明确讨论治疗、康复、预后并与干预决策直接相关；
- 讨论能不能治好、有什么影响等预后问题。
若仍在判断是否自闭症转为【自闭症诊断】。

────────────────
7【自闭症资源推荐（评价）】
简述：询问或评价医院、医生、康复机构、老师等资源，包括挂号、去哪个科室等咨询。
规则限制：
- 核心问题是“去哪里看 / 找谁 / 哪个机构好不好”；
- 涉及地点、机构、人员选择。
强调地点或推荐优先本类。

────────────────
8【其他求助】
简述：存在求助意图但无法归入任何具体类别。
规则限制（兜底类）：
- 不符合任何上述类别的最低判定条件；
- 问题高度混乱、泛化或与自闭症关联较弱。
若能勉强归入具体类别不得使用本类，通过抓核心观点也无法具体分类再使用本类，若询问的检查、日常管理、治疗等均与自闭症无关是询问的别的疾病即使用本类

输出约束：
必须输出严格 JSON，不包含 Markdown 或解释文字。
输出格式：
{{"classification": {{"topic": "在此处填入上述9个主题中的一个"}}}}

任务开始：
医患对话上下文(QA)：
{QA}

需要分类的患者需求文本(text)：
{text}""",
    input_variables=["QA", "text"],
)

# NOTE:
# The prompt above produces a 9-topic label set, while VALID_OPTIONS currently lists 8 labels.
# To keep the code consistent and avoid rejecting correct labels, we update VALID_OPTIONS
# to match the prompt's 9-topic set (and remove the older 8-topic set).
VALID_OPTIONS = [
    "自闭症症状",
    "自闭症检查",
    "自闭症诊断",
    "病因/诱发因素咨询",
    "费用与经济负担咨询",
    "自闭症干预",
    "自闭症资源推荐（评价）",
    "其他求助",
]

LETTER_MAP = {
    "A": "自闭症症状",
    "B": "自闭症检查",
    "C": "自闭症诊断",
    "D": "病因/诱发因素咨询",
    "E": "费用与经济负担咨询",
    "F": "自闭症干预",
    "G": "自闭症资源推荐（评价）",
    "H": "其他求助",
}


# Worker function
def process_chunk_worker(args):
    """
    Worker function that processes one dataframe chunk.
    args: (df_chunk, api_base, worker_id, qa_col_name, model_name)
    """
    df_chunk, api_base, worker_id, qa_col_name, model_name = args
    llm = ModelChat(api_base=api_base, model_name=model_name)
    df_chunk_processed = df_chunk.copy()

    for _, index in enumerate(df_chunk_processed.index):
        # 1) Read the text to classify
        if "内容" in df_chunk_processed.columns:
            text_val = df_chunk_processed.at[index, "内容"]
        elif "Content" in df_chunk_processed.columns:
            text_val = df_chunk_processed.at[index, "Content"]
        else:
            text_val = ""

        # 2) Read QA context (context only; not a classification basis)
        if qa_col_name in df_chunk_processed.columns:
            qa_val = df_chunk_processed.at[index, qa_col_name]
        else:
            qa_val = "暂无上下文"

        # 3) Normalize and check emptiness
        text_str = str(text_val) if pd.notna(text_val) else ""
        qa_str = str(qa_val) if pd.notna(qa_val) else ""

        if not text_str.strip():
            df_chunk_processed.at[index, "secondary_topic"] = "数据缺失"
            continue

        # 4) Build the prompt
        try:
            prompt = PROMPT_TEMPLATE.format(QA=qa_str, text=text_str)
        except Exception as e:
            safe_print(f"[Worker {worker_id}] Prompt Format Error: {e}")
            continue

        retry_count = 0
        final_label = "INVALID"

        # 5) Request loop (parse-level retry: at most 2 attempts)
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
                    raw_topic = parsed_output.classification.topic

                    validated = clean_and_validate_label(raw_topic)
                    if validated != "INVALID":
                        final_label = validated
                        break
                    else:
                        safe_print(f"[Worker {worker_id}] Invalid label: '{raw_topic}' -> retrying...")
                        retry_count += 1
            except Exception:
                retry_count += 1

        # 6) Write result
        df_chunk_processed.at[index, "secondary_topic"] = (np.nan if final_label == "INVALID" else final_label)

    return df_chunk_processed


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

    # IMPORTANT: set this to the column name that stores full dialogue QA context in your file
    QA_COLUMN_NAME = "医患对话内容"

    CHECKPOINT_BATCH_SIZE = 50

    base, ext = os.path.splitext(FILE_PATH)
    OUTPUT_FILE_PATH = r"path/to/output.xlsx"

    print(f"Input:  {FILE_PATH}")
    print(f"Output: {OUTPUT_FILE_PATH}")

    # vLLM configs (you can add multiple instances for parallelism)
    vllm_configs = [
        {
            "api_base": "http://localhost:8080/v1",
            "model_name": "DeepSeek-R1-Distill-Qwen-32B",
        }
    ]
    num_workers = len(vllm_configs)

    # Read data (supports CSV and Excel)
    if not os.path.exists(FILE_PATH):
        raise FileNotFoundError("Input file does not exist.")

    try:
        if FILE_PATH.endswith(".csv"):
            df_source = pd.read_csv(FILE_PATH)
        else:
            df_source = pd.read_excel(FILE_PATH, engine="openpyxl")
    except Exception as e:
        raise RuntimeError(f"Failed to read input file: {e}")

    df_target = df_source.copy()

    # Show available columns for debugging
    print("Columns in the file:", df_target.columns.tolist())

    if QA_COLUMN_NAME not in df_target.columns:
        print(f"Warning: QA context column '{QA_COLUMN_NAME}' not found.")
        print("Please set QA_COLUMN_NAME to the correct column name.")

    # Load previous progress if any
    if os.path.exists(OUTPUT_FILE_PATH):
        try:
            prog = pd.read_excel(OUTPUT_FILE_PATH, engine="openpyxl")
            if "secondary_topic" in prog.columns:
                df_target["secondary_topic"] = prog["secondary_topic"]
                print("Loaded previous progress.")
        except Exception:
            pass

    if "secondary_topic" not in df_target.columns:
        df_target["secondary_topic"] = np.nan

    # --- Main loop: up to 3 passes ---
    loop_count = 0
    max_loops = 3

    while loop_count < max_loops:
        loop_count += 1
        print(f"\n======== Pass {loop_count} / {max_loops} ========")

        def needs_rerun(row):
            val = row.get("secondary_topic", np.nan)
            if pd.isna(val) or str(val).lower() == "nan":
                return True
            if clean_and_validate_label(val) == "INVALID":
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
                        (
                            chunks[i],
                            vllm_configs[i % num_workers]["api_base"],
                            i + 1,
                            QA_COLUMN_NAME,
                            vllm_configs[i % num_workers]["model_name"],
                        )
                        for i in range(num_workers)
                        if len(chunks[i]) > 0
                    ]

                    results = list(executor.map(process_chunk_worker, tasks))
                    for res in results:
                        df_target.update(res)
                    _save_checkpoint(df_target, OUTPUT_FILE_PATH, "Sheet1")

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")

    _save_checkpoint(df_target, OUTPUT_FILE_PATH, "Sheet1")
    print("Done.")