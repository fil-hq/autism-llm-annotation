import pandas as pd
import os
import time
import numpy as np
import threading
import json
import re
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate
from pydantic import BaseModel, Field
from langchain_community.llms.utils import enforce_stop_tokens
from typing import Optional, List, ClassVar
from langchain.llms.base import LLM
import requests
from retry import retry
import warnings

# 忽略 pandas 的 FutureWarning 警告
warnings.simplefilter(action='ignore', category=FutureWarning)

print_lock = threading.Lock()

def safe_print(message):
    with print_lock:
        print(message)

# --- 0. 配置区域 (一级分类白名单) ---
VALID_L1_OPTIONS = ["分享", "寻求帮助", "广告", "其他", "表意不明", "数据缺失"]

LETTER_MAP_L1 = {
    "A": "表意不明",
    "B": "广告",
    "C": "寻求帮助",
    "D": "分享",
    "E": "其他"
}

# --- 1. 清洗函数 ---
def clean_and_validate_primary_topic(raw_label):
    if pd.isna(raw_label): return "INVALID"
    s_label = str(raw_label).strip()
    s_label = s_label.replace('"', '').replace("'", "").replace("。", "")
    
    if s_label in VALID_L1_OPTIONS: return s_label
    
    cleaned = re.sub(r'^[\d\w]+\.|^Category:|^\w+\s*-\s*|^一级主题：', '', s_label).strip()
    if cleaned in VALID_L1_OPTIONS: return cleaned
    
    clean_letter = s_label.replace(".", "").strip().upper()
    if clean_letter in LETTER_MAP_L1: return LETTER_MAP_L1[clean_letter]

    for option in VALID_L1_OPTIONS:
        if option in s_label:
            if len(s_label) < len(option) + 10: return option

    return "INVALID"

# --- LLM 定义 ---
class Classification(BaseModel):
    primary_topic: str = Field(description="一级主题")

class TopicClassification(BaseModel):
    classification: Classification = Field(description="分类结果")

class ModelChat(LLM):
    history: ClassVar[List] = []
    api_secret: str = ""
    model_name: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B" 
    
    def __init__(self, api_secret: str, model_name: Optional[str] = None):
        super().__init__()
        self.api_secret = api_secret
        if model_name: self.model_name = model_name

    @property
    def _llm_type(self) -> str: return None

    # 网络层面的重试，保持为3次，防止网络波动
    @retry(tries=3, delay=2, backoff=2, jitter=(1, 3))
    def model_completion(self, prompt):
        url = "https://api.siliconflow.cn/v1/chat/completions"
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "max_tokens": 4060,
            "enable_thinking": False, # <--- 记得根据需要修改这里 (True/False)
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
            "Content-Type": "application/json"
        }
        resp = requests.post(url, json=payload, headers=headers, timeout=180)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> Optional[str]:
        for _ in range(3): # 内部调用也限制为最多尝试3次
            try:
                response = self.model_completion(prompt)
                if stop is not None: response = enforce_stop_tokens(response, stop)
                return response
            except Exception as e:
                safe_print(f"[LLM Error] {e}, retrying...")
                time.sleep(5)
        safe_print("[LLM Error] Skipping due to repeated errors.")
        return None

    def __call__(self, prompt: str, stop: Optional[List[str]] = None) -> Optional[str]:
        return self._call(prompt, stop)

PARSER = PydanticOutputParser(pydantic_object=TopicClassification)

# --- 2. Prompt ---
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

# --- 3. Worker ---
def process_chunk_worker(args):
    df_chunk, api_secret, worker_id, is_rerun = args  # 新增 is_rerun 参数
    llm = ModelChat(api_secret)
    df_chunk_processed = df_chunk.copy()
    
    for i, index in enumerate(df_chunk_processed.index):
        # 如果是错分重标任务，特殊处理
        if is_rerun:
            safe_print(f"[Worker {worker_id}] 重新标注错分行: 索引 {index}")
        
        # 预检：content2 为缺失
        content2_val = df_chunk_processed.at[index, '标签']
        if str(content2_val).strip() == "数据缺失":
            df_chunk_processed.at[index, 'primary_topic'] = "数据缺失"
            continue 

        original_identity = df_chunk_processed.at[index, 'identity_new']
        original_title = df_chunk_processed.at[index, 'Title']
        original_content = df_chunk_processed.at[index, 'Content']
        
        identity_str = str(original_identity) if pd.notna(original_identity) else "未知"
        title_str = str(original_title) if pd.notna(original_title) else ""
        content_str = str(original_content) if pd.notna(original_content) else ""
        
        if not title_str.strip() and not content_str.strip():
            df_chunk_processed.at[index, 'primary_topic'] = "数据缺失"
            continue
            
        try:
            combined_text = f"帖子标题：{title_str}\n帖子内容：{content_str}"
            prompt = PROMPT_TEMPLATE.format(poster_role=identity_str, text=combined_text)
        except Exception as e:
            safe_print(f"[Worker {worker_id}] Prompt Error: {e}")
            continue
            
        retry_count = 0
        final_label = "INVALID"
        
        # 内部重试限制：最多试 2 次 (retry_count 0, 1) -> 共2次请求
        # 加上 @retry 的3次，每次请求都很稳，这里是为了应对解析失败
        while retry_count <= 1: 
            try:
                with ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(llm, prompt)
                    response_str = future.result(timeout=180)
                    if not response_str: raise Exception("Empty Response")

                    cleaned_str = response_str.strip()
                    if cleaned_str.startswith('```json'): cleaned_str = cleaned_str[7:]
                    if cleaned_str.endswith('```'): cleaned_str = cleaned_str[:-3]
                    cleaned_str = cleaned_str.strip()
                    
                    parsed_output = PARSER.parse(cleaned_str)
                    raw_primary = parsed_output.classification.primary_topic
                    
                    validated = clean_and_validate_primary_topic(raw_primary)
                    
                    if validated != "INVALID":
                        final_label = validated
                        break 
                    else:
                        safe_print(f"[Worker {worker_id}] 非法标签: '{raw_primary}'，重试中...")
                        retry_count += 1
                        
            except Exception as e:
                retry_count += 1
        
        if final_label == "INVALID":
            df_chunk_processed.at[index, 'primary_topic'] = np.nan
        else:
            df_chunk_processed.at[index, 'primary_topic'] = final_label

    return df_chunk_processed

def _load_progress(file_path, sheet_name):
    try: return pd.read_excel(file_path, sheet_name=sheet_name, engine='openpyxl')
    except: return None

def _save_checkpoint(df, path, sheet):
    safe_print(f"Saving checkpoint to {path}...")
    try:
        if os.path.exists(path):
            with pd.ExcelWriter(path, engine='openpyxl', mode='a', if_sheet_exists='replace') as w:
                df.to_excel(w, sheet_name=sheet, index=False)
        else:
            with pd.ExcelWriter(path, engine='openpyxl', mode='w') as w:
                df.to_excel(w, sheet_name=sheet, index=False)
    except Exception as e: safe_print(f"Save failed: {e}")

# --- 5. 主程序 ---
if __name__ == "__main__":
    FILE_PATH = r"社区数据/LLM标注分类/百度贴吧/重制版/百度贴吧数据标注_deepseek-7b-f-3_错分分析.xlsx"
    SHEET_NAME = "Sheet1" 
    CHECKPOINT_BATCH_SIZE = 80
    
    base, ext = os.path.splitext(FILE_PATH)
    # >>> 修改此处文件名以区分不同进程 <<<
    OUTPUT_FILE_PATH = r"社区数据/LLM标注分类/百度贴吧/重制版/百度贴吧数据标注_deepseek-7b-f-3.xlsx" 
    
    print(f"输出文件: {OUTPUT_FILE_PATH}")
    
    api_secrets = ["sk-nvrapxrvuzitomiewzwrkkmdjcpyyykwvsosrxfwmbrjmxet"]
    num_workers = len(api_secrets)

    if not os.path.exists(FILE_PATH): exit("源文件不存在")
    df_source = pd.read_excel(FILE_PATH, sheet_name=SHEET_NAME, engine='openpyxl')
    df_target = df_source.copy() 
    
    # 检查 content1
    if '标签' not in df_target.columns: exit("缺少 标签 列")
    
    # 检查错分类型列是否存在
    has_misclassification_col = '错分类型' in df_target.columns
    
    if os.path.exists(OUTPUT_FILE_PATH):
        prog = _load_progress(OUTPUT_FILE_PATH, SHEET_NAME)
        if prog is not None and 'primary_topic' in prog.columns:
            df_target['primary_topic'] = prog['primary_topic']
            print("已加载旧进度。")
    
    if 'primary_topic' not in df_target.columns: df_target['primary_topic'] = np.nan
    
    # --- 第一步：处理错分类型为1的数据行 ---
    if has_misclassification_col:
        print("\n" + "="*50)
        print("第一步：处理错分类型为1的数据行")
        print("="*50)
        
        # 找出错分类型为1的数据行
        misclassified_indices = df_target[df_target['错分类型'] == 1].index
        misclassified_count = len(misclassified_indices)
        
        print(f"发现 {misclassified_count} 行需要重新标注的错分数据")
        
        if misclassified_count > 0:
            # 重置这些行的 primary_topic 为 NaN，以便后续重新标注
            df_target.loc[misclassified_indices, 'primary_topic'] = np.nan
            
            # 分批处理错分数据
            misclassified_batches = [misclassified_indices[i:i + CHECKPOINT_BATCH_SIZE] 
                                    for i in range(0, misclassified_count, CHECKPOINT_BATCH_SIZE)]
            
            try:
                with ThreadPoolExecutor(max_workers=num_workers) as executor:
                    for batch_idx in misclassified_batches:
                        current_batch_df = df_target.loc[batch_idx]
                        chunks = np.array_split(current_batch_df, num_workers)
                        tasks = [(chunks[i], api_secrets[i % num_workers], i+1, True)  # 新增 True 表示错分重标
                                for i in range(num_workers) if len(chunks[i]) > 0]
                        
                        results = list(executor.map(process_chunk_worker, tasks))
                        for res in results: df_target.update(res)
                        _save_checkpoint(df_target, OUTPUT_FILE_PATH, SHEET_NAME)
                        
                        print(f"已完成错分重标批次: {len(batch_idx)} 行")
                        
            except KeyboardInterrupt:
                print("错分重标被用户中断")
            except Exception as e:
                print(f"错分重标过程中出错: {e}")
    
    # --- 第二步：处理剩余的未标注数据 ---
    print("\n" + "="*50)
    print("第二步：处理剩余的未标注数据")
    print("="*50)
    
    # --- 核心修改：最大循环次数限制为 3 ---
    loop_count = 0
    max_loops = 3 # <--- 严格限制为3轮
    
    while loop_count < max_loops:
        loop_count += 1
        print(f"\n======== 第 {loop_count} / {max_loops} 轮扫描 ========")

        def needs_rerun(row):
            val = row.get('primary_topic', np.nan)
            s_val = str(val).strip()
            if pd.isna(val) or s_val.lower() == 'nan' or s_val == "": return True
            if clean_and_validate_primary_topic(val) == "INVALID": return True
            return False

        target_indices = df_target[df_target.apply(needs_rerun, axis=1)].index
        count_rerun = len(target_indices)
        print(f">>> 剩余需处理行数: {count_rerun}")
        
        if count_rerun == 0:
            print("处理完毕！")
            break
        
        if loop_count > 1: time.sleep(2)

        index_batches = [target_indices[i:i + CHECKPOINT_BATCH_SIZE] for i in range(0, len(target_indices), CHECKPOINT_BATCH_SIZE)]

        try:
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                for batch_idx in index_batches:
                    current_batch_df = df_target.loc[batch_idx]
                    chunks = np.array_split(current_batch_df, num_workers)
                    tasks = [(chunks[i], api_secrets[i % num_workers], i+1, False)  # 新增 False 表示普通标注
                            for i in range(num_workers) if len(chunks[i]) > 0]
                    
                    results = list(executor.map(process_chunk_worker, tasks))
                    for res in results: df_target.update(res)
                    _save_checkpoint(df_target, OUTPUT_FILE_PATH, SHEET_NAME)
                    
        except KeyboardInterrupt: break
        except Exception as e: print(f"Error: {e}")
            
    _save_checkpoint(df_target, OUTPUT_FILE_PATH, SHEET_NAME)
    
    # 统计最终结果
    print("\n" + "="*50)
    print("标注完成！统计结果：")
    print("="*50)
    print(f"总数据行数: {len(df_target)}")
    print(f"已标注行数: {df_target['primary_topic'].notna().sum()}")
    print(f"未标注行数: {df_target['primary_topic'].isna().sum()}")
    
    if has_misclassification_col:
        misclassified_final = df_target[df_target['错分类型'] == 1]
        print(f"错分类型为1的行数: {len(misclassified_final)}")
        print(f"其中已重新标注的行数: {misclassified_final['primary_topic'].notna().sum()}")
    
    print("一级分类完成。")