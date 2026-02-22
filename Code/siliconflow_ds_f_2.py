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

warnings.simplefilter(action='ignore', category=FutureWarning)
print_lock = threading.Lock()
def safe_print(message):
    with print_lock: print(message)

# --- 0. 配置区域 (二级分类白名单) ---
VALID_L2_MAP = {
    "分享": ["案例分享", "科普"],
    "寻求帮助": ["综合", "自闭症诊断", "医院推荐(评价)", "医生推荐(评价)", "机构推荐(评价)", "老师推荐(评价)", "干预措施", "日常生活"],
    "广告": ["广告"],
    "其他": ["其他"],
    "表意不明": ["表意不明"],
    "数据缺失": ["数据缺失"],
    "一级缺失": ["一级缺失"]
}

LETTER_MAP = {
    "A": "案例分享", "C": "自闭症诊断", "D": "干预措施", "E": "日常生活",
    "F": "医院推荐(评价)", "G": "医生推荐(评价)", "H": "机构推荐(评价)",
    "I": "老师推荐(评价)", "J": "综合"
}

# --- 1. 清洗函数 ---
def clean_and_validate_label(raw_label, primary_topic):
    if pd.isna(raw_label): return "INVALID"
    s_label = str(raw_label).strip()
    s_label = s_label.replace('"', '').replace("'", "").replace("。", "")
    
    valid_options = VALID_L2_MAP.get(primary_topic, [])
    if s_label in valid_options: return s_label
    
    cleaned = re.sub(r'^[\d\w]+\.|^Category:|^\w+\s*-\s*', '', s_label).strip()
    if cleaned in valid_options: return cleaned
    
    for option in valid_options:
        if option in s_label:
            if len(s_label) < len(option) + 10: return option

    if primary_topic == "寻求帮助" and s_label.upper() in LETTER_MAP:
        target = LETTER_MAP[s_label.upper()]
        if target in valid_options: return target

    return "INVALID"

# --- LLM 定义 ---
class Classification(BaseModel):
    secondary_topic: str = Field(description="二级主题")

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

    # 改进的错误处理，包含API密钥验证
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
        
        try:
            resp = requests.post(url, json=payload, headers=headers, timeout=180)
            resp.raise_for_status()  # 如果响应状态码不是200，会抛出HTTPError
            data = resp.json()
            return data["choices"][0]["message"]["content"]
        except requests.exceptions.HTTPError as e:
            if resp.status_code == 403:
                safe_print(f"[API Error] 403 Forbidden - API密钥无效或过期")
                safe_print(f"[API Error] 请检查API密钥是否正确或是否已过期")
                safe_print(f"[API Error] 当前使用的API密钥前几位: {self.api_secret[:20]}...")
                raise Exception(f"API密钥无效: {e}")
            elif resp.status_code == 429:
                safe_print(f"[API Error] 429 Too Many Requests - 请求频率超限")
                time.sleep(10)  # 等待更长时间再重试
                raise
            else:
                safe_print(f"[API Error] HTTP {resp.status_code}: {e}")
                raise
        except Exception as e:
            safe_print(f"[API Error] 其他错误: {e}")
            raise

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> Optional[str]:
        for attempt in range(3):
            try:
                response = self.model_completion(prompt)
                if stop is not None: response = enforce_stop_tokens(response, stop)
                return response
            except Exception as e:
                error_msg = str(e)
                safe_print(f"[LLM Error] 第{attempt+1}次尝试失败: {error_msg}")
                
                # 如果是API密钥问题，直接退出
                if "API密钥无效" in error_msg:
                    safe_print("[LLM Error] API密钥问题，跳过重试")
                    return None
                
                if attempt < 2:  # 如果不是最后一次尝试
                    wait_time = 5 * (attempt + 1)  # 递增等待时间
                    safe_print(f"[LLM Error] 等待 {wait_time} 秒后重试...")
                    time.sleep(wait_time)
                else:
                    safe_print("[LLM Error] 跳过此条数据")
                    return None
        
        return None

    def __call__(self, prompt: str, stop: Optional[List[str]] = None) -> Optional[str]:
        return self._call(prompt, stop)

PARSER = PydanticOutputParser(pydantic_object=TopicClassification)

# --- 2. Prompt ---
PROMPT_TEMPLATE = PromptTemplate(
    template="""角色：
你是一个严格遵循分类规范的中文文本分类引擎。你的目标是稳定、可复现、零随机地输出唯一正确标签。

任务：
给定一段中文帖子文本、发帖人身份（poster_role），以及已确定的一级主题 primary_topic，你必须从对应的二级主题集合中选择且仅选择一个 secondary_topic，并严格输出指定 JSON 结构，不得输出任何解释、注释或额外字段。

二级主题集合（按 primary_topic 限定，禁止创造新标签）：
当 primary_topic = 分享 时，secondary_topic 只能是：
1.案例分享
2.科普

当 primary_topic = 寻求帮助 时，secondary_topic 只能是：
1.综合
2.自闭症诊断
3.医院推荐(评价)
4.医生推荐(评价)
5.机构推荐(评价)
6.老师推荐(评价)
7.干预措施
8.日常生活

当 primary_topic = 广告 时，secondary_topic 必须是：广告
当 primary_topic = 其他 时，secondary_topic 必须是：其他
当 primary_topic = 表意不明 时，secondary_topic 必须是：表意不明

二级主题操作化定义（用于 primary_topic = 寻求帮助 / 分享 的细分）：
A. 分享-案例分享：以个人/家庭/孩子的具体经历为主线（诊断过程、干预经历、教育体验、生活记录），主要目的为经验交流与记录，不以提问求助为核心，没有分享具体内容仅是说要谈谈孩子的不算，没有明显的求助意图，有分享故事出来相互交流或者发表自己感想的意图。
B. 分享-科普：以知识解释、资料整理、链接/图片资源分享为主，语气相对客观中性，不以引流推广为目的。
C. 寻求帮助-自闭症诊断：描述症状或发展史，请他人协助判断是否为自闭症，或咨询评估、量表、诊断流程与结果理解（如"是不是自闭症""帮忙判断""评估/量表/诊断"）。
D. 寻求帮助-干预措施：患者或家属通过描述症状、行为表现或既往情况，寻求他人协助判断是否为自闭症，或咨询评估、诊断相关问题，但不包括预后结果，询问自闭症相关的检查项目等也算，求助意图较明显。
E. 寻求帮助-日常生活：用户描述患者在日常生活中的行为或照护问题，家庭关系方面的问题，如奶奶带还是妈妈带，家里人和孩子的相处问题等并寻求应对建议。
F. 寻求帮助-医院推荐(评价)：以选择医院/医疗机构为核心对象，咨询其权威性、挂号流程、诊断能力、服务体验等。
G. 寻求帮助-医生推荐(评价)：以选择具体医生为核心对象，咨询其专业能力、就诊体验、沟通态度、经验与口碑等。
H. 寻求帮助-机构推荐(评价)：以选择康复机构/培训组织/托管机构为核心对象，咨询其专业性、口碑、费用、师资、是否"踩坑"等。
I. 寻求帮助-老师推荐(评价)：以选择具体康复老师/语言治疗师/行为分析师/特教老师为核心对象，咨询个人能力、风格、经验与评价。
J. 寻求帮助-综合：帖子存在明确求助意图，但内容不符合该类别中的任何具体子类，或涉及自闭症方面、其他疾病的诊断、求职、最后表明需要添加联系方式关于自闭症问题相互鼓励等多个问题混杂，又包含日常生活又包含症状诊断，难以进一步细分，或帖子在咨询预后的影响和结果。。

特殊情况下的判定规则（必须执行，用于降低随机性）：
1.单标签原则：无论文本涉及多少方面，你只能输出一个 secondary_topic。
2.多重求助（primary referent 规则）：当同一帖文同时请求医院推荐与医生推荐，或同时涉及多个"寻求帮助"子主题时，必须先判断主要关注对象（primary referent），即文本中被反复强调、描述更具体、或作为最终决策焦点的对象：
A.主要关注对象是医院/医疗机构 -> 医院推荐(评价)
B.主要关注对象是具体医生 -> 医生推荐(评价)
C.主要关注对象是康复机构/中心/组织 -> 组织推荐(评价)
D.主要关注对象是具体老师/治疗师 -> 老师推荐(评价)
E.主要关注对象是"怎么做训练/用什么方法" -> 干预措施
F.主要关注对象是生活照护困难 -> 日常生活
G.若无法判断主次或多个对象同等重要 -> 综合
H.关键词只作线索，必须以"主要关注对象"和"核心意图"作为最终判定依据。

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
    df_chunk, api_secret, worker_id, is_rerun = args  # 新增 is_rerun 参数
    llm = ModelChat(api_secret)
    df_chunk_processed = df_chunk.copy()
    
    for i, index in enumerate(df_chunk_processed.index):
        # 如果是错分重标任务，特殊处理
        if is_rerun:
            safe_print(f"[Worker {worker_id}] 重新标注错分行: 索引 {index}")
        
        original_identity = df_chunk_processed.at[index, 'identity_new']
        original_title = df_chunk_processed.at[index, 'Title']
        original_content = df_chunk_processed.at[index, 'Content']
        primary_topic_val = df_chunk_processed.at[index, 'primary_topic']
        
        identity_str = str(original_identity) if pd.notna(original_identity) else "未知"
        title_str = str(original_title) if pd.notna(original_title) else ""
        content_str = str(original_content) if pd.notna(original_content) else ""
        
        if pd.isna(primary_topic_val) or str(primary_topic_val).strip() == "" or str(primary_topic_val).lower() == 'nan':
            df_chunk_processed.at[index, 'secondary_topic'] = "一级缺失"
            continue
        primary_topic_str = str(primary_topic_val).strip()

        if not title_str.strip() and not content_str.strip():
            df_chunk_processed.at[index, 'secondary_topic'] = "数据缺失"
            continue
            
        try:
            combined_text = f"帖子标题：{title_str}\n帖子内容：{content_str}"
            prompt = PROMPT_TEMPLATE.format(poster_role=identity_str, primary_topic=primary_topic_str, text=combined_text)
        except Exception as e:
            safe_print(f"[Worker {worker_id}] Prompt Error: {e}")
            continue
            
        retry_count = 0
        final_label = "INVALID"
        
        # 限制为内部重试最多2次 (retry_count <= 1)
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
                    raw_secondary = parsed_output.classification.secondary_topic
                    
                    validated = clean_and_validate_label(raw_secondary, primary_topic_str)
                    
                    if validated != "INVALID":
                        final_label = validated
                        break 
                    else:
                        safe_print(f"[Worker {worker_id}] 非法标签: '{raw_secondary}'，重试中...")
                        retry_count += 1
            except Exception as e:
                error_msg = str(e)
                if "API密钥无效" in error_msg:
                    safe_print(f"[Worker {worker_id}] API密钥无效，停止处理")
                    raise e  # 重新抛出异常，让上层处理
                retry_count += 1
                safe_print(f"[Worker {worker_id}] 错误: {error_msg}")
        
        if final_label == "INVALID":
            df_chunk_processed.at[index, 'secondary_topic'] = np.nan 
        else:
            df_chunk_processed.at[index, 'secondary_topic'] = final_label

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

def test_api_key(api_key):
    """测试API密钥是否有效"""
    safe_print("正在测试API密钥...")
    url = "https://api.siliconflow.cn/v1/models"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    try:
        resp = requests.get(url, headers=headers, timeout=10)
        if resp.status_code == 200:
            safe_print("✓ API密钥有效")
            return True
        elif resp.status_code == 403:
            safe_print("✗ API密钥无效或已过期")
            return False
        else:
            safe_print(f"? 未知响应: HTTP {resp.status_code}")
            return False
    except Exception as e:
        safe_print(f"! 连接测试失败: {e}")
        return False

# --- 5. 主程序 ---
if __name__ == "__main__":
    # 指向一级分类结果文件 (Level 1 Result)
    FILE_PATH = '社区数据/LLM标注分类/百度贴吧/重制版/百度贴吧数据标注_deepseek-7b-f-3.xlsx' 
    SHEET_NAME = "Sheet1" 
    CHECKPOINT_BATCH_SIZE = 50 
    
    base, ext = os.path.splitext(FILE_PATH)
    # >>> 修改此处文件名以区分不同进程 <<<
    OUTPUT_FILE_PATH = f"{base}_Level2{ext}" 
    
    print(f"读取源: {FILE_PATH}")
    print(f"输出到: {OUTPUT_FILE_PATH}")
    
    # 提供多个API密钥选项
    api_keys_to_try = [
        "sk-fgyebgcznjslykwnetybtxvtjkgrqtsmprascddqhcgmkdrp",  # 原来的密钥
        # 可以添加更多备用密钥
        # "sk-你的第二个API密钥",
        # "sk-你的第三个API密钥",
    ]
    
    # 测试API密钥
    valid_api_keys = []
    for api_key in api_keys_to_try:
        if test_api_key(api_key):
            valid_api_keys.append(api_key)
    
    if not valid_api_keys:
        safe_print("错误：没有有效的API密钥！")
        safe_print("请按照以下步骤操作：")
        safe_print("1. 访问 https://cloud.siliconflow.cn/")
        safe_print("2. 注册并登录账户")
        safe_print("3. 在控制台创建API密钥")
        safe_print("4. 确保账户有足够的余额")
        safe_print("5. 将新的API密钥替换到代码中")
        exit(1)
    
    safe_print(f"找到 {len(valid_api_keys)} 个有效的API密钥")
    api_secrets = valid_api_keys
    num_workers = len(api_secrets)

    if not os.path.exists(FILE_PATH): exit("一级分类文件不存在，请先运行一级分类代码")

    df_source = pd.read_excel(FILE_PATH, sheet_name=SHEET_NAME, engine='openpyxl')
    df_target = df_source.copy()
    
    # 检查错分类型列是否存在
    has_misclassification_col = '错分类型' in df_target.columns
    
    if os.path.exists(OUTPUT_FILE_PATH):
        prog = _load_progress(OUTPUT_FILE_PATH, SHEET_NAME)
        if prog is not None and 'secondary_topic' in prog.columns:
            df_target['secondary_topic'] = prog['secondary_topic']
            print("已加载旧进度。")
    
    if 'secondary_topic' not in df_target.columns: df_target['secondary_topic'] = np.nan

    # 创建测试数据来验证API连接
    safe_print("\n测试API连接...")
    test_llm = ModelChat(api_secrets[0])
    test_prompt = "请回复'测试成功'四个字"
    try:
        test_result = test_llm(test_prompt)
        if test_result:
            safe_print("✓ API连接测试成功")
        else:
            safe_print("✗ API连接测试失败，但将继续尝试")
    except Exception as e:
        safe_print(f"! API连接测试异常: {e}")
        safe_print("将继续运行，但可能会失败")
    
    # --- 第一步：处理错分类型为1或2的数据行 ---
    if has_misclassification_col:
        print("\n" + "="*50)
        print("第一步：处理错分类型为1或2的数据行（二级分类重标）")
        print("="*50)
        
        # 找出错分类型为1或2的数据行
        misclassified_indices = df_target[df_target['错分类型'].isin([1, 2])].index
        misclassified_count = len(misclassified_indices)
        
        # 统计不同类型数量
        type1_count = len(df_target[df_target['错分类型'] == 1])
        type2_count = len(df_target[df_target['错分类型'] == 2])
        
        print(f"发现 {misclassified_count} 行需要重新标注的错分数据")
        print(f"  其中错分类型为1（一级分类错误）: {type1_count} 行")
        print(f"  其中错分类型为2（二级分类错误）: {type2_count} 行")
        
        if misclassified_count > 0:
            # 重置这些行的 secondary_topic 为 NaN，以便后续重新标注
            df_target.loc[misclassified_indices, 'secondary_topic'] = np.nan
            
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
                if "API密钥无效" in str(e):
                    safe_print("API密钥问题，程序终止")
                    exit(1)
    
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
            p_val = row.get('primary_topic', '')
            if pd.isna(p_val) or str(p_val).strip() == "": return False 
            
            val = row.get('secondary_topic', np.nan)
            if pd.isna(val) or str(val).lower() == 'nan': return True
            if clean_and_validate_label(val, str(p_val).strip()) == "INVALID": return True
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

        except KeyboardInterrupt: 
            safe_print("程序被用户中断")
            break
        except Exception as e: 
            print(f"Error: {e}")
            if "API密钥无效" in str(e):
                safe_print("API密钥问题，程序终止")
                break
    
    _save_checkpoint(df_target, OUTPUT_FILE_PATH, SHEET_NAME)
    
    # 统计最终结果
    print("\n" + "="*50)
    print("二级分类完成！统计结果：")
    print("="*50)
    print(f"总数据行数: {len(df_target)}")
    print(f"已标注行数: {df_target['secondary_topic'].notna().sum()}")
    print(f"未标注行数: {df_target['secondary_topic'].isna().sum()}")
    
    if has_misclassification_col:
        # 统计不同类型的错分数据
        type1_final = df_target[df_target['错分类型'] == 1]
        type2_final = df_target[df_target['错分类型'] == 2]
        
        print(f"错分类型为1的行数: {len(type1_final)}")
        print(f"  其中已重新标注的行数: {type1_final['secondary_topic'].notna().sum()}")
        print(f"错分类型为2的行数: {len(type2_final)}")
        print(f"  其中已重新标注的行数: {type2_final['secondary_topic'].notna().sum()}")
        
        # 总错分数据统计
        all_misclassified = df_target[df_target['错分类型'].isin([1, 2])]
        print(f"总错分数据行数（类型1或2）: {len(all_misclassified)}")
        print(f"  其中已重新标注的行数: {all_misclassified['secondary_topic'].notna().sum()}")
    
    print("二级分类完成。")