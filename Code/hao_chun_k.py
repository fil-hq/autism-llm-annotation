import pandas as pd
import os
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score

# ================= 1. 配置区域 =================

# 你的Excel文件夹路径
folder_path = r'社区数据/百度贴吧' 

# 文件名前缀
file_prefix = '百度贴吧数据标注v'

# 模型标识符列表 (对应文件名中间部分)
model_identifiers = [
    #'x-7b',
    #'x-14b',
    #'x-32b',
    #'x-671b',
    #'m-7b',
    #'m-14b',
    #'m-32b',
    #'m-671b',
    '6.4_去重'
    #'test_ma-fei'
    # ... 请补全你的8个模型ID
]

# 指定列名
col_human = 'identity'      # 人工标注
col_model = 'identity_new'

# ================= 2. 核心逻辑 =================

def get_file_path(folder, prefix, model_id):
    """
    构造文件名
    情况 A: 如果文件名依然保留了 "-1" (如: ..._deepseek-7b-f-1.xlsx)
    """
    #filename = f"{prefix}_{model_id}-1.xlsx"
    
    # 情况 B: 如果文件名没有 "-1" 了 (如: ..._deepseek-7b-f.xlsx)
    # 请取消下面这行的注释，并注释掉上面那行
    filename = f"{prefix}_{model_id}.xlsx"
    
    return os.path.join(folder, filename)

def calc_single_run_metrics(file_path, model_name):
    # 1. 读取文件
    if not os.path.exists(file_path):
        print(f"[警告] 文件不存在: {file_path}")
        return None
        
    try:
        df = pd.read_excel(file_path)
    except Exception as e:
        print(f"[错误] 无法读取 {file_path}: {e}")
        return None

    # 2. 检查列
    if col_human not in df.columns or col_model not in df.columns:
        print(f"[错误] 模型 {model_name} 的文件中缺少列 '{col_human}' 或 '{col_model}'")
        return None

    # 3. 数据清洗
    df_clean = df[[col_human, col_model]].dropna().astype(str)
    y_true = df_clean[col_human]
    y_pred = df_clean[col_model]

    if len(y_true) == 0:
        print(f"[警告] 模型 {model_name} 有效数据为空")
        return None

    # 4. 计算指标
    acc = accuracy_score(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro')

    return {
        'Model': model_name,
        'Accuracy': f"{acc:.2%}",      # 百分比格式
        'Kappa': f"{kappa:.4f}",       # 保留4位小数
        'F1 Score': f"{f1:.4f}"        # 保留4位小数
    }

# ================= 3. 执行分析 =================

results = []

print("开始处理单次运行数据...\n")

for mid in model_identifiers:
    # 获取路径
    fpath = get_file_path(folder_path, file_prefix, mid)
    
    # 计算
    metrics = calc_single_run_metrics(fpath, mid)
    
    if metrics:
        results.append(metrics)

# ================= 4. 输出结果 =================

if results:
    df_results = pd.DataFrame(results)
    
    print("\n" + "="*40)
    print("最终评测报告 (Single Run)")
    print("="*40)
    print(df_results.to_string(index=False))
    
    # 保存
    output_file = '社区数据/百度贴吧/身份.xlsx'
    df_results.to_excel(output_file, index=False)
    print(f"\n结果已保存至: {output_file}")
else:
    print("没有生成任何有效结果，请检查文件路径配置。")