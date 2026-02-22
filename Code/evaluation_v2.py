import pandas as pd
import numpy as np
import os
import itertools
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score, confusion_matrix
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# ================= 0. 中文字体设置 =================
def setup_chinese_fonts():
    """自动设置中文字体"""
    try:
        import platform
        
        # 根据操作系统选择字体
        system = platform.system()
        
        if system == 'Windows':
            font_names = ['SimHei', 'Microsoft YaHei', 'SimSun', 'FangSong', 'KaiTi']
        elif system == 'Darwin':
            font_names = ['Arial Unicode MS', 'STHeiti', 'Hiragino Sans GB', 'PingFang SC']
        else:
            font_names = ['DejaVu Sans', 'WenQuanYi Micro Hei', 'AR PL UMing CN']
        
        # 获取系统可用字体
        available_fonts = matplotlib.font_manager.get_font_names()
        
        # 查找可用的中文字体
        for font_name in font_names:
            if font_name in available_fonts:
                matplotlib.rcParams['font.sans-serif'] = [font_name]
                matplotlib.rcParams['axes.unicode_minus'] = False
                print(f"已设置中文字体: {font_name}")
                return True
        
        print("警告: 未找到合适的中文字体，将使用默认字体")
        return False
        
    except Exception as e:
        print(f"设置中文字体时出错: {e}")
        return False

# 初始化中文字体
setup_chinese_fonts()

# ================= 1. 配置区域 (请仔细修改这里) =================

# 你的Excel文件夹路径
folder_path = '社区数据/人标注/抽样-百度贴吧/test1' 

# 文件名前缀 (社区名称)
file_prefix = '百度贴吧数据标注' 

# 这里的列表中填入你 8 个模型的"中间部分名称"
model_identifiers = [
    'deepseek-7b-t',   # 模型1 (False模式)
    'deepseek-14b-t',  # 模型2
    'deepseek-32b-t',  # 模型3
    #'deepseek-671b-f',          # 模型4
    'deepseek-671b-t',          # 模型5

]

# Excel中的列名
columns_mapping = {
    'human_l1': '标签',
    'human_l2': '二级标签',
    'model_l1': 'primary_topic',
    'model_l2': 'secondary_topic'
}

# ================= 2. 新增功能：创建详细错分分析文件 =================

def create_detailed_error_analysis(df, model_name, run_num):
    """
    创建详细的错分分析文件，包含错分类型标注
    
    Args:
        df: 包含标注数据的DataFrame
        model_name: 模型名称
        run_num: 运行次数
        
    Returns:
        error_df: 包含错分分析的新DataFrame
    """
    # 创建副本
    error_df = df.copy()
    
    # 初始化错分类型列
    error_df['错分类型'] = 0  # 0表示正确，1表示一级错分，2表示二级错分
    error_df['模型'] = model_name  # 添加模型列
    error_df['运行次数'] = run_num  # 添加运行次数列
    
    # 检查一级标签错分
    human_l1 = columns_mapping['human_l1']
    model_l1 = columns_mapping['model_l1']
    
    if human_l1 in df.columns and model_l1 in df.columns:
        # 标记一级标签错分
        l1_mask = (df[human_l1].astype(str) != df[model_l1].astype(str))
        error_df.loc[l1_mask, '错分类型'] = 1
    
    # 检查二级标签错分（只在二级标签都有值且一级标签正确的情况下检查）
    human_l2 = columns_mapping['human_l2']
    model_l2 = columns_mapping['model_l2']
    
    if human_l2 in df.columns and model_l2 in df.columns:
        # 创建掩码：一级标签正确且二级标签都有值
        l1_correct_mask = (error_df['错分类型'] == 0)
        l2_not_null_mask = (df[human_l2].notna()) & (df[model_l2].notna())
        
        # 标记二级标签错分（只在二级标签都有值且一级标签正确的情况下）
        l2_mask = l1_correct_mask & l2_not_null_mask & (df[human_l2].astype(str) != df[model_l2].astype(str))
        error_df.loc[l2_mask, '错分类型'] = 2
    
    # 添加错分详情列
    error_df['错分详情'] = ''
    
    # 填充错分详情
    for idx, row in error_df.iterrows():
        error_type = row['错分类型']
        if error_type == 1:
            error_df.at[idx, '错分详情'] = f"一级错分: 人工标注={row.get(human_l1, '')}, 模型预测={row.get(model_l1, '')}"
        elif error_type == 2:
            error_df.at[idx, '错分详情'] = f"二级错分: 人工标注={row.get(human_l2, '')}, 模型预测={row.get(model_l2, '')}"
        else:
            if pd.notna(row.get(human_l2, '')) and pd.notna(row.get(model_l2, '')):
                error_df.at[idx, '错分详情'] = f"完全正确: 一级标签{row.get(human_l1, '')}, 二级标签{row.get(human_l2, '')}"
            else:
                error_df.at[idx, '错分详情'] = f"正确: 一级标签{row.get(human_l1, '')}"
    
    return error_df

def filter_l2_correct_data(df, col_true_l1, col_pred_l1, col_true_l2, col_pred_l2):
    """
    筛选出一级标签正确且二级标签有值的数据
    
    Args:
        df: 原始DataFrame
        col_true_l1: 人工一级标签列名
        col_pred_l1: 模型一级标签列名
        col_true_l2: 人工二级标签列名
        col_pred_l2: 模型二级标签列名
        
    Returns:
        filtered_df: 筛选后的DataFrame，只包含一级标签正确且二级标签有值的行
    """
    # 确保是字符串类型并处理空值
    df_filtered = df.copy()
    
    # 一级标签正确
    l1_correct_mask = (df_filtered[col_true_l1].astype(str) == df_filtered[col_pred_l1].astype(str))
    
    # 二级标签都有值
    l2_not_null_mask = (df_filtered[col_true_l2].notna()) & (df_filtered[col_pred_l2].notna())
    
    # 组合条件
    final_mask = l1_correct_mask & l2_not_null_mask
    
    return df_filtered[final_mask]

# ================= 3. 修改后的工具函数 =================

def get_file_path(folder, prefix, model_id, run):
    """根据规则构造文件名"""
    filename = f"{prefix}_{model_id}-{run}.xlsx"
    return os.path.join(folder, filename)

def calc_metrics(df, col_true, col_pred):
    """计算单次运行的指标"""
    # 清洗数据：转为字符串并去空
    df_clean = df[[col_true, col_pred]].dropna().astype(str)
    y_true = df_clean[col_true]
    y_pred = df_clean[col_pred]
    
    if len(y_true) == 0:
        return None, None, None
        
    # 如果只有一个类别，kappa设为0
    unique_labels = set(y_true) | set(y_pred)
    if len(unique_labels) < 2:
        kappa = 0.0
    else:
        kappa = cohen_kappa_score(y_true, y_pred)
    
    return {
        'acc': accuracy_score(y_true, y_pred),
        'kappa': kappa,
        'f1': f1_score(y_true, y_pred, average='macro')
    }, y_true, y_pred

def analyze_misclassifications_with_type(df_error, level_name):
    """
    分析带错分类型的错分情况
    
    Args:
        df_error: 包含错分类型标记的DataFrame
        level_name: 'l1'或'l2'
        
    Returns:
        misclass_df: 错分分析DataFrame
        misclass_matrix: 错分矩阵
    """
    misclassifications = []
    
    if level_name == 'l1':
        true_col = columns_mapping['human_l1']
        pred_col = columns_mapping['model_l1']
        error_type_filter = 1  # 一级错分
    else:
        true_col = columns_mapping['human_l2']
        pred_col = columns_mapping['model_l2']
        error_type_filter = 2  # 二级错分
    
    # 获取所有类别
    all_labels = sorted(set(df_error[true_col].dropna().astype(str)) | 
                       set(df_error[pred_col].dropna().astype(str)))
    
    # 创建错分矩阵
    misclass_matrix = defaultdict(lambda: defaultdict(int))
    
    # 筛选特定类型的错分
    error_rows = df_error[df_error['错分类型'] == error_type_filter]
    
    for _, row in error_rows.iterrows():
        true_label = str(row[true_col]) if pd.notna(row[true_col]) else '未知'
        pred_label = str(row[pred_col]) if pd.notna(row[pred_col]) else '未知'
        
        if true_label != pred_label:
            misclass_matrix[true_label][pred_label] += 1
    
    # 转换为DataFrame便于分析
    misclass_data = []
    for true_label in misclass_matrix:
        for pred_label in misclass_matrix[true_label]:
            misclass_data.append({
                '真实标签': true_label,
                '预测标签': pred_label,
                '错分次数': misclass_matrix[true_label][pred_label],
                '错分类型': f"{level_name}级错分"
            })
    
    # 按错分次数排序
    misclass_df = pd.DataFrame(misclass_data)
    if not misclass_df.empty:
        misclass_df = misclass_df.sort_values('错分次数', ascending=False)
    
    return misclass_df, misclass_matrix

def plot_confusion_matrix_chinese(cm, labels, model_name, level_name, run_num=None):
    """绘制支持中文的混淆矩阵热图"""
    plt.figure(figsize=(max(10, len(labels)), max(8, len(labels))))
    
    # 创建DataFrame
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    
    # 计算百分比
    if cm.sum() > 0:
        cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm_percent = np.nan_to_num(cm_percent)
    else:
        cm_percent = cm.astype('float')
    
    # 创建掩码
    mask = cm_percent < 0.01
    
    try:
        ax = sns.heatmap(cm_percent, 
                        annot=True, 
                        fmt='.2f', 
                        cmap='Blues',
                        xticklabels=labels, 
                        yticklabels=labels,
                        cbar_kws={'label': '比例'},
                        mask=mask if np.any(mask) else None,
                        annot_kws={'size': 10})
        
        try:
            title = f'{model_name} - {level_name}'
            if run_num:
                title += f' (运行{run_num})'
            plt.title(title, fontsize=16)
            plt.xlabel('预测标签', fontsize=12)
            plt.ylabel('真实标签', fontsize=12)
        except:
            plt.title(f'{model_name} - {level_name}' + (f' (Run{run_num})' if run_num else ''), fontsize=16)
            plt.xlabel('Predicted Label', fontsize=12)
            plt.ylabel('True Label', fontsize=12)
            
    except Exception as e:
        print(f"绘制热图时出错: {e}")
        plt.imshow(cm_percent, cmap='Blues', interpolation='nearest')
        plt.colorbar()
        plt.xticks(range(len(labels)), labels, rotation=45)
        plt.yticks(range(len(labels)), labels)
        plt.title(f'{model_name} - {level_name}' + (f' (Run{run_num})' if run_num else ''))
    
    plt.tight_layout()
    
    filename = f"混淆矩阵_{model_name}_{level_name}"
    if run_num:
        filename += f"_run{run_num}"
    filename += ".png"
    
    save_path = os.path.join(folder_path, filename)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"混淆矩阵已保存: {save_path}")
    return save_path, cm_df

# ================= 4. 主循环 (修改版) =================

# 创建保存图片的目录
os.makedirs(folder_path, exist_ok=True)

summary_l1 = []
summary_l2 = []
all_misclassifications_l1 = []
all_misclassifications_l2 = []
all_error_details = []  # 存储所有详细错分信息

print(f"正在分析 {len(model_identifiers)} 个模型配置...\n")

for mid in model_identifiers:
    dfs = []  # 存储原始数据
    error_dfs = []  # 存储每个运行的错分分析DataFrame
    
    # 读取3次运行的文件
    for run in [1, 2, 3]:
        fpath = get_file_path(folder_path, file_prefix, mid, run)
        
        if os.path.exists(fpath):
            try:
                # 读取所有列（因为我们需要完整数据）
                df = pd.read_excel(fpath)
                dfs.append(df)
                
                # 创建详细错分分析
                error_df = create_detailed_error_analysis(df, mid, run)
                error_dfs.append(error_df)
                
                # 保存错分分析文件
                error_filename = f"{file_prefix}_{mid}-{run}_错分分析.xlsx"
                error_save_path = os.path.join(folder_path, error_filename)
                error_df.to_excel(error_save_path, index=False)
                print(f"✓ 已保存错分分析: {error_filename}")
                
            except Exception as e:
                print(f"✗ 处理文件 {os.path.basename(fpath)} 时出错: {e}")
                # 如果出错，添加一个空的DataFrame以保持索引一致
                dfs.append(pd.DataFrame())
                error_dfs.append(pd.DataFrame())
        else:
            print(f"⚠ 文件缺失: {os.path.basename(fpath)}")
            # 文件缺失时添加空DataFrame
            dfs.append(pd.DataFrame())
            error_dfs.append(pd.DataFrame())
    
    # 检查是否有有效数据
    valid_dfs = [df for df in dfs if not df.empty]
    if len(valid_dfs) < 2:
        print(f"→ 模型 {mid} 的有效文件不足 2 个，跳过。")
        continue

    print(f"\n{'='*60}")
    print(f"分析模型: {mid}")
    print(f"{'='*60}")

    # 分析一级标签
    print("\n【一级标签分析】")
    col_true_l1 = columns_mapping['human_l1']
    col_pred_l1 = columns_mapping['model_l1']
    
    metrics_list_l1 = []
    all_y_true_l1 = []
    all_y_pred_l1 = []
    
    for run_idx in range(1, 4):
        # 跳过空的DataFrame
        if run_idx > len(dfs) or dfs[run_idx-1].empty or run_idx > len(error_dfs) or error_dfs[run_idx-1].empty:
            print(f"\n运行 {run_idx}: 数据为空或缺失，跳过")
            continue
            
        df = dfs[run_idx-1]
        error_df = error_dfs[run_idx-1]
        
        print(f"\n运行 {run_idx}:")
        
        # 统计错分类型
        if '错分类型' in error_df.columns:
            error_stats = error_df['错分类型'].value_counts()
            print(f"  错分统计:")
            for err_type, count in error_stats.items():
                if err_type == 0:
                    print(f"    正确: {count} 条")
                elif err_type == 1:
                    print(f"    一级错分: {count} 条")
                elif err_type == 2:
                    print(f"    二级错分: {count} 条")
        
        m, y_true, y_pred = calc_metrics(df, col_true_l1, col_pred_l1)
        if m:
            metrics_list_l1.append(m)
            all_y_true_l1.append(y_true)
            all_y_pred_l1.append(y_pred)
            
            print(f"  准确率: {m['acc']:.2%}")
            print(f"  Kappa: {m['kappa']:.3f}")
            print(f"  F1: {m['f1']:.3f}")
            
            # 使用带错分类型的数据进行分析
            if not error_df.empty:
                misclass_df, _ = analyze_misclassifications_with_type(error_df, 'l1')
                if not misclass_df.empty:
                    print(f"  一级错分Top 5:")
                    print(misclass_df.head().to_string(index=False))
            
            # 计算混淆矩阵
            labels = sorted(set(y_true) | set(y_pred))
            cm = confusion_matrix(y_true, y_pred, labels=labels)
            
            # 绘制混淆矩阵
            plot_confusion_matrix_chinese(cm, labels, mid, "一级标签", run_idx)
    
    if metrics_list_l1:
        df_metrics_l1 = pd.DataFrame(metrics_list_l1)
        
        # 使用第一个有效的error_df来计算统计
        first_valid_error_df = None
        for err_df in error_dfs:
            if not err_df.empty and '错分类型' in err_df.columns:
                first_valid_error_df = err_df
                break
        
        if first_valid_error_df is not None:
            total_samples = len(first_valid_error_df)
            l1_correct_count = first_valid_error_df[first_valid_error_df['错分类型'] == 0].shape[0]
            l1_error_count = first_valid_error_df[first_valid_error_df['错分类型'] == 1].shape[0]
            l1_correct_rate = l1_correct_count / total_samples if total_samples > 0 else 0
            l1_error_rate = l1_error_count / total_samples if total_samples > 0 else 0
        else:
            total_samples = 0
            l1_correct_rate = 0
            l1_error_rate = 0
        
        summary_l1.append({
            'Model': mid,
            'Avg_Acc': f"{df_metrics_l1['acc'].mean():.2%} ± {df_metrics_l1['acc'].std():.2%}",
            'Avg_Kappa': f"{df_metrics_l1['kappa'].mean():.3f} ± {df_metrics_l1['kappa'].std():.3f}",
            'Avg_F1': f"{df_metrics_l1['f1'].mean():.3f} ± {df_metrics_l1['f1'].std():.3f}",
            '一级准确率': f"{l1_correct_rate:.2%}",
            '一级错分率': f"{l1_error_rate:.2%}",
            '有效运行次数': len(metrics_list_l1)
        })
        
        # 合并所有运行进行整体分析
        if all_y_true_l1:
            # 合并错分数据
            valid_error_dfs = [err_df for err_df in error_dfs if not err_df.empty]
            if valid_error_dfs:
                combined_error = pd.concat(valid_error_dfs, ignore_index=True)
                all_error_details.append(combined_error)
                
                # 整体混淆矩阵
                combined_y_true_l1 = pd.concat(all_y_true_l1)
                combined_y_pred_l1 = pd.concat(all_y_pred_l1)  # 修复：使用正确的变量名
                labels = sorted(set(combined_y_true_l1) | set(combined_y_pred_l1))  # 修复：使用正确的变量名
                if len(labels) > 0:
                    overall_cm = confusion_matrix(combined_y_true_l1, combined_y_pred_l1, labels=labels)
                    plot_confusion_matrix_chinese(overall_cm, labels, mid, "一级标签_整体")
                
                # 整体错分分析
                misclass_df, _ = analyze_misclassifications_with_type(combined_error, 'l1')
                if not misclass_df.empty:
                    all_misclassifications_l1.append(misclass_df)
    
    # 分析二级标签（只基于一级标签正确的数据）
    print("\n【二级标签分析】")
    col_true_l2 = columns_mapping['human_l2']
    col_pred_l2 = columns_mapping['model_l2']
    
    metrics_list_l2 = []
    all_y_true_l2 = []
    all_y_pred_l2 = []
    
    for run_idx in range(1, 4):
        # 跳过空的DataFrame
        if run_idx > len(dfs) or dfs[run_idx-1].empty or run_idx > len(error_dfs) or error_dfs[run_idx-1].empty:
            print(f"\n运行 {run_idx}: 数据为空或缺失，跳过")
            continue
            
        df = dfs[run_idx-1]
        error_df = error_dfs[run_idx-1]
        
        print(f"\n运行 {run_idx}:")
        
        # 检查必要的列是否存在
        if col_true_l1 not in df.columns or col_pred_l1 not in df.columns:
            print(f"  警告: 一级标签列不存在，跳过")
            continue
            
        if col_true_l2 not in df.columns or col_pred_l2 not in df.columns:
            print(f"  警告: 二级标签列不存在，跳过")
            continue
        
        # 筛选一级标签正确且二级标签都有值的数据
        l2_filtered_df = filter_l2_correct_data(df, col_true_l1, col_pred_l1, col_true_l2, col_pred_l2)
        
        if len(l2_filtered_df) == 0:
            print(f"  警告: 没有找到一级标签正确且二级标签都有值的数据")
            continue
        
        print(f"  有效二级标签样本数: {len(l2_filtered_df)}")
        
        # 计算二级标签指标
        m, y_true, y_pred = calc_metrics(l2_filtered_df, col_true_l2, col_pred_l2)
        if m:
            metrics_list_l2.append(m)
            all_y_true_l2.append(y_true)
            all_y_pred_l2.append(y_pred)
            
            print(f"  准确率: {m['acc']:.2%}")
            print(f"  Kappa: {m['kappa']:.3f}")
            print(f"  F1: {m['f1']:.3f}")
        
        # 分析二级错分
        if not error_df.empty:
            misclass_df, _ = analyze_misclassifications_with_type(error_df, 'l2')
            if not misclass_df.empty:
                print(f"  二级错分Top 5:")
                print(misclass_df.head().to_string(index=False))
    
    # 添加二级标签评估结果
    if metrics_list_l2:
        df_metrics_l2 = pd.DataFrame(metrics_list_l2)
        
        # 计算整体二级标签指标
        if all_y_true_l2:
            # 合并所有有效二级标签数据
            combined_y_true_l2 = pd.concat(all_y_true_l2)
            combined_y_pred_l2 = pd.concat(all_y_pred_l2)
            
            # 计算整体准确率
            overall_acc = accuracy_score(combined_y_true_l2, combined_y_pred_l2)
            
            # 计算kappa（处理单类别情况）
            unique_labels = set(combined_y_true_l2) | set(combined_y_pred_l2)
            if len(unique_labels) < 2:
                overall_kappa = 0.0
            else:
                overall_kappa = cohen_kappa_score(combined_y_true_l2, combined_y_pred_l2)
                
            overall_f1 = f1_score(combined_y_true_l2, combined_y_pred_l2, average='weighted')
            
            # 绘制整体混淆矩阵
            labels_l2 = sorted(set(combined_y_true_l2) | set(combined_y_pred_l2))
            if len(labels_l2) > 1:  # 确保有多个类别
                overall_cm_l2 = confusion_matrix(combined_y_true_l2, combined_y_pred_l2, labels=labels_l2)
                plot_confusion_matrix_chinese(overall_cm_l2, labels_l2, mid, "二级标签_整体")
            
            # 计算二级标签总体错分率（基于所有数据）
            first_valid_error_df = None
            for err_df in error_dfs:
                if not err_df.empty and '错分类型' in err_df.columns:
                    first_valid_error_df = err_df
                    break
            
            if first_valid_error_df is not None:
                total_samples = len(first_valid_error_df)
                l2_error_count = first_valid_error_df[first_valid_error_df['错分类型'] == 2].shape[0]
                l2_error_rate = l2_error_count / total_samples if total_samples > 0 else 0
            else:
                l2_error_rate = 0
            
            summary_l2.append({
                'Model': mid,
                'Avg_Acc': f"{df_metrics_l2['acc'].mean():.2%} ± {df_metrics_l2['acc'].std():.2%}",
                'Avg_Kappa': f"{df_metrics_l2['kappa'].mean():.3f} ± {df_metrics_l2['kappa'].std():.3f}",
                'Avg_F1': f"{df_metrics_l2['f1'].mean():.3f} ± {df_metrics_l2['f1'].std():.3f}",
                '整体Acc': f"{overall_acc:.2%}",
                '整体Kappa': f"{overall_kappa:.3f}",
                '整体F1': f"{overall_f1:.3f}",
                '二级错分率': f"{l2_error_rate:.2%}",
                '有效样本数': f"{len(combined_y_true_l2)}",
                '有效运行次数': len(metrics_list_l2)
            })
        else:
            summary_l2.append({
                'Model': mid,
                'Avg_Acc': 'N/A',
                'Avg_Kappa': 'N/A',
                'Avg_F1': 'N/A',
                '整体Acc': 'N/A',
                '整体Kappa': 'N/A',
                '整体F1': 'N/A',
                '二级错分率': 'N/A',
                '有效样本数': '0',
                '有效运行次数': '0'
            })
    else:
        summary_l2.append({
            'Model': mid,
            'Avg_Acc': 'N/A',
            'Avg_Kappa': 'N/A',
            'Avg_F1': 'N/A',
            '整体Acc': 'N/A',
            '整体Kappa': 'N/A',
            '整体F1': 'N/A',
            '二级错分率': 'N/A',
            '有效样本数': '0',
            '有效运行次数': '0'
        })

# ================= 5. 错分类型汇总分析 =================

def analyze_error_types(all_error_details):
    """分析所有模型的错分类型分布"""
    if not all_error_details:
        print("\n没有错误分析数据可用")
        return None
    
    # 合并所有模型的错分数据
    all_errors_combined = pd.concat(all_error_details, ignore_index=True)
    
    # 统计错分类型
    error_type_stats = all_errors_combined['错分类型'].value_counts().sort_index()
    
    print(f"\n{'='*60}")
    print("错分类型全局统计")
    print(f"{'='*60}")
    print("\n错分类型分布:")
    for err_type, count in error_type_stats.items():
        percentage = count / len(all_errors_combined) * 100
        if err_type == 0:
            print(f"  正确: {count} 条 ({percentage:.1f}%)")
        elif err_type == 1:
            print(f"  一级错分: {count} 条 ({percentage:.1f}%)")
        elif err_type == 2:
            print(f"  二级错分: {count} 条 ({percentage:.1f}%)")
    
    # 按模型统计
    print("\n各模型错分统计:")
    error_summary_by_model = []
    
    for error_df in all_error_details:
        # 提取模型名
        if '模型' in error_df.columns and not error_df.empty:
            model_name = error_df['模型'].iloc[0]
        else:
            # 如果没有模型列，使用索引
            model_idx = all_error_details.index(error_df) + 1
            model_name = f"模型{model_idx}"
            
        total = len(error_df)
        
        correct = (error_df['错分类型'] == 0).sum()
        l1_errors = (error_df['错分类型'] == 1).sum()
        l2_errors = (error_df['错分类型'] == 2).sum()
        
        error_summary_by_model.append({
            '模型': model_name,
            '总样本数': total,
            '正确数': correct,
            '正确率': f"{correct/total:.2%}" if total > 0 else "0.00%",
            '一级错分数': l1_errors,
            '一级错分率': f"{l1_errors/total:.2%}" if total > 0 else "0.00%",
            '二级错分数': l2_errors,
            '二级错分率': f"{l2_errors/total:.2%}" if total > 0 else "0.00%"
        })
    
    error_summary_df = pd.DataFrame(error_summary_by_model)
    print(error_summary_df.to_string(index=False))
    
    return error_summary_df

# ================= 6. 输出结果 =================

print(f"\n{'='*60}")
print("分析完成！")
print(f"{'='*60}")

# 执行错分类型汇总分析
error_summary_df = analyze_error_types(all_error_details)

if summary_l1:
    df_res1 = pd.DataFrame(summary_l1)
    print("\n一级标签评估结果:")
    print(df_res1.to_string(index=False))

if summary_l2:
    df_res2 = pd.DataFrame(summary_l2)
    print("\n二级标签评估结果:")
    print(df_res2.to_string(index=False))

# 保存综合报告
save_path = os.path.join(folder_path, '综合模型评估报告.xlsx')
with pd.ExcelWriter(save_path) as writer:
    if summary_l1:
        df_res1.to_excel(writer, sheet_name='一级标签评估', index=False)
    
    if summary_l2:
        df_res2.to_excel(writer, sheet_name='二级标签评估', index=False)
    
    if error_summary_df is not None:
        error_summary_df.to_excel(writer, sheet_name='错分类型统计', index=False)
    
    if all_misclassifications_l1:
        combined_l1 = pd.concat(all_misclassifications_l1, ignore_index=True)
        combined_l1.to_excel(writer, sheet_name='一级错分详情', index=False)
    
    if all_misclassifications_l2:
        combined_l2 = pd.concat(all_misclassifications_l2, ignore_index=True)
        combined_l2.to_excel(writer, sheet_name='二级错分详情', index=False)

print(f"\n综合报告已保存: {save_path}")
print(f"所有混淆矩阵图片已保存到: {folder_path}")
print(f"各模型的详细错分分析文件也已保存到: {folder_path}")