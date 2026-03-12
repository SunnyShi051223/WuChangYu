import pandas as pd
import numpy as np
import os
from factor_analyzer.factor_analyzer import calculate_kmo, calculate_bartlett_sphericity

def cronbach_alpha(df):
    """
    计算 Cronbach's Alpha 系数
    公式: alpha = (k / (k - 1)) * (1 - sum(si^2) / st^2)
    其中 k 为题目数，si^2 为第 i 题的方差，st^2 为总得分的方差
    """
    # 题目数
    k = df.shape[1]
    # 每题的方差
    item_variances = df.var(axis=0, ddof=1)
    # 总分的方差
    total_score_variance = df.sum(axis=1).var(ddof=1)
    
    alpha = (k / (k - 1)) * (1 - item_variances.sum() / total_score_variance)
    return alpha

def run_reliability_validity():
    # 数据路径调整：尝试从根目录或当前目录寻找
    path_options = [
        os.path.join('data', '新汇总.xlsx'),
        os.path.join('..', 'data', '新汇总.xlsx')
    ]
    
    file_path = None
    for opt in path_options:
        if os.path.exists(opt):
            file_path = opt
            break
            
    if not file_path:
        print(f"错误: 找不到数据文件，请检查 data 文件夹路径。")
        return

    df = pd.read_excel(file_path)
    
    # 提取量表题 Q10-Q21 (12个项)
    likert_items = df.iloc[:, 21:33]
    
    print("="*50)
    print("   武昌鱼问卷 - 信效度分析报告 (Python)")
    print("="*50)
    
    # 1. 信度分析
    alpha = cronbach_alpha(likert_items)
    print(f"\n[1] 信度检验 (Reliability)")
    print(f"Cronbach's α 系数: {alpha:.4f}")
    
    if alpha >= 0.8:
        print("评价: 非常好 (Very Good)")
    elif alpha >= 0.7:
        print("评价: 良好 (Good)")
    elif alpha >= 0.6:
        print("评价: 合格 (Acceptable)")
    else:
        print("评价: 不理想 (Poor)")

    # 2. 效度分析
    kmo_all, kmo_model = calculate_kmo(likert_items)
    chi_square, p_value = calculate_bartlett_sphericity(likert_items)
    
    print(f"\n[2] 效度检验 (Validity)")
    print(f"KMO (Kaiser-Meyer-Olkin) 测度: {kmo_model:.4f}")
    print(f"Bartlett's 球形检验 Chi-square: {chi_square:.4f}")
    print(f"Bartlett's 球形检验 P-value: {p_value:.4f}")
    
    if kmo_model >= 0.7:
        print("评价: 效度良好，适合做因子分析。")
    elif kmo_model >= 0.6:
        print("评价: 效度基本合格。")
    else:
        print("评价: 效度较低，数据聚类效果可能一般。")
        
    print("\n" + "="*50)

if __name__ == "__main__":
    run_reliability_validity()
