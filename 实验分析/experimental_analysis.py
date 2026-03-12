import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.formula.api import ols
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 设置中文支持
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def run_experimental_analysis():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, '..', 'data', '新汇总.xlsx')
    persona_path = os.path.join(script_dir, '..', '消费者画像分析', 'clustered_customers.xlsx')
    
    if not os.path.exists(file_path):
        print(f"错误: 找不到数据文件 {file_path}")
        return

    # 1. 数据预处理
    df = pd.read_excel(file_path)
    # 质量控制过滤 (Q22=3)
    # Pandas 读取时，有效数据为 102 份（对应 Excel 表中含表头共 103 行）
    df = df[df.iloc[:, 33] == 3].copy()
    
    analysis_df = pd.DataFrame()
    analysis_df['Scenario'] = df['情景随机'].astype(int)
    
    # 提取 WTP 并清理字符串，例如 "10（因为...）" -> 10,  "10000元" -> 异常值
    import re
    def extract_wtp(val):
        if pd.isna(val): return np.nan
        val_str = str(val)
        # 提取第一个出现的数字
        nums = re.findall(r'\d+', val_str)
        if nums:
            num = float(nums[0])
            # 过滤极端异常值 (如 10000元 或 1元) 和无效文字
            if 5 <= num <= 200:
                return num
        return np.nan

    analysis_df['WTP'] = df.iloc[:, 35].apply(extract_wtp)
    analysis_df['Purchase'] = pd.to_numeric(df.iloc[:, 36], errors='coerce')
    analysis_df['Evaluation'] = pd.to_numeric(df.iloc[:, 37], errors='coerce')
    
    # 清理 WTP 中的异常值
    analysis_df = analysis_df.dropna()
    print(f"--- 实验样本预处理详情 ---")
    print(f"原始有效问卷 (Q22=3): {len(df)} 份 (Excel含表头为 {len(df)+1} 行)")
    print(f"剔除 WTP 无效作答 (如'不知道', '10000元'等): {len(df) - len(analysis_df)} 份")
    print(f"最终进入双刃剑分析的样本量 (N={len(analysis_df)})")
    
    # 2. 主效应分析 (Main Effect)
    # 检验不同情景下 WTP 是否有显著差异
    model = ols('WTP ~ C(Scenario)', data=analysis_df).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)
    print("\n[1] WTP 主效应分析 (ANOVA):")
    print(anova_table)
    
    # 3. 交互效应分析 (Interaction Effect)
    # 结合之前的消费者画像 (需要合并)
    if os.path.exists(persona_path):
        persona_df = pd.read_excel(persona_path)[['序号', 'Cluster']]
        analysis_df = analysis_df.merge(persona_df, left_index=True, right_index=True, how='left')
        analysis_df['Cluster'] = analysis_df['Cluster'].fillna(0).astype(int)
        
        # 建立交互模型: Scenario * Cluster
        inter_model = ols('WTP ~ C(Scenario) * C(Cluster)', data=analysis_df).fit()
        inter_anova = sm.stats.anova_lm(inter_model, typ=2)
        print("\n[2] 交互效应分析 (Scenario x Persona):")
        print(inter_anova)
        
        # 绘图展示交互效应
        plt.figure(figsize=(10, 6))
        sns.pointplot(x='Scenario', y='WTP', hue='Cluster', data=analysis_df, capsize=.1)
        plt.title('情境与人群画像对支付意愿的交互效应')
        plt.xlabel('实验情景 (1-4)')
        plt.ylabel('平均支付意愿 (元/斤)')
        plt.savefig(os.path.join(script_dir, 'interaction_effect.png'))
        plt.close()

    # 4. 中介效应检验 (Mediation Analysis)
    # 路径 A: IV -> Mediator (Scenario -> Evaluation)
    # 路径 B+C': IV + Mediator -> DV (Scenario + Evaluation -> WTP)
    
    print("\n[3] 中介效应检验 (Evaluation 作为中介):")
    # 路径 A
    res_a = sm.OLS(analysis_df['Evaluation'], sm.add_constant(analysis_df['Scenario'])).fit()
    # 路径 B + Direct Effect (C')
    res_bc = sm.OLS(analysis_df['WTP'], sm.add_constant(analysis_df[['Scenario', 'Evaluation']])).fit()
    
    print("Path A (Scenario -> Evaluation):")
    print(f"Coeff: {res_a.params['Scenario']:.4f}, p: {res_a.pvalues['Scenario']:.4f}")
    print("Path B (Evaluation -> WTP):")
    print(f"Coeff: {res_bc.params['Evaluation']:.4f}, p: {res_bc.pvalues['Evaluation']:.4f}")
    print("Direct Effect C' (Scenario -> WTP):")
    print(f"Coeff: {res_bc.params['Scenario']:.4f}, p: {res_bc.pvalues['Scenario']:.4f}")

    # 5. 结果及可视化输出
    plt.figure(figsize=(8, 6))
    sns.barplot(x='Scenario', y='WTP', data=analysis_df, palette='viridis')
    plt.title('不同实验情境下的平均支付意愿 (WTP)')
    plt.savefig(os.path.join(script_dir, 'main_effect_wtp.png'))
    plt.close()
    
    print(f"\n✅ 实验分析完成，图表已保存至所在目录。")

if __name__ == "__main__":
    run_experimental_analysis()
