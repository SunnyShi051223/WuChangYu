import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 设置中文支持
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def run_descriptive_analysis():
    # 获取脚本所在目录的绝对路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = script_dir
    
    # 数据路径 (相对于脚本所在目录)
    file_path = os.path.join(script_dir, '..', 'data', '汇总版_已清洗数值化.xlsx')
    
    if not os.path.exists(file_path):
        print(f"错误: 找不到数据文件 {file_path}")
        return

    df = pd.read_excel(file_path)

    # --- 1. 受访者基本情况 (Demographics) ---
    # 定义映射
    gender_map = {1: '男', 2: '女'}
    age_map = {1: '<18', 2: '18-25', 3: '26-35', 4: '36-45', 5: '>45'}
    edu_map = {1: '专科及以下', 2: '本科', 3: '研究生及以上'}
    income_map = {1: '<3000', 2: '3000-6000', 3: '6001-10000', 4: '10001-15000', 5: '>15000', 6: '无收入(学生)'}
    loc_map = {1: '武汉', 2: '鄂州', 3: '湖北省内其他'}
    
    # 提取列名
    cols = df.columns
    gender_col = [c for c in cols if '1. 您的性别' in str(c)][0]
    age_col = [c for c in cols if '2. 您的年龄' in str(c)][0]
    edu_col = [c for c in cols if '3. 您的学历' in str(c)][0]
    income_col = [c for c in cols if '4. 您的月收入' in str(c)][0]
    loc_col = [c for c in cols if '5. 您目前居住' in str(c)][0]

    demo_df = df[[gender_col, age_col, edu_col, income_col, loc_col]].copy()
    demo_df[gender_col] = demo_df[gender_col].map(gender_map)
    demo_df[age_col] = demo_df[age_col].map(age_map)
    demo_df[edu_col] = demo_df[edu_col].map(edu_map)
    demo_df[income_col] = demo_df[income_col].map(income_map)
    demo_df[loc_col] = demo_df[loc_col].map(loc_map)

    # 绘制人口学分布图
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    demo_cols = [gender_col, age_col, edu_col, income_col, loc_col]
    titles = ['性别分布', '年龄分布', '学历分布', '月收入分布', '居住地分布']
    
    for i, col in enumerate(demo_cols):
        counts = demo_df[col].value_counts()
        counts.plot(kind='pie', autopct='%1.1f%%', ax=axes[i], startangle=90, colors=sns.color_palette('pastel'))
        axes[i].set_title(titles[i])
        axes[i].set_ylabel('')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'demographics_distribution.png'))
    plt.close()
    print("✅ 受访者基本情况分布图已生成。")

    # --- 2. 产地认知现状统计 (Origin Cognition) ---
    # Q7: 关于武昌鱼说法 (1=苏轼, 2=三国/古武昌, 3=清代)
    # Q8: 原产地认知 (1=武汉, 2=鄂州, 3=黄石, 5=不清楚)
    
    q7_col = [c for c in cols if '7. 关于武昌鱼' in str(c)][0]
    q8_col = [c for c in cols if '8. 您认为武昌鱼' in str(c)][0]
    
    q7_map = {1: '苏轼命名', 2: '三国起源(古武昌)', 3: '清代产物'}
    q8_map = {1: '武汉', 2: '鄂州', 3: '黄石', 5: '不清楚'}
    
    q7_counts = df[q7_col].map(q7_map).value_counts(normalize=True) * 100
    q8_counts = df[q8_col].map(q8_map).value_counts(normalize=True) * 100
    
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    q7_counts.plot(kind='bar', color='skyblue')
    plt.title('武昌鱼历史认知统计 (%)')
    plt.ylabel('比例 (%)')
    
    plt.subplot(1, 2, 2)
    q8_counts.plot(kind='bar', color='salmon')
    plt.title('武昌鱼核心原产地认知统计 (%)')
    plt.ylabel('比例 (%)')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'origin_cognition_stats.png'))
    plt.close()
    print("✅ 产地认知现状统计图已生成。")

    # --- 3. 消费者购买行为与态度分析 (Behavior & Attitude) ---
    # Q6: 购买历史 (1=是, 2=否)
    # Q10-Q21: 态度题 (李克特5级量表)
    
    q6_col = [c for c in cols if '6. 您在过去一年' in str(c)][0]
    q6_counts = df[q6_col].map({1: '有食用史', 2: '无食用史'}).value_counts()
    
    # 聚合四个心理维度
    df['产地敏感度'] = df.iloc[:, 21:24].mean(axis=1) # Q10-12
    df['文化涉入度'] = df.iloc[:, 24:27].mean(axis=1) # Q13-15
    df['品牌依赖度'] = df.iloc[:, 27:30].mean(axis=1) # Q16-18
    df['价格敏感度'] = df.iloc[:, 30:33].mean(axis=1) # Q19-21
    
    mean_scores = df[['产地敏感度', '文化涉入度', '品牌依赖度', '价格敏感度']].mean()
    
    plt.figure(figsize=(10, 6))
    mean_scores.plot(kind='barh', color='lightgreen')
    plt.title('消费者心理态度均值对比 (1-5分)')
    plt.xlabel('均值分数')
    plt.xlim(1, 5)
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'behavior_attitude_summary.png'))
    plt.close()
    print("✅ 消费者购买行为与态度图已生成。")

    # 保存统计摘要文本
    with open(os.path.join(output_dir, 'descriptive_summary.txt'), 'w', encoding='utf-8') as f:
        f.write("--- 描述性分析摘要 ---\n\n")
        f.write("1. 性别比例:\n" + str(demo_df[gender_col].value_counts(normalize=True)*100) + "\n\n")
        f.write("2. 历史认知正确率 (三国起源):\n" + str(q7_counts.get('三国起源(古武昌)', 0)) + "%\n\n")
        f.write("3. 产地正确识别率 (鄂州):\n" + str(q8_counts.get('鄂州', 0)) + "%\n\n")
        f.write("4. 态度得分均值:\n" + str(mean_scores) + "\n")

if __name__ == "__main__":
    run_descriptive_analysis()
