import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from scipy.stats import chi2_contingency
import matplotlib.pyplot as plt

# 设置中文字体和兼容性字体
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 原始数据
data = pd.DataFrame({
    'Height': [175, 182, 170, 178, 184, 173, 179, 176, 171, 180, 177, 181, 172, 183, 174],
    'Weight': [57, 65, 55, 58, 75, 58, 63, 65, 53, 60, 62, 71, 57, 67, 60]
})
data['Gender'] = np.where(data['Height'] < 175, 'Female', 'Male')

# 使用整体模型训练
model = LinearRegression().fit(data[['Height']], data['Weight'])
data['Predicted'] = model.predict(data[['Height']])
data['MAE'] = abs(data['Predicted'] - data['Weight'])

# 误差分组：Low / High Error
threshold = np.median(data['MAE'])
data['ErrorGroup'] = np.where(data['MAE'] <= threshold, 'Low Error', 'High Error')

# 身高中位数分组
median_height = np.median(data['Height'])
data['HeightGroup'] = np.where(data['Height'] <= median_height, 'Short', 'Tall')

# -------- 通用分析函数 --------
def analyze_attribute(attr_name, attr_label):
    contingency = pd.crosstab(data[attr_name], data['ErrorGroup']).reindex(columns=['Low Error', 'High Error'])
    chi2, p, dof, expected = chi2_contingency(contingency)

    obs = contingency.values
    exp = expected
    chi_vals = (obs - exp) ** 2 / exp

    row_labels = list(contingency.index)
    row_totals = obs.sum(axis=1)
    col_totals = obs.sum(axis=0)
    grand_total = obs.sum()

    # 构造表格内容，避免使用特殊符号，使用 **2 显示平方
    table_data = [["", "Low Error", "High Error", "Total"]]
    for i, label in enumerate(row_labels):
        row = [
            label,
            f"{obs[i,0]}/{int(exp[i,0])}\n(({obs[i,0]} - {int(exp[i,0])})^2 / {int(exp[i,0])}) = {chi_vals[i,0]:.2f}",
            f"{obs[i,1]}/{int(exp[i,1])}\n(({obs[i,1]} - {int(exp[i,1])})^2 / {int(exp[i,1])}) = {chi_vals[i,1]:.2f}",
            str(row_totals[i])
        ]
        table_data.append(row)
    table_data.append(["Total", str(col_totals[0]), str(col_totals[1]), str(grand_total)])

    # 绘制表格图
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.axis('off')
    t = ax.table(cellText=table_data, cellLoc='center', loc='center')
    t.auto_set_font_size(False)
    t.set_fontsize(12)
    t.scale(1.4, 2)
    plt.tight_layout()
    plt.show()

    # 打印分析结论
    print(f"\n【分析：{attr_label} 与预测误差的关系】")
    if p < 0.05:
        print(f"χ² = {chi2:.2f}, p = {p:.4f} < 0.05 → 有统计显著性差异")
        print(f"→ {attr_label} 显著影响预测误差\n")
    else:
        print(f"χ² = {chi2:.2f}, p = {p:.4f} ≥ 0.05 → 无统计显著性差异")
        print(f"→ {attr_label} 对误差影响不显著\n")

# -------- 分析：性别 --------
analyze_attribute('Gender', '性别')

# -------- 分析：身高（高/低） --------
analyze_attribute('HeightGroup', '身高（高低分组）')