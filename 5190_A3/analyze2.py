import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from scipy import stats
import matplotlib.pyplot as plt

# ---------- Training data ----------
train_data = pd.DataFrame({
    'Height': [175, 182, 170, 178, 184, 173, 179, 176, 171, 180, 177, 181, 172, 183, 174],
    'Weight': [57, 65, 55, 58, 75, 58, 63, 65, 53, 60, 62, 71, 57, 67, 60]
})
train_data['Gender'] = np.where(train_data['Height'] < 175, 'Female', 'Male')

# One-hot encode Gender for comparison model
train_data['Gender_Male'] = (train_data['Gender'] == 'Male').astype(int)

# ---------- Model 1: Height only ----------
model_height = LinearRegression().fit(train_data[['Height']], train_data['Weight'])

# ---------- Model 2: Height + Gender ----------
model_height_gender = LinearRegression().fit(train_data[['Height', 'Gender_Male']], train_data['Weight'])

# ---------- Test data ----------
test_df = pd.DataFrame({
    'Height': [177, 174, 179, 172, 181, 175, 178, 173, 180, 176],
    'Weight': [60, 60, 64, 57, 66, 63, 66, 56, 62, 59]
})
test_df['Gender'] = np.where(test_df['Height'] < 175, 'Female', 'Male')
test_df['Gender_Male'] = (test_df['Gender'] == 'Male').astype(int)

# Predict using both models
test_df['Pred_Height'] = model_height.predict(test_df[['Height']])
test_df['Pred_HeightGender'] = model_height_gender.predict(test_df[['Height', 'Gender_Male']])

# Compute errors
test_df['Error_Height'] = abs(test_df['Weight'] - test_df['Pred_Height'])
test_df['Error_HeightGender'] = abs(test_df['Weight'] - test_df['Pred_HeightGender'])
test_df['Diff'] = test_df['Error_Height'] - test_df['Error_HeightGender']

# ---------- Output comparison table ----------
comparison_df = test_df[['Height', 'Weight', 'Gender', 'Pred_Height', 'Pred_HeightGender', 'Error_Height', 'Error_HeightGender']].round(2)
print("\n Model Comparison Table:")
print(comparison_df.to_string(index=False))

# ---------- Save table as image ----------
fig, ax = plt.subplots(figsize=(14, 4))
ax.axis('off')
table = ax.table(cellText=comparison_df.values,
                 colLabels=comparison_df.columns,
                 cellLoc='center',
                 loc='center')
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 1.5)
plt.title("Model Comparison: Height vs Height+Gender", fontsize=14)
plt.tight_layout()
plt.savefig("model_comparison_table.png", dpi=300)
plt.close()

# ---------- Task 1: One-sample t-test on new model MAE < 5 ----------
print("\nTask 1: One-sample t-test on MAE < 5 (Height + Gender model)")
mu_0 = 5
n = len(test_df)
x̄ = test_df['Error_HeightGender'].mean()
s = test_df['Error_HeightGender'].std(ddof=1)
SE = s / np.sqrt(n)
t_stat = (x̄ - mu_0) / SE
df = n - 1
t_crit = stats.t.ppf(0.05, df)
p = stats.t.cdf(t_stat, df)

print(f"H₀: μ = 5")
print(f"H₁: μ < 5")
print(f"n = {n}, df = {df}")
print(f"x̄ = {x̄:.3f}, s = {s:.3f}, SE = {SE:.3f}")
print(f"t = {t_stat:.3f}, t(0.05, {df}) = {t_crit:.3f}, p = {p:.4f}")
print("Conclusion:", "Reject H₀ (MAE < 5 is significant)"
      if t_stat < t_crit else " Retain H₀ (no significant evidence MAE < 5)")

# ---------- Task 2: Paired t-test comparing two models ----------
print("\nTask 2: Paired t-test (Error_Height - Error_HeightGender)")
d̄ = test_df['Diff'].mean()
sd = test_df['Diff'].std(ddof=1)
SE_d = sd / np.sqrt(n)
t2 = d̄ / SE_d
t2_crit = stats.t.ppf(0.95, df)
p2 = 1 - stats.t.cdf(t2, df)

print(f"H₀: μ_diff = 0")
print(f"H₁: μ_diff > 0")
print(f"d̄ = {d̄:.3f}, s = {sd:.3f}, SE = {SE_d:.3f}")
print(f"t = {t2:.3f}, t(0.05, {df}) = {t2_crit:.3f}, p = {p2:.4f}")
print("Conclusion:", "Reject H₀ (Height+Gender model outperforms Height only)"
      if t2 > t2_crit else " Retain H₀ (no significant difference)")
