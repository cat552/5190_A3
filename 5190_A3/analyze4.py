import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr, rankdata
import matplotlib.pyplot as plt

# Step 1: Prepare Data
heights = np.array([175, 182, 170, 178, 184, 173, 179, 176, 171, 180, 177, 181, 172, 183, 174])
weights = np.array([57, 65, 55, 58, 75, 58, 63, 65, 53, 60, 62, 71, 57, 67, 60])

# === Model Description ===
print("=== Model Description ===")
print("We analyze the relationship between height (cm) and weight (kg) using correlation analysis.")
print("Two statistical techniques are used: Pearson (for linear relationship) and Spearman (for monotonic relationship).\n")

# === Pearson Calculation ===
mean_h = np.mean(heights)
mean_w = np.mean(weights)
dev_h = heights - mean_h
dev_w = weights - mean_w
cov_hw = np.sum(dev_h * dev_w) / (len(heights) - 1)
std_h = np.std(heights, ddof=1)
std_w = np.std(weights, ddof=1)
pearson_manual = cov_hw / (std_h * std_w)
pearson_scipy, pearson_p = pearsonr(heights, weights)

# === Spearman Calculation ===
rank_h = rankdata(heights)
rank_w = rankdata(weights)
dev_rh = rank_h - np.mean(rank_h)
dev_rw = rank_w - np.mean(rank_w)
cov_r = np.sum(dev_rh * dev_rw) / (len(rank_h) - 1)
std_rh = np.std(rank_h, ddof=1)
std_rw = np.std(rank_w, ddof=1)
spearman_manual = cov_r / (std_rh * std_rw)
spearman_scipy, spearman_p = spearmanr(heights, weights)

# === Technical Details ===
print("=== Technical Details ===")
print(f"Mean Height = {mean_h:.2f}, Mean Weight = {mean_w:.2f}")
print(f"Cov(Height, Weight) = {cov_hw:.4f}")
print(f"Std Height = {std_h:.4f}, Std Weight = {std_w:.4f}")
print(f"Pearson Correlation (manual) = {pearson_manual:.4f}")
print(f"Pearson Correlation (scipy)  = {pearson_scipy:.4f}, p = {pearson_p:.4f}\n")

print(f"Rank(Height) = {rank_h}")
print(f"Rank(Weight) = {rank_w}")
print(f"Cov(Rank Height, Rank Weight) = {cov_r:.4f}")
print(f"Std Rank Height = {std_rh:.4f}, Std Rank Weight = {std_rw:.4f}")
print(f"Spearman Correlation (manual) = {spearman_manual:.4f}")
print(f"Spearman Correlation (scipy)  = {spearman_scipy:.4f}, p = {spearman_p:.4f}\n")

# === Evaluation ===
print("=== Evaluation ===")
if pearson_p < 0.05:
    print("Pearson correlation is statistically significant.")
else:
    print("Pearson correlation is NOT statistically significant.")

if spearman_p < 0.05:
    print("Spearman correlation is statistically significant.")
else:
    print("Spearman correlation is NOT statistically significant.")
print()

# === Conclusion ===
print("=== Conclusion ===")
print("Both Pearson and Spearman correlation coefficients are positive,")
print("indicating that as height increases, weight tends to increase as well.")
print("Pearson supports a linear relationship; Spearman supports a general monotonic trend.\n")

# === PLOT 1: Pearson (original data with regression line) ===
plt.figure(figsize=(8, 6))
plt.scatter(heights, weights, color='blue', label='Data Points')
m, b = np.polyfit(heights, weights, 1)
plt.plot(heights, m * heights + b, color='red', linestyle='--', label=f'Linear Fit: y={m:.2f}x+{b:.1f}')
plt.title('Pearson Correlation: Height vs Weight')
plt.xlabel('Height (cm)')
plt.ylabel('Weight (kg)')
plt.legend()
plt.grid(True)
plt.text(0.05, 0.95, f'Pearson r = {pearson_manual:.4f}', transform=plt.gca().transAxes,
         fontsize=12, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
plt.tight_layout()
plt.show()

# === PLOT 2: Spearman (ranked data, no regression line) ===
plt.figure(figsize=(8, 6))
plt.scatter(rank_h, rank_w, color='green', marker='x', label='Ranked Data Points')
plt.title('Spearman Correlation: Rank(Height) vs Rank(Weight)')
plt.xlabel('Rank of Height')
plt.ylabel('Rank of Weight')
plt.grid(True)
plt.legend()
plt.text(0.05, 0.95, f'Spearman ρ = {spearman_manual:.4f}', transform=plt.gca().transAxes,
         fontsize=12, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
plt.tight_layout()
plt.show()

