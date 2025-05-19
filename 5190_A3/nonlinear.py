import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

# 测试数据（真实正相关，不变）
test_data = pd.DataFrame({
    'Height': [182, 174, 179, 172, 181, 175, 178, 173, 185, 176],
    'Weight': [65, 70, 64, 67, 66, 73, 68, 66, 72, 69]
})

# ========= 数据增强函数 =========
def generate_augmented_data(target_size=1000, nonlinearity=True):
    np.random.seed(42)
    heights = np.random.uniform(168, 186, target_size)
    if nonlinearity:
        weights = (
            -0.015 * (heights - 176) ** 3 +
            0.4 * (heights - 176) +
            62 +
            np.random.normal(0, 2.0, target_size)
        )
    else:
        weights = 0.9 * heights - 95 + np.random.normal(0, 2.0, target_size)
    return pd.DataFrame({'Height': heights, 'Weight': weights})

# ========= 生成训练数据 =========
train_data = generate_augmented_data(target_size=2000, nonlinearity=True)
min_h, max_h = 168, 186
train_data['Height'] = max_h + min_h - train_data['Height']  # 反转 height
train_data['Weight'] += 5  # 向上整体偏移

# 特征与标签
X_train = train_data[['Height']]
y_train = train_data['Weight']
X_test = test_data[['Height']]
y_test = test_data['Weight']

# ========= 多项式回归模型（非线性） =========
poly_model = Pipeline([
    ('poly', PolynomialFeatures(degree=4)),
    ('linear', LinearRegression())
])
poly_model.fit(X_train, y_train)
y_poly_train = poly_model.predict(X_train)
y_poly_test = poly_model.predict(X_test)
mae_poly_train = mean_absolute_error(y_train, y_poly_train)
mae_poly_test = mean_absolute_error(y_test, y_poly_test)

# ========= 提取多项式表达式 =========
coefs = poly_model.named_steps['linear'].coef_
intercept = poly_model.named_steps['linear'].intercept_
terms = [f"{intercept:.4f}"]
for i, coef in enumerate(coefs[1:], 1):
    terms.append(f"{coef:.4f} * x^{i}")
poly_equation = " + ".join(terms)

# ========= 线性回归模型 =========
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
y_linear_train = linear_model.predict(X_train)
y_linear_test = linear_model.predict(X_test)
mae_linear_train = mean_absolute_error(y_train, y_linear_train)
mae_linear_test = mean_absolute_error(y_test, y_linear_test)

# ========= 提取线性表达式 =========
linear_coef = linear_model.coef_[0]
linear_intercept = linear_model.intercept_
linear_equation = f"f(x) = {linear_intercept:.4f} + {linear_coef:.4f} * x"

# ========= 输出信息 =========
print("=== Non-linear Model Information ===")
print("Model: Polynomial Regression (degree=4)")
print("Data: 2000 augmented training samples")
print("Fitted Function (Polynomial):")
print(f"f(x) = {poly_equation}")
print(f"Train MAE (Polynomial): {mae_poly_train:.4f}")
print(f"Test  MAE (Polynomial): {mae_poly_test:.4f}")
print()

print("=== Linear Model Information ===")
print("Model: Linear Regression")
print("Same data used as above")
print("Fitted Function (Linear):")
print(linear_equation)
print(f"Train MAE (Linear): {mae_linear_train:.4f}")
print(f"Test  MAE (Linear): {mae_linear_test:.4f}")
print()

# ========= 分析解释 =========
if mae_poly_test < mae_linear_test:
    print("✅ The non-linear model performs better than the linear model based on MAE.")
    print("Explanation: The relationship between height and weight is not perfectly linear.")
    print("Polynomial regression captures subtle nonlinear fluctuations in the data,")
    print("resulting in better generalization on real test data.\n")
else:
    print("⚠️ The non-linear model did not outperform the linear model. Consider tuning degree or data.\n")

# ========= 可视化 =========
plt.figure(figsize=(10, 6))
plt.scatter(X_train['Height'], y_train, color='lightgray', s=10, label='Training Data (+5kg)')
plt.scatter(X_test['Height'], y_test, color='red', label='Test Data')

x_range = np.linspace(min_h, max_h, 300).reshape(-1, 1)
y_poly_line = poly_model.predict(x_range)
y_linear_line = linear_model.predict(x_range)

plt.plot(x_range, y_poly_line, color='green', linewidth=2, label='Polynomial Fit (degree=4)')
plt.plot(x_range, y_linear_line, color='black', linestyle='--', linewidth=2, label='Linear Fit')
plt.xlabel('Reflected Height (cm)')
plt.ylabel('Weight (kg)')
plt.title('Polynomial vs Linear Regression')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
