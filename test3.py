import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# 输入数据
rain = np.array([0.13, 4.99, 16.97, 7.62, 6.12, 10.25, 7.13, 0.75, 3.63, 0.5,
                 0., 0.13, 1.25, 1., 0.63, 1.5, 1.63, 0.87, 0.13, 1.25,
                 2., 1.25, 1.13, 0.63])
water_level = np.array([7.6, 7.58, 7.57, 7.57, 7.62, 7.77, 8.25, 8.63, 8.9, 9.,
                        8.97, 8.85, 8.74, 8.63, 8.55, 8.48, 8.43, 8.39, 8.35, 8.32,
                        8.28, 8.25, 8.23, 8.22])
next_hour_change = np.array([-0.02, -0.01, 0., 0.05, 0.15, 0.48, 0.38, 0.27, 0.1, -0.03,
                             -0.12, -0.11, -0.11, -0.08, -0.07, -0.05, -0.04, -0.04, -0.03,
                             -0.04, -0.03, -0.02, -0.01, 0.01])

# 构建自变量矩阵X，包括water_level和过去24小时的rain的高阶多项式特征
X = np.column_stack((water_level[:-24], rain[24:]))
print("X", X)
# 创建多项式特征
degree = 2  # 多项式特征的阶数
poly = PolynomialFeatures(degree=degree, include_bias=False)
X_poly = poly.fit_transform(X)

# 创建并训练多项式回归模型
model = LinearRegression()
model.fit(X_poly, next_hour_change[24:])

# 输出回归系数
beta0 = model.intercept_
beta = model.coef_
print("回归系数:")
print("beta0:", beta0)
print("beta:", beta)