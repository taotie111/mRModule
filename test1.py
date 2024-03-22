import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# 输入数据
rain = np.array([0.13, 4.99, 16.97, 7.62, 6.12, 10.25, 7.13, 0.75, 3.63, 0.5,
                 0., 0.13, 1.25, 1., 0.63, 1.5, 1.63, 0.87, 0.13, 1.25,
                 2., 1.25, 1.13, 0.63, 0])
water_level = np.array([7.6, 7.58, 7.57, 7.57, 7.62, 7.77, 8.25, 8.63, 8.9, 9.,
                        8.97, 8.85, 8.74, 8.63, 8.55, 8.48, 8.43, 8.39, 8.35, 8.32,
                        8.28, 8.25, 8.23, 8.22, 8.22])
next_hour_change = np.array([-0.02, -0.01, 0., 0.05, 0.15, 0.48, 0.38, 0.27, 0.1, -0.03,
                             -0.12, -0.11, -0.11, -0.08, -0.07, -0.05, -0.04, -0.04, -0.03,
                             -0.04, -0.03, -0.02, -0.01, 0.01, 0])

# 构建输入矩阵 X
X = np.zeros((len(rain) - 24, 24))
for i in range(24, len(rain)):
    X[i-24, :] = rain[i-24:i]
    print("回归系数:")

# 构建输出向量 y
y = next_hour_change[24:]

# 创建并训练多项式回归模型
degree = 2  # 多项式的阶数
poly = PolynomialFeatures(degree=degree, include_bias=False)
X_poly = poly.fit_transform(X)
model = LinearRegression()
model.fit(X_poly, y)

# 输出回归系数
beta0 = model.intercept_
beta = model.coef_
print("回归系数:")
print("beta0:", beta0)
print("beta:", beta)

# 构建多项式回归模型函数
def regression_function(x):
    x_poly = poly.transform(x.reshape(1, -1))
    y_pred = np.dot(x_poly, beta) + beta0
    return y_pred

# 使用模型进行预测
past_rain = np.array([0.13, 4.99, 16.97, 7.62, 6.12, 10.25, 7.13, 0.75, 3.63, 0.5,
                      0., 0.13, 1.25, 1., 0.63, 1.5, 1.63, 0.87, 0.13, 1.25,
                      2., 1.25, 1.13, 0.63])
next_hour_rain = 2.5  # 下一个小时的降雨量
past_rain = np.append(past_rain, next_hour_rain)  # 将下一个小时的降雨量添加到过去的降雨数据中
next_hour_change_pred = regression_function(past_rain)
print("预测下一个小时的水位变化:", next_hour_change_pred)

# 打印多项式回归模型函数
print("多项式回归模型函数:")
print("f(x) =", end=" ")
for i in range(len(beta)):
    if i == 0:
        print(f"{beta0:.4f}", end=" ")
    else:
        print(f"+ {beta[i]:.4f} * x^{i}", end=" ")
