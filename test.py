import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# 读取 Excel 文件数据
def read_excel_data(file_path):
    df = pd.read_excel(file_path)
    TP = pd.to_datetime(df['TP'])
    TQ = pd.to_datetime(df['TQ'])
    rain = df.loc[TP, 'P'].values
    water_level = df.loc[TQ, 'Q'].values
    next_hour_change = np.diff(water_level)  # 计算下一个小时的水位变化
    return rain, water_level, next_hour_change

# 输入文件路径
file_path = '何岙场次洪水.xlsx'

# 读取 Excel 文件数据
rain, water_level, next_hour_change = read_excel_data(file_path)

# 将输入数据转换为二维数组
X = rain.reshape(-1, 1)

# 添加高次项和交互项
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X)

# 创建并训练多元非线性回归模型
model = LinearRegression()
model.fit(X_poly, next_hour_change)

# 输出回归系数和截距
beta0 = model.intercept_
beta = model.coef_
print("回归系数:")
print("beta0:", beta0)
print("beta:", beta)

# 构建多元非线性回归模型函数
def regression_function(x):
    # 添加高次项和交互项
    x_poly = poly.transform(x.reshape(-1, 1))
    # 计算预测值
    y_pred = np.dot(x_poly, beta) + beta0
    return y_pred

# 使用模型进行预测
rain_new = np.array([1.7, 1.9, 2.1, 2.3, 2.5, 2.7, 2.9, 3.1, 3.3, 3.5,
                     3.7, 3.9, 4.1, 4.3, 4.5, 4.7, 4.9, 5.1, 5.3, 5.5,
                     5.7, 5.9, 6.1, 6.3, 6.5])
next_hour_change_pred = regression_function(rain_new)
print("预测下一个小时的水位变化:", next_hour_change_pred)

# 打印多元非线性回归模型函数
print("多元非线性回归模型函数:")
print("f(x) =", end=" ")

for i in range(len(beta)):
    if i == 0:
        print(f"{beta0:.4f}", end=" ")
    else:
        print(f"+ {beta[i]:.4f} * x^{i}", end=" ")
