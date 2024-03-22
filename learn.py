"""
@Title: linear_regression_with_polynomial
@Time: 2024/2/29 19:41
@Author: Michael
"""
 
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
 
# 数据集，y = 1.4 * x ** 2 - 3.1 * x + 2.6
x = np.random.uniform(-3, 3, (100, 1))
y = 1.4 * x ** 2 - 3.1 * x + 2.6 + np.random.normal(0, 0.5, (100, 1))
print("输入数据集：({x}, {y})".format(x=x, y=y))
# 预处理数据集，将一元二次函数转化成三元一次函数，然后使用线性回归训练
poly = PolynomialFeatures(degree=2)
poly.fit(x)
x = poly.transform(x)
# 手动实现预处理
degree = np.array([[0, 1, 2]])
# x = x ** degree
 
# 回归训练
linear = LinearRegression()
linear.fit(x, y)
print("训练后参数为：({w}, {b})".format(w=linear.coef_, b=linear.intercept_))
print("输入10的预测值为：{y}".format(y=linear.predict(np.array([[1, 10, 100]]))))
 
"""
训练后参数为：([[ 0.         -3.1180901   1.40622675]], [2.62986504])
输入10的预测值为：[[112.07163862]]
"""