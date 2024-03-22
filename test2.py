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
print((water_level[:-4], rain[4:]))
# 构建自变量矩阵X，包括water_level和过去24小时的rain
X = np.column_stack((water_level[:-4], rain[4:]))

# 判断样本数据的维度是否为零
if X.shape[0] > 0:
    # 创建多项式特征
    degree = 2  # 多项式特征的阶数
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    X_poly = poly.fit_transform(X)

    # 创建并训练多项式回归模型
    model = LinearRegression()
    model.fit(X_poly, next_hour_change[4:])

    # 输出回归系数
    beta0 = model.intercept_
    beta = model.coef_
    print("回归系数:")
    print("beta0:", beta0)
    print("beta:", beta)

    # 打印多项式
    print("多项式:")
    poly_features = poly.get_feature_names_out(['water_level', 'past_rain'])
    for p, feature in zip(beta, poly_features):
        print(f"{feature}: {p}")

    # 测试用例
    test_rain = np.array([0.2, 3.5, 5.6])  # 测试用例的rain数据
    test_water_level = np.array([8.7, 8.4, 8.2])  # 测试用例的water_level数据
    test_X = np.column_stack((test_water_level, test_rain))
    test_X_poly = poly.transform(test_X)  # 使用transform方法将测试数据转换为多项式特征
    test_predictions = model.predict(test_X_poly)  # 使用训练好的模型进行预测
    print("测试用例的预测结果:")
    print(test_predictions)

else:
    print("样本数据的维度为零，无法进行多项式回归")