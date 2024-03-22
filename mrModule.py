import pandas as pd
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
# 创建一个空列表来存储结果
new_array = []

# 遍历 water_level 列表
for i in range(len(water_level)):
    if (i < 3):
        continue
    # 获取当前水位的索引
    water_level_index = i
    # 获取与当前水位对应的四个降雨量的索引
    rain_indices = list(range(i, i-4, -1))
    # 使用 np.take 函数从 rain 列表中获取对应索引的值，并将它们组成一个新的数组
    combined_array = np.take(rain, rain_indices)
    # 将组合后的数组与当前水位值一起添加到新的数组中
    new_array.append([water_level[i]] + list(combined_array))
new_next_hour_change= next_hour_change[3:]
model = LinearRegression()
model.fit(new_array, new_next_hour_change)
pre = model.predict(new_array)
print('系数', model.coef_)
print('截距', model.intercept_)
print('预测值', pre)