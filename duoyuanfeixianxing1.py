import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# 读取 Excel 文件数据
def read_excel_data(file_path, sheet_name):
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    # TP = pd.to_datetime(df['TP'])
    # TQ = pd.to_datetime(df['TQ'])
    rain = df['P'].values
    water_level = df['Q'].values
    next_hour_change = np.diff(water_level)  # 计算下一个小时的水位变化
    return rain, water_level, next_hour_change

# 输入文件路径
file_path = 'data.xlsx'

# 读取 Excel 文件中的每张表的数据
dfs = pd.read_excel(file_path, sheet_name=None)

rain_list = []
water_level_list = []
next_hour_change_list = []

for sheet_name, df in dfs.items():
    rain, water_level, next_hour_change = read_excel_data(file_path, sheet_name)
    rain_list.append(rain)
    water_level_list.append(water_level)
    next_hour_change_list.append(next_hour_change)

# 创建一个空列表来存储结果
new_array_list = []

# 遍历每张表的 water_level 列表
for water_level in water_level_list:
    new_array = []
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
    new_array_list.append(new_array)

# 将输入特征转换为多项式特征，并根据每张表的 new_array_list 进行预测
poly = PolynomialFeatures(degree=2)
pre_list = []
for new_array in new_array_list:
    new_array_poly = poly.fit_transform(new_array)
    model = LinearRegression()
    model.fit(new_array_poly, next_hour_change)
    pre = model.predict(new_array_poly)
    pre_list.append(pre)

# 打印多项式
features = poly.get_feature_names_out()
equation = "f(x) = "
for i, feature in enumerate(features):
    if i == 0:
        equation += "{:.2f}".format(model.intercept_)
    else:
        equation += " + {:.2f}{}".format(model.coef_[i], feature)
print(equation)

# 打印预测值和残差
for i, pre in enumerate(pre_list):
    pre_res = next_hour_change_list[i] - pre
    pre = np.round(pre, 4)
    pre_res = np.round(pre_res, 4)
    print(f"第{i+1}张表预测值：", pre)
    print(f"第{i+1}张表残差：", pre_res)
