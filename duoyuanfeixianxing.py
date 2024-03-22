import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt

# 过去 N 小时降雨数据作为参数
N = 0

# 读取 Excel 文件数据
def read_excel_data(file_path):
    df = pd.read_excel(file_path)
    # TP = pd.to_datetime(df['TP'])
    # TQ = pd.to_datetime(df['TQ'])
    rain1 = df['drp_8306'].values
    rain2 = df['drp_02938'].values
    water_level = df['rz'].values
    TM = df['tm'].values
    next_hour_change = np.diff(water_level)  # 计算下一个小时的水位变化
    return rain1, rain2, water_level, next_hour_change, TM

# 处理水位和降雨数据
def handle_water_level_rain1_rain2(water_level, rain1, rain2, next_hour_change , TM):
    new_array = []
    new_next_hour_change = []
    new_TM = []
    used_water_level = []
    # 遍历 water_level 列表
    for i in range(len(water_level)-1):
        if (i < N-1):
            continue
        # 获取当前水位的索引
        water_level_index = i
        # 获取与当前水位对应的四个降雨量的索引
        rain_indices = list(range(i, i-N, -1))
        # 使用 np.take 函数从 rain 列表中获取对应索引的值，并将它们组成一个新的数组
        combined_array1 = np.take(rain1, rain_indices)
        combined_array2 = np.take(rain2, rain_indices)
            # 检查数据是否为NAN
        if np.isnan(combined_array1).any() or np.isnan(combined_array2).any() or np.isnan(water_level[water_level_index]):
            print('存在NAN值的数据:', TM[i] , combined_array1, combined_array2, water_level[water_level_index])
            continue
        # 将组合后的数组与当前水位值一起添加到新的数组中
        new_array.append([water_level[i]] + list(combined_array1)+ list(combined_array2) )
        new_next_hour_change.append(next_hour_change[i])
        new_TM.append(TM[i])
        used_water_level.append(water_level[i])
    return new_array, new_next_hour_change,used_water_level, new_TM

# 输入文件路径
file_path = '190101230101.xlsx'
test_file_path = '230101240101.xlsx'

# # 读取 Excel 文件数据
rain1,rain2, water_level, next_hour_change, TM = read_excel_data(file_path)
testRain1,testRain2,test_water_level, test_next_hour_change, test_TM = read_excel_data(test_file_path)

# 分别处理训练水位和降雨数据 测试水位和降雨数据
new_array, new_next_hour_change, used_water_level,new_TM = handle_water_level_rain1_rain2(water_level, rain1, rain2, next_hour_change, TM)
test_array, test_next_hour_change,test_used_water_level,test_TM = handle_water_level_rain1_rain2(test_water_level, testRain1, testRain2, test_next_hour_change, test_TM)


# 将输入特征转换为多项式特征
poly = PolynomialFeatures(degree=2)
new_array_poly = poly.fit_transform(new_array)
print('new_array_poly', new_array_poly)
model = LinearRegression()
model.fit(new_array_poly, new_next_hour_change)
# test_array = np.array([[7.82, 0, 2.37, 0.37, 0, 0.37, 0,0, 2.37, 0.37, 0, 0.37, 0],[7.79, 0,0, 2.37, 0.37, 2.37, 0.37, 0,0, 2.37, 0.37, 2.37, 0.37]])
test_array_poly = poly.fit_transform(test_array)
pre = model.predict(test_array_poly)

# 打印多项式
features = poly.get_feature_names_out()
equation = "f(x) = "
for i, feature in enumerate(features):
    if i == 0:
        equation += "{:.2f}".format(model.intercept_)
    else:
        equation += " + {:.2f}{}".format(model.coef_[i], feature)
print(equation)
pre_res = []
    
pre = np.round(pre, 4)

pre_water_level = [test_used_water_level[0]]
for i in range(len(pre)):
    pre_water_level.append(test_used_water_level[i] + pre[i])
    pre_res.append(test_used_water_level[i] - pre_water_level[i])
    if (pre_res[i] > 0.15 or pre_res[i] < -0.15):
        print('预测值:', pre[i], '实际值:', test_next_hour_change[i], '残差:', pre_res[i], '时间:', test_TM[i], '降雨量:', test_array[i])
pre_res = np.round(pre_res, 4)
test_used_water_level.append(pre_water_level[-1])
test_TM.append("2024-01-01")
# 绘制预测值、残差和实际下一小时水位变化
plt.figure(figsize=(12, 6))
# plt.plot(pre, label='pre')
plt.plot(test_TM, test_used_water_level, label='test_water_level')
plt.plot(test_TM, pre_water_level, label='pre_water_level')
plt.xlabel('tm')
plt.ylabel('water_level')
plt.title('预测值、残差和实际下一小时水位变化')
plt.legend()
plt.show()

print('预测值', pre)
print('残差', pre_res)
