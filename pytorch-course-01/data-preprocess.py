# 数据预处理

import pandas as pd

data = pd.read_csv('data/house_tiny.csv')
print(data)
print(data.head())

# index location
inputs,outputs = data.iloc[:,0:2],data.iloc[:,2]
print(inputs)
print(outputs)

# location
print('----- loc -------')

# 第一个参数为选择的行（切片）
# 第二个参数是选择的列名
# 通常该方法实用于 iloc
print(data.loc[:,['NumRooms','Alley']])

## 1、loc[]函数接收的是行/列的名称（可以是整数或者字符），
## iloc[]函数接收的是行/列的下标（从0开始），不能是字符。
## 2、loc[]函数在切片时是按闭区间切片的，也就是区间两边都能取到，
## iloc[]函数则是按传统的左闭右开的方式切片的。

print('--- end loc -----')

# 这里的 fillna 即 Fill NA 填充 N/A 占位的值，mean()
# 的意思为 “平均数”
# ----------------------------------------------
# 这里按照课程中的讲解，直接使用 inputs.mean() 将会报错
# 所以需要在 mean 方法中加入 numeric_only 参数作为指示
# 这里的 numeric_only 指的是 “只对数字进行处理”

inputs = inputs.fillna(inputs.mean(numeric_only=True))
print(inputs)
