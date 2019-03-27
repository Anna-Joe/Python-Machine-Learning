## 数据预处理   
### 导入数据   
- 手动导入   
```python
data=numpy.array([[],[],[]])
```
- 文件导入
```python
#按行读取txt
with open(filename,'rb') as f:
   f.readlines()
   
#读取csv
file_reader=csv.reader(open(filename,'rt'),delimiter=',')
```

### 处理数据   
- 均值移除 Mean removal   
消除特征彼此间的偏差。   
```python
data_standardized=sklearn.scale(data)
data_standardized.mean(axis=0)
data_standardized.std(axis=0)
```
- 范围缩放 Scaling   
将所有特征值缩放在一个范围内。   
```python
data_scaler=sklearn.MinMaxScaler(feature_range=(0,1))
data_scaled=data_scaler.fit_transform(data)
```
- 归一化 Normalization   
最常用的归一化形式是L1范数，使得特征向量的数值之和为1。   
```python
data_normalized=sklearn.normalize(data,norm='l1')
```
- 二值化 Binarization   
将特征向量转化为布尔型向量。   
```python
data_binarized=sklearn.Binarizer(threshold=1.4).transform(data)
```
- 独热编码 One-Hot Encoding   
没看懂什么意思。大概就是一种对特征值处理的规则吧。   
```python

```

### 标记编码   
感觉标记编码也挺有意思的，就是给字符串编码，可能后面处理自然语言会用到吧。   
```python
label_encoder=sklearn.LabelEncoder()
classes=['','','']#labelEncoder的对象
label_encoder.fit(input_classes)
```

## 数据分析
### 建立回归器   
```python
#选择给定集合中部分数据作为训练数据，部分数据作为测试数据
num_training=int(0.8*len(x))
num_test=len(x)-num_training

#训练数据
x_train,y_train=x[:num_training],y[:num_training]

#测试数据
x_test,y_test=x[num_training:],y[num_training:]
```
> 好像手写数据等号右边要加np.array()，导入的文件数据不用

### 三种回归模型
- 决策树 Decision Tree   
```python
dt_regressor=DecisionTreeRegressor(max_depth=4)
dt_regressor.fit(x_train,y_train)
```
- 自适应增强 Adaptive Boost   
```python
ab_regressor=AdaBoostRegressor(DecisionTreeRegressor(max_depth=4),n_estimators=400,random_state=7)
ab_regressor.fit(x_train,y_train)
```
> 这几个数据不知道有什么讲究，还有要特别注意这里的s，非常容易漏掉！

- 随机森林 Random Forest   
```python
rf_regressor=RandomForestRegressor(n_estimators=1000,max_depth=10,min_samples_split=2)
rf_regressor.fit(x_train,y_train)
```

### 模型误差测量  
- 均方误差 给定数据集的所有数据点的误差的平方的平均值
mse = mean_squared_error(y_test,y_pre)

- 解释方差分 这个分数用于衡量我们的模型对数据集波动的解释能力。分数为1.0表示模型是完美的。   
evs=explained_variance_score(y_test,y_pre)
