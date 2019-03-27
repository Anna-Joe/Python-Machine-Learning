### 数据预处理   
##### 导入数据   
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

##### 处理数据   
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

##### 标记编码   
感觉标记编码也挺有意思的，有点像预定语法规则转化。   

### 三种回归模型
- 决策树 Decision Tree   
```python

```
- 自适应增强 Adaptive Boost   
```python

```
- 随机森林 Random Forest   
```python

```
