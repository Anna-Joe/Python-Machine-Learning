### 部分问题解答   
- 为什么要使用numpy.array()？
```
python中list和array都只支持一维数组，而且没有各种运算函数，不适合数值计算。    
numpy.array()支持多维数组，并内置许多方便运算的函数。    
```

- fit()  和  transform()的区别  
```
fit()求得训练集x的均值、方差、最大最小值这些固有属性。    
transform()在fit()的基础上进行标准化、降维、归一化。    
不同算法的fit()和transform()的处理过程是不同的。 
```

- python中的*和** 
```
(1)算数运算符   
*表示乘法 2*5=10
**表示乘方 2**5=2^5=32 
```
```
(2)函数参数（包括形参和实参）   
*表示多个无名参数( , , , , ) 相当于参数列表是tuple     
**表示多个关键字参数{ : , : , : , }相当于参数列表是dict    
```
- enumerate()函数   
```
用于将一个可遍历的数据对象组合为一个索引序列，同时列出数据和下标。   
```

- 特别注意   
```
sklearn里cross_validation这个模块已经被废弃。   
如果需要使用其中的函数，例如train_test_split(),cross_validation()，只需要导入     
from sklearn.model_selection import *    
导入上述包之后，其中的函数可以直接使用，不用加前缀cross_validation。    
```

## 预测建模  
SVM(Support Vector Machines,支持向量机)   理解SVM才是预测建模的重点   
通过对数学方程组求解，可以找出两组数据之间的最佳分割边界。    
- 建立线性分类器    
```python
params={'kernel':'linear'}
classifier=SVC(**params)
```
- 建立非线性分类器    
```python
#三次多项式方程
params={'kernel':'poly','degree':3}
#径向基函数   
params={'kernel':'rbf'}
classifier=SVC(**params)
```
