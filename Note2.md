## 分类器和分析曲线

### 两种分类器
- 逻辑回归分类器
```python
classifier=linear_model.LogisticRegression(solver='liblinear',c=100)
```
- 朴素贝叶斯分类器
```python
classifier_gaussiannb=GaussianNB()
```

### 两种曲线   
- 验证曲线    
帮助理解每个超参数，例如n_estimators,max_depth，对训练得分的影响。如果改变超参数，能够看到分类器性能的变化，就说明这个超参数对分类器是有价值的。
```python
#n_estimators验证曲线
classifier=RandomForestClassifier(max_ depth=4, random_ state=7)
parameter_ grid=np. linspace(25 ,200, 8). astype(int)
train_ scores, validation_ scores=validation_ curve(classifier ,x,y,"n_ estimators" ,parameter_ grid,cv=5)
print("n#### Validation Curve ####")
print("nParam:n_ estimators(nTraining scores:(n" ,train_ scores)
print("nParam:n_ estimatorsnValidation scores: n" , validation_ scores)
```
```python
#max_depth验证曲线
classifier=RandomForestClassifier(n_ estimators=20, random_ state=7)
parameter_ grid=np. linspace(2,10,5). astype(int)
train_ scores, validation_ scores=validation_ curve(classifier ,x,y,"n_ estimators" , parameter_ grid,cv=5)
print("n#### Validation Curve ####")
print("nParam:max_ depthnTraining scores:n" , train_ scores)
print("nPar. am:max_ depthnValidation scores: 'n" ,validation_ scores)
```
- 学习曲线    
帮助我们理解训练集大小对模型的影响。   
```python
classifier=RandomForestClassifier(random_ state=7)parameter_ grid=np.array( [200 , 500, 800,1100])
train_ sizes, train_ scores ,validation_ scores=learning_ curve(classifier ,x,y,train_ sizes=parameter_ grid,cv=5)
print("n#### Learning Curve ####" )
print("(nTraining scores:(n",train_ scores)
print("nValidation scores:(n",validation_ scores)
```

### 分类（多维）函数绘制    
```python
def plot_classifier(classifier,x,y):
  #定义图形的取值范围 
  x_ min,x_ max=min(x[: ,0])-1,max(x[:,])+1.0
  y_ min,y_ max=min(x[: ,1])-1,max(x[: ,0])+1.0
      
  #设置网格数据的步长
  step_ size=0.01
      
  #定义网络
  x_ values,y_ values=np . meshgrid( np.arange(x_ min,x_ max,step_ size) ,np.arange(y_ min,y_ max,step_ size))
      
  #计算分类器输出结果
  mesh_ output=classifier . predict(np.c_ _[x_ _values.ravel(),y_ _values . ravel()])
      
  #数组维度变形
  mesh_ output=mesh_ output. reshape(x_ values . shape )
    
  #用彩图画出分类结果
  plot. figure()
  
  #选择配色方案
  plot . pcolormesh(x_ values,y_ _values ,mesh_ output, cmap=plot.cm.gray)
      
  #把数据点画在图上
  plot. scatter(x[:,0],x[:,1],c=y, s=80 , edgecolors= ' black' ,linewidth=1 , cmap=plot.cm. Paired)
    
  #设置图形的取值范围
  plot.xlim(x_ values.min(),x_ _values .max())
  plot.ylim(y_ _values .min(),y_ values .max())

  #设置x轴和y轴
  plot.xticks((np. arange(int(min(x[: ,0])-1),int(max(x[:,0])+1),1.0)
  plot.yticks( (np. arange(int(min(x[:,1])-1),int(max(x[:,1])+1),1.0)))
  plot. show()
```

  
