## 机器学习流水线以及推荐引擎
### 机器学习流水线
（有点不太明白把这个流水线和推荐引擎放在同一节有什么意义）
> 机器学习系统中的主要组成部分是数据处理流水线。   
在数据被训练之前，需要经过各种处理，为了得到一个准确的、可扩展的机器学习系统，需要一条健壮的流水线。   
> 有很多基本的函数功能可以使用，通常数据处理流水线就是这些基本函数的组合，不推荐使用嵌套或循环的方式调用这些函数而是用函数式编程的方式构建函数组合。    

- 为数据处理构建函数组合   
1.定义第一个函数、第二个函数乃至第N个函数   
2.定义一个函数组合器，将这些函数作为输入参数，返回一个组合函数。
```python
def function_composer(*args):
  return reduce(lanbda f,g: lambda x:f(g(x)),args)
  #暂时还没有看懂他这个return是怎么回事，只知道reduce是一个能组合子函数的函数但是不知道这个函数的用法
```
```
reduce() 函数
对参数序列中元素进行累积。

函数将一个数据集合（链表，元组等）中的所有数据进行下列操作：
用传给 reduce 中的函数 function（有两个参数）先对集合中的第 1、2 个元素进行操作
得到的结果再与第三个数据用 function 函数运算，最后得到一个结果。

语法
reduce(function, iterable[, initializer])

参数
function -- 函数，有两个参数
iterable -- 可迭代对象
initializer -- 可选，初始参数

返回值
返回函数计算结果。
```
3.调用组合函数的方法     
```python
func_composed=function_composer(fun1,fun2,fun3,...)
```

- 一个常用的推荐函数库 scikit-learn库    
> 包含了构建机器学习流水线的方法   
>> 机器学习流水线：预处理、特征选择、监督式学习、非监督式学习   
> 只需要指定函数，它就会构建一个组合对象，使数据通过整个流水线。   
也就是说这个库是可以完成上面那个小点里做的工作。    
具体步骤：
1.选择k个最好的特征值，k是一个随机值，可以自己设置。   
```python
selector_k_best=SelectKBest(f_regression,k=10)   
```
2.用随机森林分类器分类数据。   
```python
classifier=RandomForestClassifier(n_estimators=50,max_depth=4)
```
3.创建流水线。   
```python
pipeline_classifier=Pipeline([('selector',selector_k_best),('rf',classifier)])
```
4.训练分类器。    
5.为训练数据预测结果。    
6.评价分类器的性能。    

- 寻找最近邻    
> 最近邻模型是一个通用算法类，其目的是根据训练数据集中的最近邻数量来做决策。   
```python
#knn分类器
classifier=neighbors.KNeighborsClassifier(num_neighbors,weights='distance')
classifier.fit(x,y)
#提取knn分类的结果
dist,indices=classifier.kneighbors(test_datapoint)
#knn回归器

```


### 推荐引擎的分类

- 协同过滤 collaboration filtering  
> 从当前用户过去的行为和其他用户对当前用户的评分来构建模型，从模型来预测这个用户可能感兴趣的内容。     

我感觉这个可能就是，计算用户之间的相似信息，用户可以看成是一个信息的集合，我认为协同过滤计算的是两个集合的公共部分。      
类似的应用，比如网易云音乐吧，虽然表面上给用户推荐的是音乐（日推歌曲），但是原理好像是通过口味相同的用户推荐的，就是读取了跟当前用户匹配度比较高的用户的播放列表，来做推荐。


- 基于内容的过滤 content-based filtering
> 用商品本身的特性来给用户推荐更多的商品，商品间的相似度是模型的主要关注点。   

这里的“商品”可以指代任何产品吧，又或者只是翻译的偏差？我觉得它是上面协同过滤的信息集合的小单元。这种算法计算的是各个小单元之间的相似度。   
类似的应用，就是淘宝/抖音/还有各种广告应用吧，感觉淘宝总推送已经购买过的相似产品这一点其实并不好，我猜想它可能是由搜索记录生成的推荐列表，但它没有把已购买商品从这个列表中删除感觉不好，因为刚刚购买过的商品短时间之内应该是没有这个需求了。


### 建立电影推荐系统    
- 计算欧氏距离分数  
```python
def euclidean_score(dataset, user1, user2):
    if user1 not in dataset:
        raise TypeError('User ' + user1 + ' not present in the dataset')

    if user2 not in dataset:
        raise TypeError('User ' + user2 + ' not present in the dataset')

    # Movies rated by both user1 and user2
    rated_by_both = {} 

    for item in dataset[user1]:
        if item in dataset[user2]:
            rated_by_both[item] = 1

    # If there are no common movies, the score is 0 
    if len(rated_by_both) == 0:
        return 0

    squared_differences = [] 

    for item in dataset[user1]:
        if item in dataset[user2]:
            squared_differences.append(np.square(dataset[user1][item] - dataset[user2][item]))
        
    return 1 / (1 + np.sqrt(np.sum(squared_differences)))
```
- 计算皮尔森相关系数  
```python
def pearson_score(dataset, user1, user2):
    if user1 not in dataset:
        raise TypeError('User ' + user1 + ' not present in the dataset')

    if user2 not in dataset:
        raise TypeError('User ' + user2 + ' not present in the dataset')

    # Movies rated by both user1 and user2
    rated_by_both = {}

    for item in dataset[user1]:
        if item in dataset[user2]:
            rated_by_both[item] = 1

    num_ratings = len(rated_by_both) 

    # If there are no common movies, the score is 0 
    if num_ratings == 0:
        return 0

    # Compute the sum of ratings of all the common preferences 
    user1_sum = np.sum([dataset[user1][item] for item in rated_by_both])
    user2_sum = np.sum([dataset[user2][item] for item in rated_by_both])

    # Compute the sum of squared ratings of all the common preferences 
    user1_squared_sum = np.sum([np.square(dataset[user1][item]) for item in rated_by_both])
    user2_squared_sum = np.sum([np.square(dataset[user2][item]) for item in rated_by_both])

    # Compute the sum of products of the common ratings 
    product_sum = np.sum([dataset[user1][item] * dataset[user2][item] for item in rated_by_both])

    # Compute the Pearson correlation
    Sxy = product_sum - (user1_sum * user2_sum / num_ratings)
    Sxx = user1_squared_sum - np.square(user1_sum) / num_ratings
    Syy = user2_squared_sum - np.square(user2_sum) / num_ratings
    
    if Sxx * Syy == 0:
        return 0

    return Sxy / np.sqrt(Sxx * Syy)
```
- 生成推荐用户 
在数据集中搜索匹配度高的内容，并且输出

```
python中的单下划线
按照习惯，有时候单个独立下划线是用作一个名字，来表示某个变量是临时的或无关紧要的。

例如，在下面的循环中，我们不需要访问正在运行的索引，我们可以使用“_”来表示它只是一个临时值：

 for _ in range(32):
   print('Hello, World.')
```
