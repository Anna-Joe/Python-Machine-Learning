## 聚类算法   
最常见的无监督学习方法就是聚类
- k-means算法聚类数据   
```python
kmeans=KMeans(init="k-means++",n_clusters=num_clusters,n_init=10)
kmeans.fit(data)
```
- k-means算法的应用——矢量量化    
**(1)解析输入参数 函数**   
输入参数 是指每个像素被压缩的比特数      
**(2)压缩图片 函数**    
实质是将图片的像素转化为（样本量，特征值）数组       
运用k-means聚类算法将像素点归类 （感觉这一步是压缩图片也就是矢量量化的核心）   
**(3)计算压缩算法对图片质量的影响**    

- 均值漂移聚类模型   
用于集群数据点    
> 该算法吧数据点的分布看成是概率密度函数，希望在特征空间中个根据函数分布特征找出数据点的“模式”。
> 这些模式就对应于一群群局部最密集分布的点。
> 优势是不需要事先确定集群的数量    
```python
#创建均值漂移模型
bandwidth=estimate_bandwidth(x,quantile=0.1,n_sample=len(x))
meanshift_eatimator=MeanShift(bandwidth=bandwidth,bin_seeding=True)

#训练模型
meanshift_estimator.fit(x)

#提取标记
labels=meanshift_estimator.labels_

#提取集群中心点，打印集群数量
centroids=meanshift_estimator.cluster_centers_
num_clusters=len(np.unique(labels))
```
- 凝聚层次聚类  
层次聚类：通过不断地分解或者合并集群来构建树状集群    
层次聚类的结构可以用一颗树表示       
多次层次聚类合并称一个巨型集群就叫 凝聚层次聚类    
```python
#一个实现凝聚层次聚类的函数
def perform_clustering(x,connectivity,title,num_clusters=3,linkage='ward'):
  plot.figure()
  model=AgglomerativeClustering(linkage=linkage,connectivity=connectivity,n_clusters=num_clusters)
  model.fix(x)
  
#提取标记，指定不同类在图中的标记
labels=model.lebels_

#迭代数据，画图
for i,marker in zip(range(num)clusters),markers):
  plot.scatter(x[labels==1,0],x[labels==i],s=50,marker=marker,color='k',facecolord='none')
plot.title(title)
```

- DBSCAN算法自动估算集群数量    
(DensityBased Spatial Clustering of Applications with Noice)带噪声的基于密度的聚类方法     
