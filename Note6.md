## 自然语言处理，即文本的分析      
（这节主要是英文的语言处理，不知道跟中文语言处理是否有共同之处。因为先前好像在什么文献资料里面看到过说中文自然语言的处理跟英文不同，难度更高。这个说法 还需要进一步求证）       
自然语言处理的常用包：Natural Language Toolkit(NLTK)       
### 标记解析    
> 将文本分割成一组有意义的片段的过程。
>> 有意义的片段 即 标记    
- 句子解析器
```python
sent_tokenize(text)
```

- 单词解析器（3）    
```python
sent_tokenize(text)

#punkt word 以标点符号分割文本
punkt_word_tokenizer=PunktWordTokenizer()
punkt_word_tokenizer.tokenize(text)

#wordpunct 将标点符号保留到不同句子标记中
```
 
 ### 提取词干   
 感觉是提取单词的原型，但他说是提取一个词的词干，就和原型有区别，下边还有一个词形还原。   
 > 例如 playing player plays 提取出play    
 - 3个词干提取器    
 ```python
 stemmers=['PORTER','LANCASTER','SNOWBALL']
 
 #'PORTER'最宽松
 stemmer_porter=PorterStemmer()
 
 #'LANCASTER'最严格 得到的词干往往比较模糊，速度很快，它会减少单词的很大部分
 stemmer_lancaster=LancasterStemmer()
 
  #'SNOWBALL' 通常使用这个
 stemmer_snowball=SnowballStemmer('english')
 #看snowball的参数里有english，感觉可能这个提取器还支持其他语言。
 ```
 
 ### 词形还原  
 分块的方法划分文本。基于**任意随机条件**将输入文本分割成块。   
 
 ### 创建词袋模型  bagof-words    
 是从所有文档单词中学习词汇的模型。    
 学习之后，磁带通过构建文档中所有单词的直方图来对每篇文档进行建模。      
 ***    
 **以上所有步骤就类似于之前数据分析中，数据处理的部分**    
 
 ### 创建文本分类器   
 将文本文档分为不同的类   
 > tf-idf的统计数据，他表示词频-逆文档频率（term frequency-inverse document frequency）   
 > 这个统计工具有助于理解一个单词在一组文档中对某一个文档的重要性。    
 > 常用于信息检索领域。   
 
 词频 the term frequency(TF) 单词在给定文档出现的频次    
 逆文档频率 inverse document frequency (IDF) 给定单词的重要性  
 
 ### 性别识别  
 - 提取输入单词的特征     
 - 设置随机生成树的种子值，并混合搅乱训练数据
 - 输入训练数据和测试数据，用贝叶斯分类器做分类
 
 ### 分析句子情感   
 - 提取数据，将数据分为积极评论和消极评论
 - 将所有评论分为训练集和测试集，提取特征
 - 使用贝叶斯分类器做分类   
 
---
**意外处理——虚拟机不能联网的问题**    
*问题解决参考 百度经验*    
- 关闭虚拟机。在虚拟机的编辑菜单里，点击"虚拟网络编辑器"
- 在虚拟网络编辑器界面，直接点击左下角的回复默认默认设置，然后点击确定。此时虚拟机会自用重装虚拟网卡并重新设置网卡设置。这一步完成之后，需要重新回到第一步的设置，设置网络连接选项为NAT模式，这样就可以使虚拟机能联网了。
