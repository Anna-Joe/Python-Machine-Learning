## 隐马尔科夫模型下的语音识别    

### 隐马尔科夫模型（Hiden Markov Models,HMM)
> 隐马尔科夫模型非常擅长建立时间序列数据模型。因为一个音频信号同时也是一个时间序列信号，因此隐马尔科夫模型也同样适用于音频信号的处理。假定输出是通过隐藏状态生成的，我们的目标是找到这些隐藏状态，以便对信号建模。
- 创建一个隐马尔科夫模型类      

1.初始化该类。使用高斯隐马尔科夫模型(Gaussian HMMs)来对数据建模。
```python
class HMMTrainer(object):
    def __init__(self, model_name='GaussianHMM', n_components=4, cov_type='diag', n_iter=1000):
    #参数n_components定义了隐藏状态的个数，参数cov_type定义了转移矩阵的协方差类型，参数n_iter定义了训练的迭代次数。
        self.model_name = model_name
        self.n_components = n_components
        self.cov_type = cov_type
        self.n_iter = n_iter
        self.models = []
```
2.用以下参数定义模型
```python
        if self.model_name == 'GaussianHMM':
            self.model = hmm.GaussianHMM(n_components=self.n_components, 
                    covariance_type=self.cov_type, n_iter=self.n_iter)
        else:
            raise TypeError('Invalid model type')
```
3.训练数据。x是二维数组，每一行是13维。
```python
    def train(self, X):
        np.seterr(all='ignore')
        self.models.append(self.model.fit(X))
```
4.基于该模型定义一个提取分数的方法。
```python
    def get_score(self, input_data):
        return self.model.score(input_data)
```

### 创建一个语音识别器   
- 为每一个类构建一个隐马尔科夫模型，识别新输入的文件中的单词，需要对该文件运行所有的模型，并找出最佳分数结果。   
*用到的包*
```python
import os
import argparse 

import numpy as np
from scipy.io import wavfile 
from hmmlearn import hmm
from features import mfcc
```
1.定义一个函数来解析命令行中的输入参数   
```python
def build_arg_parser():
    parser = argparse.ArgumentParser(description='Trains the HMM classifier')
    parser.add_argument("--input-folder", dest="input_folder", required=True,
            help="Input folder containing the audio files in subfolders")
    return parser

```
2.定义main函数，解析输入参数，初始化隐马尔科夫模型的变量
```python
if __name__=='__main__':
    args = build_arg_parser().parse_args()
    input_folder = args.input_folder

    hmm_models = []
```
3.解析音频文件。     
输入路径》》提取子文件夹名称》》做标记
```python
 for dirname in os.listdir(input_folder):
        # Get the name of the subfolder 
        subfolder = os.path.join(input_folder, dirname)

        if not os.path.isdir(subfolder): 
            continue

        # Extract the label
        label = subfolder[subfolder.rfind('/') + 1:]

        # Initialize variables
        X = np.array([])
        y_words = []
```
4.迭代每一个子文件夹中的音频文件。     
读取》》提取MFCC特征》》添加标记信息
```python
        for filename in [x for x in os.listdir(subfolder) if x.endswith('.wav')][:-1]:
            # 读取
            filepath = os.path.join(subfolder, filename)
            sampling_freq, audio = wavfile.read(filepath)
            
            # 提取MFCC特征
            mfcc_features = mfcc(audio, sampling_freq)

            # 添加标记信息1
            if len(X) == 0:
                X = mfcc_features
            else:
                X = np.append(X, mfcc_features, axis=0)
            
            # 添加标记信息2
            y_words.append(label)
```
5.训练并保存隐马尔科夫模型。
```python
        print 'X.shape =', X.shape
        # Train and save HMM model
        hmm_trainer = HMMTrainer()
        hmm_trainer.train(X)
        
        # ！！！注意这一步是错误的
        hmm_models.append((hmm_trainer, label))
        
        hmm_trainer = None
```
hmm_models是dict类型，dict类型不支持append方法。在dict类型中添加新项可以参考C++中map的用法。这里应该改成:
```python
        hmm_models[hmm_trainer]=label
```
6.进行语音识别。      
获取一个未用于训练的测试文件列表》》解析输入的文件》》读取每个音频文件》》提取MFCC特征》》迭代所有模型》》提取分数》》打印结果
```python
    # 获取一个未用于训练的测试文件列表
    input_files = [
            'data/pineapple/pineapple15.wav',
            'data/orange/orange15.wav',
            'data/apple/apple15.wav',
            'data/kiwi/kiwi15.wav'
            ]

    # 解析输入的文件
    for input_file in input_files:
        # 读取每个音频文件
        sampling_freq, audio = wavfile.read(input_file)

        # 提取MFCC特征
        mfcc_features = mfcc(audio, sampling_freq)

        # 定义变量 最大分数和输出标记 最大分数的定义是错的
        max_score = None
        output_label = None
```
如果把max_score定义成None，后面比较的时候会出现类型不符合的错误。经过输出score之后确认，score都是负值，所以这里我把max_score初始化定义为：
```python
        max_score=-65535
```

```python
        # 迭代所有模型 
        for item in hmm_models:
            hmm_model, label = item
            score = hmm_model.get_score(mfcc_features)
            if score > max_score:
                max_score = score
                output_label = label
```
这里的迭代过程也是错的。dict类型的hmm_models是无法被如此迭代的。必须将key和value分开用两个值封装。我将迭代过程修改如下：
```python
        # 迭代所有模型 
        for key,value in hmm_models.items():
            hmm_model=key
            label = value
            score = hmm_model.get_score(mfcc_features)
            if score > max_score:
                max_score = score
                output_label = label
```

```python
        # 打印结果
        print "\nTrue:", input_file[input_file.find('/')+1:input_file.rfind('/')]
        print "Predicted:", output_label 

```
