## 使用OpenCV-Python库做图像处理   
### 展示图像、剪裁、调整大小
0.使用到的python包
```python
import sys
import cv2
import numpy as np
```
1.展示图像   
```python
input_file = sys.argv[1]
img = cv2.imread(input_file)
cv2.imshow('Original', img)
```
![运行结果](https://github.com/Anna-Joe/Python-Machine-Learning/blob/master/deal-img-original.png)      

2.剪裁    
这个函数好像有点问题，没有报错，但是并没有实现剪裁的功能，它只是按照所给的尺寸缩小了图像。原因不详。   
```python
h, w = img.shape[:2]
start_row, end_row = int(0.21*h), int(0.73*h)
start_col, end_col= int(0.37*w), int(0.92*w)
img_cropped = img[start_row:end_row, start_col:end_col]
cv2.imshow('Cropped', img_cropped)
```
![运行结果](https://github.com/Anna-Joe/Python-Machine-Learning/blob/master/deal-img-cropped.png)      

3.Uniform方法调整整体大小为原图的1.3倍    
也就是说长宽都调整为原图的1.3倍     
```python
scaling_factor = 1.3
img_scaled = cv2.resize(img, None, fx=scaling_factor, fy=scaling_factor, 
        interpolation=cv2.INTER_LINEAR)
cv2.imshow('Uniform resizing', img_scaled)
```
![运行结果](https://github.com/Anna-Joe/Python-Machine-Learning/blob/master/deal_img_uniform.png)      

4.Skewed方法调整大小    
这种方法可以只调整长或者只调整宽。   
```python
img_scaled = cv2.resize(img, (250, 400), interpolation=cv2.INTER_AREA)
cv2.imshow('Skewed resizing', img_scaled)
```
![运行结果](https://github.com/Anna-Joe/Python-Machine-Learning/blob/master/deal_img_skewed.png)      

5.将图像保存到输出文件
```python
output_file = input_file[:-4] + '_cropped.jpg'
cv2.imwrite(output_file, img_cropped)
```
![运行结果](https://github.com/Anna-Joe/Python-Machine-Learning/blob/master/deal_img_new.png) 
6.waitKey函数保持显示图像，直到按下键盘上的任意一个键。
```python
cv2.waitKey()
```

### 直方图均衡化     
> 直方图均衡化是指修改图像的像素以增强图像的对比度的过程。   
```python
import sys
import cv2
import numpy as np

# Load input image -- 'sunrise.jpg'
input_file = sys.argv[1]
img = cv2.imread(input_file)

# Convert it to grayscale
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow('Input grayscale image', img_gray)

# Equalize the histogram
img_gray_histeq = cv2.equalizeHist(img_gray)
cv2.imshow('Histogram equalized - grayscale', img_gray_histeq)

# Histogram equalization of color images
img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])

img_histeq = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

cv2.imshow('Input color image', img)
cv2.imshow('Histogram equalized - color', img_histeq)

cv2.waitKey()
```
怀疑这个图像函数又做了什么新的修改，但是查询未果，得到报错如下：
![运行结果](https://github.com/Anna-Joe/Python-Machine-Learning/blob/master/报错信息.png)
