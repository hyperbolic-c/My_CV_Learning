载入yolov5的代码框架后，配置好环境和依赖包
# 数据集的准备
## 文件架构
在data文件目录下创建fire_data文件夹，在fire_data文件夹下创建images和labels文件夹，文件夹结构如下：
![](tutorial_create_model_md_files%5Cimage.png?v=1&type=image)
按照文件夹结构放置处理好的数据集
## 数据集处理
如何将下载的数据集中的xml文件转化为yolo所需的对应的txt文件？
```
import xml.etree.ElementTree as ET

import os

import cv2

# xml文件的路径
annotations = "./Annotations/"
# 图片文件的路径
imgs = "./JPEGImages"
# 类别
classes = ["smoke","fire"]

  

# 获取xml的名称列表

annotations_xml = os.listdir(annotations)

  

# 归一化

def  convert(size, box):

dw = 1./size[0]

dh = 1./size[1]

x = (box[0] + box[1])/2.0

y = (box[2] + box[3])/2.0

w = box[1] - box[0]

h = box[3] - box[2]

x = x*dw

w = w*dw

y = y*dh

h = h*dh

return (x,y,w,h)

def  convert_annotation(annotation_id, output_file):

in_file = open('Annotations/%s.xml'%(annotation_id))

print(annotation_id)

tree=ET.parse(in_file)

root = tree.getroot()

size = root.find("size")

img = cv2.imread(os.path.join(imgs,annotation_id+".jpg"))

h = img.shape[0]

w = img.shape[1]

for obj in root.iter('object'):

difficult = obj.find('difficult').text

cls = obj.find('name').text # 类别

if  cls  not  in classes or  int(difficult)==1:

continue

cls_id = classes.index(cls) # 类别的索引

xmlbox = obj.find('bndbox') # 边框区域

b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text)) # 得到的是两点坐标

bb = convert((w,h), b) # 将两点坐标转换成 x y w h

output_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n') # 写入到文件夹

  

if  __name__ == "__main__":

# 对xml名称列表遍历

for annotation_name in annotations_xml:

annotation_id = annotation_name[:-4]

output_file = open("labels/%s.txt"%(annotation_id),"w")

convert_annotation(annotation_id, output_file)

output_file.write('\n')

output_file.close()
```

# 配置分类文件
在data文件夹下参考coco128.yaml文件新建fire.yaml文件，文件夹内容参考如下：
```
# train and val datasets (image directory or *.txt file with image paths)
train: ./data/fire_data/images/train/
val: ./data/fire_data/images/val/

# number of classes
nc: 2

# class names
names: ['smoke', 'fire']

```
nc为你设定的分类类别数量，注意nc的引号后面还需要一个空格
names：[]里面的内容排序要与labelImg标记的内容顺序一致，如图示意：
![0为smoke，1为fire](tutorial_create_model_md_files%5Ctrain_batch0.jpg?v=1&type=image)
0为smoke，1为fire

同时，将models/yolov5s.yaml（选择的模型）文件中的类别数nc改为对应的数字
```
# parameters  
# nc: 80  # number of classes  
nc: 2  
depth_multiple: 0.33  # model depth multiple  
width_multiple: 0.50  # layer channel multiple  
  
# anchors  
anchors:  
  - [10,13, 16,30, 33,23]  # P3/8  
  - [30,61, 62,45, 59,119]  # P4/16  
  - [116,90, 156,198, 373,326]  # P5/32
```
# 配置训练模型文件
## train.py文件
```
if __name__ == '__main__':  
    parser = argparse.ArgumentParser()  
    parser.add_argument('--weights', type=str, default='yolov5s.pt', help='initial weights path')  
    # --cfg中的default为选择的模型的路径
    parser.add_argument('--cfg', type=str, default='models/yolov5s.yaml', help='model.yaml path')
    # --data中的default为新建的fire.yaml文件的路径  
    parser.add_argument('--data', type=str, default='data/fire.yaml', help='data.yaml path')  
    parser.add_argument('--hyp', type=str, default='data/hyp.scratch.yaml', help='hyperparameters path')
    # --epochs中的default为训练的次数 默认300 可自定义  
    parser.add_argument('--epochs', type=int, default=50)
    # --batch-size中的default为批次大小 调低该值一定程度上可防止gpu内存溢出  
    parser.add_argument('--batch-size', type=int, default=4, help='total batch size for all GPUs')
```
## test.py文件
```
# --weights中的default修改为weights/yolov5s.pt
parser.add_argument('--weights', nargs='+', type=str, default='weights/yolov5s.pt', help='model.pt path(s)') 
# --data中的修改与train.py中一致
parser.add_argument('--data', type=str, default='data/fire.yaml', help='*.data path')
# --batch-size中的default自行调整  
parser.add_argument('--batch-size', type=int, default=2, help='size of each image batch')
```
# 训练与预测
## 训练模型
运行train.py文件开始训练，训练完毕后的结果在runs/train文件夹下。模型在runs/train/exp(数字)/weights文件夹中，分别是best.py和last.py文件。exp文件夹下还有训练结果的各种参数图。
## 预测
修改detect.py文件中的配置后再运行该文件进行预测
```
if __name__ == '__main__':  
    parser = argparse.ArgumentParser()
    # --weights中的default为选择训练好的模型的路径  
    parser.add_argument('--weights', nargs='+', type=str,  
  default='runs/train/exp9/weights/last.pt', help='model.pt path(s)')  
    # --source为输入的文件或文件夹 图片视频均可 设置为0则是实时监测 默认电脑摄像头  
  parser.add_argument('--source', type=str, default='data/videos', help='source')  # file/folder, 0 for webcam
```
输入的图片或者视频放置在data/images文件夹或者data/videos文件夹中，输出的结果保存在runs/detect/exp(数字)文件夹下。

