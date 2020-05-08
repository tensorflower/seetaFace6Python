#_*_coding:utf-8 _*_
"""
年龄识别demo
"""
from seetaface.api import *

""" 
使用到的函数:
    两个函数目的相同，但是结果会有一定差异
    PredictAge:  该函数检测一张只有人脸的图片,识别出年龄
    PredictAgeWithCrop：检测一张原图中一个人脸的年龄，需要人脸关键点位置,需要使用到5点关键点检测功能，而关键点检测功能又依赖检测功能 
要加载的功能 :
    年龄识别功能：FACE_AGE
依赖功能:
    FACE_DETECT：人脸检测
    LANDMARKER5：5点关键点检测

"""

""" 检测一张大图中的每个人脸的年龄 """
init_mask = FACE_DETECT|LANDMARKER5|FACE_AGE
seetaFace = SeetaFace(init_mask) #初始化引擎
print(seetaFace)
image = cv2.imread("asserts/1.jpg")
detect_result = seetaFace.Detect(image)
for i in range(detect_result.size):
    face = detect_result.data[i].pos
    c_image = image[face.y:face.y+face.height,face.x:face.x+face.width]
    points_5 = seetaFace.mark5(image, face)
    age = seetaFace.PredictAgeWithCrop(image,points_5)
    print("PredictAgeWithCrop:{}".format(age))

""" 检测一张已经裁剪好的只有人脸的人脸年龄 """
image = cv2.imread("asserts/crop1.jpg")
age = seetaFace.PredictAge(image)
print("PredictAge:{}".format(age))


