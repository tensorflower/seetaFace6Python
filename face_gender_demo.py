#_*_coding:utf-8 _*_
"""
性别识别demo
"""
from seetaface.api import *

""" 
使用到的函数:
    两个函数目的相同，但是结果会有一定差异
    PredictGender: 该函数检测一张只有人脸的图片,识别出性别 
    PredictGenderWithCrop：检测一张原图中一个人脸的性别，需要人脸关键点位置,因此需要使用到人5点关键点是检测功能，而关键点检测功能又依赖检测功能 ,
 
要加载的功能 :
    性别识别：FACE_GENDER
依赖功能:
    FACE_DETECT：人脸检测
    LANDMARKER5：5点关键点检测
    
"""

""" 检测一张大图中的每个人脸的性别 """
init_mask = FACE_DETECT|LANDMARKER5|FACE_GENDER
seetaFace = SeetaFace(init_mask)
image = cv2.imread("asserts/1.jpg")
detect_result = seetaFace.Detect(image)
for i in range(detect_result.size):
    face = detect_result.data[i].pos
    points_5 = seetaFace.mark5(image, face)
    age = seetaFace.PredictGenderWithCrop(image,points_5)
    print("PredictGenderWithCrop:{}".format(age))

""" 检测一张已经裁剪好的只有人脸图 """
image = cv2.imread("asserts/crop1.jpg")
age = seetaFace.PredictGender(image)
print("PredictGender:{}".format(age))


