#_*_coding:utf-8 _*_
"""
人脸特征提取比对 demo
"""
from seetaface.api import *
"""
使用到的函数:
    Extract：提取一张原始图像里的一个指定区域的人脸的特征值
    ExtractCroppedFace: 提取一张已经裁剪好，只包含人脸图的人脸的特征值
    CalculateSimilarity: 计算两个特征值之间的相似度
要加载的功能 :
    人脸识别功能：FACERECOGNITION
该功能依赖 ：
    FACE_DETECT :人脸检测
    LANDMARKER5 :5点关键点识别
"""
init_mask = FACE_DETECT|FACERECOGNITION|LANDMARKER5

seetaFace = SeetaFace(init_mask)

#提取第一张图中的人脸特征
image = cv2.imread("asserts/1.jpg")
detect_result1 = seetaFace.Detect(image)
face1 = detect_result1.data[0].pos
points1 = seetaFace.mark5(image,face1)
feature1 = seetaFace.Extract(image,points1)

#提取第二张图中的人脸特征
image2 = cv2.imread("asserts/2.jpg")
detect_result2 = seetaFace.Detect(image2)
face2 = detect_result2.data[0].pos
points2 = seetaFace.mark5(image2,face2)
feature2 = seetaFace.Extract(image2,points2)
#计算两个特征值的形似度
similar1 = seetaFace.CalculateSimilarity(feature1,feature2)
print(similar1)

"""
直接对两张裁剪好的人脸图做铁证提取并比对
"""
image3 = cv2.imread("asserts/crop1.jpg")
feature3 = seetaFace.ExtractCroppedFace(image3)

image4 = cv2.imread("asserts/crop2.jpg")
feature4 = seetaFace.ExtractCroppedFace(image4)

similar2 = seetaFace.CalculateSimilarity(feature3,feature4)
print(similar2)






