#_*_coding:utf-8 _*_
"""
人脸角度姿态评估demo
"""
from seetaface.api import *

"""
使用到的函数:
    check：检测人脸角度是否合适
    set: 设置人脸姿态角度的范围判断阈值，无特殊要求一般不用设置,默认已设置如下：
        yaw_low_threshold: 25
        yaw_high_threshold: 10
        pitch_low_threshold: 20
        pitch_high_threshold: 10
        roll_low_threshold: 33.33
        roll_high_threshold: 16.67
    
要加载的功能 :
    人脸姿态角度功能：FACE_POSE_EX
依赖功能:
    FACE_DETECT：人脸检测
    LANDMARKER5：5点关键点检测
"""
init_mask = FACE_POSE_EX|FACE_DETECT|LANDMARKER5

seetaFace = SeetaFace(init_mask)

image = cv2.imread("asserts/1.jpg")

detect_result = seetaFace.Detect(image)
for i in range(detect_result.size):
    face = detect_result.data[i].pos
    points = seetaFace.mark5(image,face)
    #返回角度是否合适 0：低  1：中等 2：高
    status = seetaFace.check(image,face,points)
    print(status)


