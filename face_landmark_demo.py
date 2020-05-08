#_*_coding:utf-8 _*_
"""
人脸关键点检测模型
当前主要有：
    1.  5点人脸检测
    2.  68点人脸检测
    3.  带有关键点是否遮挡判断的的5点人脸检测
"""

from seetaface.api import *

"""
总共有三种关键点检测模型
1.普通5点检测模型
要加载的功能 :
    LANDMARKER5：普通5点检测模型
依赖功能:
    FACE_DETECT：人脸检测
使用到的函数:
    mark5：检测人脸5个关键点位置

2.普通68点检测模型
要加载的功能 :
    LANDMARKER68：普通68点检测模型
依赖功能:
    FACE_DETECT：人脸检测
使用到的函数:
    mark68：检测人脸68个关键点位置

3.带关键点检测被遮挡判断的5点检测模型
要加载的功能 :
    LANDMARKER_MASK：带关键点检测遮挡判断的5点检测模型
依赖功能:
    FACE_DETECT：人脸检测
使用到的函数:
    markMask：检测人脸5个关键点位置，和这5点是否被遮挡的判断
"""
init_mask = FACE_DETECT|LANDMARKER5|LANDMARKER68|LANDMARKER_MASK

seetaFace = SeetaFace(init_mask)

#########################################################
################      5 点检测模型       ################
#########################################################
points_5_image = cv2.imread("asserts/1.jpg")
detect_result = seetaFace.Detect(points_5_image)
for i in range(detect_result.size):
    face = detect_result.data[i].pos
    #5点检测模型检测
    points_5 = seetaFace.mark5(points_5_image,face)
    #画人脸框
    cv2.rectangle(points_5_image, (face.x, face.y), (face.x + face.width, face.y + face.height), (255, 0, 0),2)
    for i in range(5):
        print(points_5[i])
        # 画关键点
        cv2.circle(points_5_image, (int(points_5[i].x), int(points_5[i].y)), 5, (255, 0, 0), -1)

#########################################################
################      68点检测模型       ################
#########################################################
points_68_image = cv2.imread("asserts/1.jpg")
detect_result = seetaFace.Detect(points_68_image)
for i in range(detect_result.size):
    face = detect_result.data[i].pos
    # 68点检测模型检测
    points_68 = seetaFace.mark68(points_68_image, face)

    #画人脸框
    cv2.rectangle(points_68_image, (face.x, face.y), (face.x + face.width, face.y + face.height), (255, 0, 0),2)

    for i in range(68):
        print(points_68[i])
        # 画关键点
        cv2.circle(points_68_image, (int(points_68[i].x), int(points_68[i].y)), 3, (255, 0, 0), -1)

#########################################################
#######带关键点是否遮挡的 5点检测模型检测      ###########
#########################################################
points_mask_image = cv2.imread("asserts/mask.png")
detect_result = seetaFace.Detect(points_mask_image)
for i in range(detect_result.size):
    face = detect_result.data[i].pos
    # 带关键点是否遮挡的 5点检测模型检测，返回两个值 （points,face_mask），
    # point 基本等价于5点检测模型中的关键点位置数据，face_mask 为一个 长度为5的数组，里面记录每个关键点是否被遮挡
    #如 戴口罩时 face_mask 可能会返回 [左眼，右眼，鼻子，左边嘴角，右边嘴角] ->[0,0,1,1,1] ,0：没被遮挡  1：被遮挡
    points_mask = seetaFace.markMask(points_mask_image, face)
    #画人脸框
    cv2.rectangle(points_mask_image, (face.x, face.y), (face.x + face.width, face.y + face.height), (255, 0, 0),2)
    points,mask = points_mask
    for i in range(5):
        print(points[i])
        if mask[i]==1:#被遮挡
            #画关键点
            cv2.circle(points_mask_image, (int(points[i].x), int(points[i].y)), 5, (0, 0, 255), -1)
        else:
            # 画关键点
            cv2.circle(points_mask_image, (int(points[i].x), int(points[i].y)), 5, (255, 0, 0), -1)

cv2.imshow("points_5_image",points_5_image)
cv2.imshow("points_68_image",points_68_image)
cv2.imshow("points_mask_image",points_mask_image)
cv2.waitKey(0)

