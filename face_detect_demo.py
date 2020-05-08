#_*_coding:utf-8 _*_
"""
人脸检测演示 demo
"""
from seetaface.api import *

"""
使用到的函数:
    Detect：检测一帧中的人脸位置信息
    SetProperty: 设置人脸检测引擎的一些属性，如最小人脸 ，人脸检测阈值设置等
要加载的功能 :
    FACE_DETECT:人脸检测功能
"""
init_mask = FACE_DETECT

seetaFace = SeetaFace(init_mask)

image = cv2.imread("asserts/1.jpg")

"""设置人脸检测器的属性"""
#设置最小检测人脸大小
#最小人脸是人脸检测器常用的一个概念，默认值为20，单位像素。
# 它表示了在一个输入图片上可以检测到的最小人脸尺度，注意这个尺度并非严格的像素值，
# 例如设置最小人脸80，检测到了宽度为75的人脸是正常的，这个值是给出检测能力的下限。
#最小人脸和检测器性能息息相关。主要方面是速度，使用建议上，建议在应用范围内，
# 这个值设定的越大越好。 如果最小人脸参数设置为80的话，从检测能力上，可以将原图缩小的原来的1/4，
# 这样从计算复杂度上，能够比最小人脸设置为20时，提速到16倍
seetaFace.SetProperty(DetectProperty.PROPERTY_MIN_FACE_SIZE,80)
#设置最小检测人脸阈值0.9，大于0.9得分的人脸才返回
seetaFace.SetProperty(DetectProperty.PROPERTY_THRESHOLD,0.9)

#检测图像中的人脸
detect_result = seetaFace.Detect(image)

#打印检测结果
print(detect_result)

for i in range(detect_result.size):
    face = detect_result.data[i].pos
    cv2.rectangle(image, (face.x, face.y), (face.x + face.width, face.y + face.height), (255, 0, 0),2)

cv2.imshow("s",image)
cv2.waitKey(0)

