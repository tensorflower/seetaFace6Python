#_*_coding:utf-8 _*_
"""
人脸追踪demo （用于视频模式下）
以打开本地摄像头为例
"""
from seetaface.api import *

"""
使用到的函数:
    Track：将测一帧中的人脸位置信息
    Reset: 更换视频源时，需要调用
    SetSingleCalculationThreads: 设置追踪处理的线程数
    SetInterval: #设置检测人脸间隔帧数
    SetMinFaceSize: 设置最小人脸检测大小，默认20
    SetThreshold: 设置人脸检测得分阈值
要加载的功能 :
    FACE_TRACK:人脸跟踪功能
"""
init_mask = FACE_TRACK

seetaFace = SeetaFace(init_mask)

camera = cv2.VideoCapture(0)

if camera.isOpened():
    while 1:
        flag,frame = camera.read()
        if flag:
            detect_result = seetaFace.Track(frame)
            for i in range(detect_result.size):
                face = detect_result.data[i].pos
                PID = detect_result.data[i].PID  #同一张人脸没有离开视频则其PID 一般不会改变
                cv2.rectangle(frame, (face.x, face.y), (face.x + face.width, face.y + face.height),(255, 0, 0), 2)
                cv2.putText(frame,"pid:{}".format(PID),(face.x,face.y),1,1,(0,0,255))
            cv2.imshow("track",frame)
            cv2.waitKey(30)

else:
    print("摄像头打开失败！")

