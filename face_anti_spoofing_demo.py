#_*_coding:utf-8 _*_
"""
人脸攻击判断（rgb活体检测）
"""
from seetaface.api import *
"""
使用到的函数:
    Predict:对单帧做判断
    PredictVideo：多视频做判断（连续多帧统计结果）
    ResetVideo:重置，用于切换检测视频时需要调用
要加载的功能 :
    活体检测功能：LIVENESS
该功能依赖 ：
    人脸检测：FACE_DETECT 
    5点关键点识别：LANDMARKER5
"""
init_mask = LIVENESS|FACE_DETECT|LANDMARKER5

seetaFace = SeetaFace(init_mask)

"""
单帧情况 Predict
"""
image = cv2.imread("asserts/1.jpg")
detect_result = seetaFace.Detect(image)
for i in range(detect_result.size):
    _face = detect_result.data[i].pos
    points = seetaFace.mark5(image,_face)
    # 0: 真实人脸
    # 1: 攻击人脸（假人脸）
    # 2: 无法判断（人脸成像质量不好）
    livnees = seetaFace.Predict(image,_face,points)
    print("livnees:",livnees)
"""
视频情况 PredictVideo
PredictVideo 基于多帧图像（10帧）的合计，如果真人被识别出来，
马上替换成假人出现，有可能刚出现假人的前几帧也会被认为是真人，再等待几帧后会出现比较合适的结果
或者可以通过调用 ResetVideo 函数，达到重新开始识别新的一个人脸出现的情况

摄像头为例
"""
#视频模式下建议将 FACE_DETECT 改为 FACE_TRACK，降低每帧做人脸检测带来的消耗
init_mask = LIVENESS|FACE_TRACK|LANDMARKER5

seetaFace = SeetaFace(init_mask)

video_cap = cv2.VideoCapture(0)
if video_cap.isOpened():
    while 1:
        flag, frame = video_cap.read()
        if flag:
            detect_result = seetaFace.Track(frame)
            print(detect_result)
            for i in range(detect_result.size):
                face = detect_result.data[i].pos
                points = seetaFace.mark5(frame, face)
                # 0: 真实人脸
                # 1: 攻击人脸（假人脸）
                # 2: 无法判断（人脸成像质量不好）
                # 3 : 正在判断（PredictVodeo）
                livnees = seetaFace.PredictVideo(frame, face, points)
                clarity, reality = seetaFace.GetPreFrameScore()
                if (livnees == 3):
                    cv2.rectangle(frame, (face.x, face.y),
                                  (face.x + face.width, face.y + face.height),
                                  (255, 255, 255), 2)
                    cv2.putText(frame, "wait", (face.x, face.y), 1, 1, (255, 255, 255))
                elif (livnees == 2):
                    cv2.rectangle(frame, (face.x, face.y),
                                  (face.x + face.width, face.y + face.height),
                                  (255, 255, 0), 2)
                    cv2.putText(frame, "unknow", (face.x, face.y), 1, 1, (255, 255, 0))
                elif (livnees == 1):
                    cv2.rectangle(frame, (face.x, face.y),
                                  (face.x + face.width, face.y + face.height),
                                  (0, 0, 255), 2)
                    cv2.putText(frame, "fake", (face.x, face.y), 1, 1, (0, 0, 255))
                elif (livnees == 0):
                    cv2.rectangle(frame, (face.x, face.y),
                                  (face.x + face.width, face.y + face.height),
                                  (0, 255, 0), 2)
                    cv2.putText(frame, "real", (face.x, face.y), 1, 1, (0, 255, 0))
        cv2.imshow("video",frame)
        cv2.waitKey(30)
else:
    print('打开本地摄像头失败')
