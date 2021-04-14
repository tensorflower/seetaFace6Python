#_*_coding:utf-8 _*_
from  .face_struct import *
import cv2
import numpy as np
import base64
import sys,os
import platform

API_DIR = os.path.dirname(os.path.abspath(__file__))
platform_name = platform.platform().lower()

LIB_PATH=None
dll=None
if "windows" in platform_name:
    LIB_PATH = os.path.join(API_DIR,"lib","win")
    os.environ["PATH"] += os.pathsep + LIB_PATH
    dll = CDLL(os.path.join(LIB_PATH, "libFaceAPI.dll"))
elif "ubuntu" in platform_name or "debian" in platform_name:
    LIB_PATH = os.path.join(API_DIR,"lib","ubuntu")
    dll = CDLL(os.path.join(LIB_PATH, "libFaceAPI.so"))
elif "centos" in platform_name:
    LIB_PATH = os.path.join(API_DIR,"lib","centos")
    dll = CDLL(os.path.join(LIB_PATH, "libFaceAPI.so"))
else:
    print("Unsupported platform:{}".format(platform_name))
    exit()

MODEL_DIR = os.path.join(API_DIR,"model")

# dllc = cdll.msvcrt
# malloc = dllc.malloc
# malloc.restype = c_void_p
# malloc.argtypes = (c_size_t,)
# free = dllc.free
# free.restype = None
# free.argtypes = (c_void_p,)

#人脸检测
FACE_DETECT = 0x00000001
#人脸跟踪
FACE_TRACK = 0x00000002
#人脸识别（特征提取）
FACERECOGNITION = 0x00000004
#rgb活体检测
LIVENESS = 0x00000008
#人脸5点关键点检测
LANDMARKER5 = 0x00000010
#人脸68点关键点检测
LANDMARKER68 = 0x00000020
#带遮挡识别的人脸5点关键点检测
LANDMARKER_MASK = 0x00000040
#人脸姿态角度方向评估
FACE_POSE_EX = 0x00000080
#性别识别
FACE_GENDER = 0x00000100
#年龄识别
FACE_AGE = 0x00000200


#人脸检测器属性设置枚举
class DetectProperty():
    #默认值大小为20
    PROPERTY_MIN_FACE_SIZE = 0
    #默认为0.9
    PROPERTY_THRESHOLD = 1
    PROPERTY_MAX_IMAGE_WIDTH = 2
    PROPERTY_MAX_IMAGE_HEIGHT = 3
    #默认为1
    PROPERTY_NUMBER_THREADS = 4

def get_numpy_by_seetaImageData(image_data):
    width = image_data.width
    height = image_data.height
    channels = image_data.channels
    row_array = np.array(np.fromiter(image_data.data, dtype=np.uint8, count=width*height*channels))
    image_np = row_array.reshape([height,width,channels])
    return image_np

def get_seetaImageData_by_numpy(image_np):
    seetaImageData = SeetaImageData()
    height, width, channels = image_np.shape
    seetaImageData.height = int(height)
    seetaImageData.width = int(width)
    seetaImageData.channels = int(channels)
    seetaImageData.data = image_np.ctypes.data_as(POINTER(c_ubyte))
    return seetaImageData


class SeetaFace(object):
    def __init__(self,init_mask):
        self._init_mask = init_mask
        self._dll_func_def()
        self._init_engine()

    def check_init(self,init_flag):
        if not (init_flag & self._init_mask):
            raise Exception("该功能对应的引擎未初始化!")

    def _dll_func_def(self):
        self._Track = dll.Track
        self._Track.restype = SeetaTrackingFaceInfoArray
        self._Track.argtypes = (SeetaImageData,)

        self._SetSingleCalculationThreads = dll.SetSingleCalculationThreads
        self._SetSingleCalculationThreads.restype = SeetaTrackingFaceInfoArray
        self._SetSingleCalculationThreads.argtypes = (SeetaImageData,)

        self._SetInterval = dll.SetInterval
        self._SetInterval.argtypes =  (c_int32,)

        self._SetMinFaceSize = dll.SetMinFaceSize
        self._SetMinFaceSize.argtypes = (c_int32,)

        self._SetThreshold = dll.SetThreshold
        self._SetThreshold.argtypes = (c_float,)

        self._Reset = dll.Reset

        self._Predict = dll.Predict
        self._Predict.restype = c_int32
        self._Predict.argtypes = (SeetaImageData, SeetaRect, POINTER(SeetaPointF))

        self._PredictVideo = dll.PredictVideo
        self._PredictVideo.restype = c_int32
        self._PredictVideo.argtypes = (SeetaImageData, SeetaRect, POINTER(SeetaPointF))

        self._ResetVideo = dll.ResetVideo

        self._GetPreFrameScore = dll.GetPreFrameScore
        self._GetPreFrameScore.argtypes = (POINTER(c_float), POINTER(c_float))

        self._mark5 = dll.mark5
        self._mark5.restype = c_int32
        self._mark5.argtypes = (SeetaImageData, SeetaRect, POINTER(SeetaPointF))

        self._mark68 = dll.mark68
        self._mark68.restype = c_int32
        self._mark68.argtypes = (SeetaImageData, SeetaRect, POINTER(SeetaPointF))

        self._markMask = dll.markMask
        self._markMask.restype = c_int32
        self._markMask.argtypes = (SeetaImageData, SeetaRect, POINTER(SeetaPointF),POINTER(c_int32))

        self._CropFace = dll.CropFace
        self._CropFace.restype = SeetaImageData
        self._CropFace.argtypes = (SeetaImageData, POINTER(SeetaPointF))

        self._ExtractCroppedFace = dll.ExtractCroppedFace
        self._ExtractCroppedFace.restype = c_int32
        self._ExtractCroppedFace.argtypes = (SeetaImageData, POINTER(c_float))

        self._Extract = dll.Extract
        self._Extract.restype = c_int32
        self._Extract.argtypes = (SeetaImageData, POINTER(SeetaPointF), POINTER(c_float))

        self._CalculateSimilarity = dll.CalculateSimilarity
        self._CalculateSimilarity.restype = c_float
        self._CalculateSimilarity.argtypes = (POINTER(c_float), POINTER(c_float))

        self._Detect = dll.Detect
        self._Detect.restype = SeetaFaceInfoArray
        self._Detect.argtypes = (SeetaImageData,)

        self._SetProperty = dll.SetProperty
        self._SetProperty.argtypes = (c_int32,c_double)

        self._check = dll.check
        self._check.restype = c_int32
        self._check.argtypes = (SeetaImageData, SeetaRect, POINTER(SeetaPointF))

        self._set = dll.set
        self._set.argtypes = (c_int32, c_int32, c_int32,c_int32,c_int32,c_int32)

        self._PredictGenderWithCrop =dll.PredictGenderWithCrop
        self._PredictGenderWithCrop.restype = c_int32
        self._PredictGenderWithCrop.argtypes = (SeetaImageData, POINTER(SeetaPointF))

        self._PredictGender = dll.PredictGender
        self._PredictGender.restype = c_int32
        self._PredictGender.argtypes = (SeetaImageData,)

        self._PredictAgeWithCrop = dll.PredictAgeWithCrop
        self._PredictAgeWithCrop.restype = c_int32
        self._PredictAgeWithCrop.argtypes = (SeetaImageData,  POINTER(SeetaPointF))

        self._PredictAge = dll.PredictAge
        self._PredictAge.restype = c_int32
        self._PredictAge.argtypes = (SeetaImageData,)

        self.InitEngine = dll.InitEngine
        self.InitEngine.restype = c_int32
        self.InitEngine.argtypes = (c_int32,)

    def _init_engine(self):
        cwd = os.getcwd()
        os.chdir(os.path.dirname(os.path.abspath(__file__)) + "/../")
        self.InitEngine(self._init_mask)
        os.chdir(cwd)

    def Track(self, frame):
        """
        检测图像中的位置信息，
        追踪模式下和检测模式下返回的检测结果相似
        但是追踪模式下会额外多 人脸追踪id（PID），frame_no，step等一些额外参数
        大部分情况下只用关心其中的PID参数(为每一个出现的人脸分配的id，从0开始)
        :param frame:
        :return:
        """
        self.check_init(FACE_TRACK)
        seetaImageData = get_seetaImageData_by_numpy(frame)
        return self._Track(seetaImageData)

    def SetSingleCalculationThreads(self,thread_num):
        """
        设置追踪处理的线程数
        :param thread_num:
        :return:
        """
        self._SetSingleCalculationThreads(thread_num)

    def SetInterval(self,interval):
        """
        设置追踪过程中的检测间隔
        间隔默认值为10。这里跟踪间隔是为了发现新增PID的间隔。
        检测器会通过整张图像检测人脸去发现是否有新增的PID，
        所以这个值太小会导致跟踪速度变慢（不断做全局检测）
        这个值太大会导致画面中新增加的人脸不会立马被跟踪到
        :param interval: 检测间隔帧数
        :return: None
        """
        self._SetInterval(interval)

    def SetMinFaceSize(self, size: int):
        """
        设置人脸追踪最小检测人脸大小，默认已设置20
        :param size:
        :return:
        """
        self._SetMinFaceSize(size)

    def SetThreshold(self,thresh:float):
        self._SetThreshold(thresh)

    def Reset(self):
        """
        人脸跟踪模块 重置，更换视频源 时需要调用
        :return:
        """
        self._Reset()

    def Detect(self, frame) :
        """
        人脸检测
        :param frame: 原始图像
        :return: 人脸检测信息数组
        """

        self.check_init(FACE_DETECT)
        seetaImageData = get_seetaImageData_by_numpy(frame)
        return self._Detect(seetaImageData)

    def SetProperty(self,property,value):
        self._SetProperty(property,value)

    def Predict(self, frame, face,
                   points) :
        """
        单帧rgb活体检测
        :param frame: 原始图像
        :param face: 人脸区域
        :param points:  人脸关键点位置
        :return:  活体检测结果
        0:真实人脸
        1:攻击人脸（假人脸）
        2:无法判断（人脸成像质量不好）
        """
        self.check_init(LIVENESS)
        seetaImageData = get_seetaImageData_by_numpy(frame)
        return self._Predict(seetaImageData, face, points)

    def PredictVideo(self, frame, face,points) :
        """
        视频rgb活体检测（多帧判断出的结果）
        相比较于Predict 函数，多了一个正在检测状态的返回值
        :param frame: 原始图像
        :param face: 人脸区域
        :param points:  人脸关键点位置
        :return:  活体检测结果
        0:真实人脸
        1:攻击人脸（假人脸）
        2:无法判断（人脸成像质量不好）
        3: 正在检测
        """
        self.check_init(LIVENESS)
        seetaImageData = get_seetaImageData_by_numpy(frame)
        return self._PredictVideo(seetaImageData, face, points)

    def ResetVideo(self):
        self._ResetVideo()

    def GetPreFrameScore(self):
        clarity= c_float()
        reality = c_float()
        self._GetPreFrameScore(clarity, reality)
        return (clarity.value,reality.value)


    def mark5(self, frame, face) :
        """
        给定一张原始图片，和其中人脸区域，返回该人脸区域中5个关键点位置 [左眼，右眼，鼻子，左边嘴角，右边嘴角]
        :param frame: 原始图像
        :param face: 人脸区域位置
        :return:
        """
        self.check_init(LANDMARKER5)
        seetaImageData = get_seetaImageData_by_numpy(frame)
        points = (SeetaPointF * 5)()
        self._mark5(seetaImageData, face, points)
        return points

    def mark68(self, frame, face) :
        """
        给定一张原始图片，和其中人脸区域，返回该人脸区域中的68个关键点位置
        :param frame: 原始图像
        :param face: 人脸区域位置
        :return:
        """
        self.check_init(LANDMARKER68)
        seetaImageData = get_seetaImageData_by_numpy(frame)
        points = (SeetaPointF * 68)()
        self._mark68(seetaImageData, face, points)
        return points

    def markMask(self, frame, face) :
        """
        给定一张原始图片，和其中人脸区域，返回该人脸区域中的5个关键点位置,
        和这 5点是否被遮挡的数组 [左眼，右眼，鼻子，左边嘴角，右边嘴角]
        :param frame: 原始图像
        :param face: 人脸区域位置
        :return:
        points：5关键点数组
       face_mask： 如戴口罩时 基本会返回数组【0，0，1，1，1】，0：没被遮挡  1：被遮挡
        """
        self.check_init(LANDMARKER_MASK)
        seetaImageData = get_seetaImageData_by_numpy(frame)
        points = (SeetaPointF * 5)()
        face_mask = (c_int32 * 5)()
        self._markMask(seetaImageData, face, points,face_mask)
        return points,face_mask


    def CropFace(self, frame, points):
        """
        根据关键点位置，裁剪出矫正后的人脸区域图片
        :param frame:
        :param points:
        :return: [256*256*c]
        """
        seetaImageData = get_seetaImageData_by_numpy(frame)
        out_seetaImageData = self._CropFace(seetaImageData, points)
        return get_numpy_by_seetaImageData(out_seetaImageData)

    def ExtractCroppedFace(self, frame):
        """
        #提取人脸图像特征值（整个一张图片为一张人脸时）
        :param frame:
        :return:
        """
        self.check_init(FACERECOGNITION)
        if frame.shape[0]!=256 or frame.shape[1]!=256:
            seetaImageData = get_seetaImageData_by_numpy(cv2.resize(frame,(256,256)))
        else:
            seetaImageData = get_seetaImageData_by_numpy(frame)
        feature = (c_float * 1024)()
        self._ExtractCroppedFace(seetaImageData, feature)
        return  feature

    def Extract(self, frame, points):
        """
        在一张图片中提取指定人脸关键点区域的人脸的特征值
        :param frame:
        :param points:
        :return:
        """
        self.check_init(FACERECOGNITION)
        seetaImageData = get_seetaImageData_by_numpy(frame)
        feature = (c_float * 1024)()
        self._Extract(seetaImageData, points, feature)
        return feature

    def CalculateSimilarity(self, features1, features2):
        self.check_init(FACERECOGNITION)
        return self._CalculateSimilarity(features1, features2)

    def compare_feature_np(self, feature1, feature2) :
        """
        使用numpy 计算，比较人脸特征值相似度
       :param feature1: 人脸特征值1
        :param feature2: 人脸特征值2
        :return: 人脸相似度
        """
        dot = np.sum(np.multiply(feature1, feature2))
        norm = np.linalg.norm(feature1) * np.linalg.norm(feature2)
        dist = dot / norm
        return float(dist)

    def get_feature_by_byte(self, feature_byte: bytes):
        """
        通过特征值二进制 获取 feature 数据
        :param feature:
        :return:
        """
        feature = np.frombuffer(feature_byte).ctypes.data_as(POINTER(c_float))
        return feature

    def get_feature_byte(self, feature):
        """
        获取 feature 的字节流表示数据
        :param feature:
        :return:
        """
        return string_at(feature, 1024 * 4)

    def get_feature_numpy(self,feature):
        """
        获取 feature 的numpy表示数据
        :param feature:
        :return:
        """
        face_encoding = (
            np.frombuffer(string_at(feature, 1024 * 4), dtype=np.float32))
        return face_encoding

    def get_feature_base64(self, feature):
        """
        获取 feature 的base64表示形式
        :param feature:
        :return: base64 字符串
        """
        return base64.b64encode(self.get_feature_byte(feature)).decode(encoding="UTF-8")

    def check(self,frame,face, points):
        """
        #检测人脸姿态角度是否合适
        :param frame:
        :param face:
        :param points:
        :return:  0：低  1：中等 2：高
        """
        self.check_init(FACE_POSE_EX)
        seetaImageData = get_seetaImageData_by_numpy(frame)
        return self._check(seetaImageData,face,points)

    def set(self,yaw_low_threshold,
            yaw_high_threshold,
            pitch_low_threshold,
            pitch_high_threshold,
            roll_low_threshold,
            roll_high_threshold):
        """
        设置人脸姿态角度评估模型判定范围
        :param yaw_low_threshold:
        :param yaw_high_threshold:
        :param pitch_low_threshold:
        :param pitch_high_threshold:
        :param roll_low_threshold:
        :param roll_high_threshold:
        """
        self._set(yaw_low_threshold,yaw_high_threshold,pitch_low_threshold,pitch_high_threshold,roll_low_threshold,roll_high_threshold)

    def PredictGenderWithCrop(self,frame,points) :
        """
        检测一张原图中一个人脸的性别，需要人脸关键点位置
        :param frame: 原图
        :param points: 人脸关键点
        :return: 0：男   1：女
        """
        self.check_init(FACE_GENDER)
        seetaImageData = get_seetaImageData_by_numpy(frame)
        return self._PredictGenderWithCrop(seetaImageData,points)

    def PredictGender(self,frame) :
        """
        检测一张只有人脸的图片,识别出性别
        :param frame: 原图
        :param face: 人脸检测框
        :param points: 人脸关键点
        :return: 0：男   1：女
        """
        self.check_init(FACE_GENDER)
        if frame.shape[0] != 128 or frame.shape[1] != 128:
            seetaImageData = get_seetaImageData_by_numpy(cv2.resize(frame, (128, 128)))
        else:
            seetaImageData = get_seetaImageData_by_numpy(frame)
        return self._PredictGender(seetaImageData)

    def PredictAgeWithCrop(self,frame, points) :
        """
        检测一张原图中一个人脸的年龄，需要人脸关键点位置
        :param frame: 原图
        :param points: 人脸关键点
        :return: 年龄大小
        """
        self.check_init(FACE_AGE)
        seetaImageData = get_seetaImageData_by_numpy(frame)
        return self._PredictAgeWithCrop(seetaImageData, points)


    def PredictAge(self,frame) :
        """
        检测一张只有人脸的图片,识别出年龄
        :param frame: 原图
        :param face: 人脸检测框
        :param points: 人脸关键点
        :return: 年龄大小
        """
        self.check_init(FACE_AGE)
        if frame.shape[0]!=256 or frame.shape[1]!=256:
            seetaImageData = get_seetaImageData_by_numpy(cv2.resize(frame,(256,256)))
        else:
            seetaImageData = get_seetaImageData_by_numpy(frame)
        return self._PredictAge(seetaImageData)


