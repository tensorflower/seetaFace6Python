#_*_coding:utf-8 _*_
from ctypes import *
from typing import List

class SeetaPointF(Structure):
    _fields_=[('x',c_double),('y',c_double)]
    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return "SeetaPointF(x={},y={})".format(self.x,self.y)

class SeetaImageData(Structure):
    _fields_=[('width',c_int32),('height',c_int32),('channels',c_int32),("data",POINTER(c_ubyte))]

    def __str__(self):
        return "SeetaImageData(width={},height={},channels={},data:{})".format(self.width,self.height,self.channels,List[c_ubyte])

class SeetaRect(Structure):
    _fields_=[('x',c_int32),('y',c_int32),('width',c_int32),("height",c_int32)]

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return "SeetaRect(x={},y={},width={},height={})".format(self.x,self.y,self.width,self.height)

class SeetaTrackingFaceInfo(Structure):
    _fields_=[('pos',SeetaRect),('score',c_float),('frame_no',c_int32),("PID",c_int32),("step",c_int32)]

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return "SeetaTrackingFaceInfo(pos={},score={},frame_no={},PID={},step={})".\
            format(self.pos,self.score,self.frame_no,self.PID,self.step)


class SeetaTrackingFaceInfoArray(Structure):

    _fields_ = [('data', POINTER(SeetaTrackingFaceInfo)), ('size', c_int32)]

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        s = "SeetaTrackingFaceInfo(data:{},size:{})".format([self.data[i] for i in range(self.size)],
                                                         self.size)
        return s

class SeetaFaceInfo(Structure):

    _fields_=[('pos',SeetaRect),('score',c_float)]

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return "SeetaFaceInfo(pos={},score={})". \
            format(self.pos, self.score)

class SeetaFaceInfoArray(Structure):
    _fields_ = [('data', POINTER(SeetaFaceInfo)), ('size', c_int32)]
    def __repr__(self):
        return self.__str__()

    def __str__(self):
        s = "SeetaFaceInfoArray(data:{},size:{})".format([self.data[i] for i in range(self.size)],self.size)
        return s
