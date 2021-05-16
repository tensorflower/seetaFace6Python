# **seetaFace6 python api(10分钟搞定商业级别人脸识别应用)**

## 1. 简介

项目基于`SeetaFace6` 封装的python接口,使用简便,性能与原始c++模块基本一致,
模块上`SeetaFace6` 相较于`SeetaFace2` 上训练样本更多，准确率更高，提供的功能和识别模型也更多
接口封装上，放弃了使用 `pybind11` 封装python 接口，接口函数完全纯 c 接口，使用 ctypes 调用，解除不同版本python使用上的限制
`10分钟搞定`是夸张说法，但本次项目基本涵盖了普通商用人脸识别所需的大部分功能，并且使用简单。
### 支持 windows/linux 下的所有python 版本
#### &ensp;&ensp;python3.5及以下版本的python需要将seetaface/back_py目录下的 `api.py` 和 `face_struct.py` 两个文件放到setaface目录下，替换setaface目录下的`api.py` 和 `face_struct.py` 文件
## 2.下载模型（已下载则忽略）
百度网盘：https://pan.baidu.com/share/init?surl=LlXe2-YsUxQMe-MLzhQ2Aw 提取码：ngne  
将下载的所有 *.csta 模型文件 放入 `seetaFace6Python/seetaface/model` 目录下

## 3. 运行示例
### 3.1 运行依赖
当前需要机器同时支持以下几个指令集 AVX | SSE | FMA ，请先确认机器是否同时支持它们

&ensp;&ensp;&ensp;&ensp;示例依赖 `opencv` ,安装`opencv`(若已有cv2模块则忽略)

```key
    pip install -i https://pypi.tuna.tsinghua.edu.cn/simple opencv-python
``` 

### 3.2 演示demo
&ensp;&ensp;`linux`下则需要添加库路径
```shell script
    临时:
        ubuntu:
            export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${seetaFace6Python目录路径}/seetaface/lib/ubuntu
        centos:
            export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${seetaFace6Python目录路径}/seetaface/lib/centos
``` 
```shell script
    永久:
        ubuntu:
            sudo echo  ${seetaFace6Python目录路径}/seetaface/lib/ubuntu  > /etc/ld.so.conf.d/seetaface6.conf
        centos:
            sudo echo  ${seetaFace6Python目录路径}/seetaface/lib/centos  > /etc/ld.so.conf.d/seetaface6.conf
        
        sudo ldconfig
    
``` 



```key
    进入 seetaFace6Python 目录下，有各种基本功能的使用demo
```


