from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import cv2
from skimage.util import random_noise
import random
from collections import Counter
from sklearn.model_selection import LeaveOneGroupOut
from scipy.signal import find_peaks
from Utils.mean_average_precision.mean_average_precision import MeanAveragePrecision2d

random.seed(1)

# 测试是否启用 GPU
print("测试tf是否启用gpu")
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        # 打印 GPU 详细信息
        for gpu in gpus:
            details = tf.config.experimental.get_device_details(gpu)
            print("GPU 设备名称:", details['device_name'])
            print("GPU 计算能力:", details['compute_capability'])
    except RuntimeError as e:
        print(e)

# 测试 TensorFlow GPU 加速
import time
start_time = time.time()
a = tf.random.normal([10000, 10000])
b = tf.random.normal([10000, 10000])
c = tf.matmul(a, b)
end_time = time.time()
print("TensorFlow 矩阵乘法耗时: {:.2f} 秒".format(end_time - start_time))

# 测试 OpenCV 是否支持 CUDA
print("测试cv2是否支持gpu")
print(cv2.getBuildInformation())

try:
    print("测试OpenCV CUDA加速")
    img = cv2.imread('test.jpg', cv2.IMREAD_COLOR)
    img_gpu = cv2.cuda_GpuMat()
    img_gpu.upload(img)
    img_blur = cv2.cuda.bilateralFilter(img_gpu, 15, 75, 75)
    result = img_blur.download()
    print("OpenCV CUDA加速可用")
except Exception as e:
    print("OpenCV CUDA加速不可用:", e)

# 测试 skimage 随机噪声
image = np.zeros((100, 100), dtype=np.uint8)
noisy_image = random_noise(image, mode='s&p', amount=0.1)
plt.imshow(noisy_image, cmap='gray')
plt.title("随机噪声测试")
plt.show()

# 测试 MeanAveragePrecision2d
print("测试MeanAveragePrecision2d")
gt = np.array([[50, 50, 100, 100, 1]])
pred = np.array([[55, 55, 100, 100, 0.9]])
mean_ap = MeanAveragePrecision2d()
mean_ap.add(pred, gt)
ap = mean_ap.value(iou_thresholds=0.5)
print("mAP 测试结果:", ap)

