import time
import numpy as np
import matplotlib.pyplot as plt
import random
import cv2
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from skimage.util import random_noise
from Utils.mean_average_precision.mean_average_precision import MeanAveragePrecision2d
import dlib

print(dlib.DLIB_USE_CUDA)  # 输出 True 则表示启用了 GPU 支持
# 设置随机种子
random.seed(1)


# 检查并配置 GPU
def configure_gpu():
    print("测试tf是否启用GPU")
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
            for gpu in gpus:
                details = tf.config.experimental.get_device_details(gpu)
                print("GPU 设备名称:", details['device_name'])
                print("GPU 计算能力:", details['compute_capability'])
        except RuntimeError as e:
            print(e)


# 测试 TensorFlow GPU 加速
def test_tensorflow_gpu():
    print("测试 TensorFlow GPU 加速")
    start_time = time.time()
    a = tf.random.normal([10000, 10000])
    b = tf.random.normal([10000, 10000])
    c = tf.matmul(a, b)
    end_time = time.time()
    print("TensorFlow 矩阵乘法耗时: {:.2f} 秒".format(end_time - start_time))


# 测试 OpenCV CUDA 支持和光流算法检测
def test_opencv_cuda():
    print("测试 OpenCV 是否支持 GPU")
    print(cv2.getBuildInformation())
    help(cv2.cuda)

    try:
        print("测试 OpenCV CUDA 加速")
        img = cv2.imread('/kaggle/working/SoftNet-SpotME-test/test.jpg', cv2.IMREAD_COLOR)
        if img is None:
            print("无法加载测试图像，请检查路径")
            return

        img_gpu = cv2.cuda.GpuMat()
        img_gpu.upload(img)
        img_blur = cv2.cuda.bilateralFilter(img_gpu, 15, 75, 75)
        result = img_blur.download()
        print("OpenCV CUDA 加速可用")

        # 检查支持的光流算法
        optical_flow_algorithms = {
            "Brox Optical Flow": cv2.cuda.BroxOpticalFlow_create,
            # "DualTVL1 Optical Flow 1": cv2.cuda.DualTVL1OpticalFlow_create,
            "DualTVL1 Optical Flow 2": cv2.cuda.OpticalFlowDual_TVL1.create,
            "Farneback Optical Flow": cv2.cuda.FarnebackOpticalFlow_create,
            "SparsePyrLK Optical Flow": cv2.cuda.SparsePyrLKOpticalFlow_create,
            "DensePyrLK Optical Flow": cv2.cuda.DensePyrLKOpticalFlow_create,
        }

        for name, create_func in optical_flow_algorithms.items():
            try:
                optical_flow = create_func()
                print(f"{name} is supported")
            except AttributeError:
                print(f"{name} is not supported")
                continue
            except Exception as e:
                print(f"检测 {name} 时发生错误: {e}")
                continue

    except Exception as e:
        print("OpenCV CUDA 加速不可用:", e)



# 测试 skimage 随机噪声
def test_random_noise():
    print("测试 skimage 随机噪声")
    image = np.zeros((100, 100), dtype=np.uint8)
    noisy_image = random_noise(image, mode='s&p', amount=0.1)
    plt.imshow(noisy_image, cmap='gray')
    plt.title("随机噪声测试")
    plt.show()


# 测试 MeanAveragePrecision2d
def test_mean_average_precision():
    print("测试 MeanAveragePrecision2d")
    gt = np.array([[50, 50, 100, 100, 1, 1, 0]])  # 真实框，格式为 (x1, y1, x2, y2, score, class_id, ignore)
    pred = np.array([[55, 55, 90, 90, 0.9, 1]])  # 更小的预测框
    mean_ap = MeanAveragePrecision2d(num_classes=1)
    # 添加预测和真实框
    mean_ap.add(pred, gt)
    # 计算并输出 mAP
    ap = mean_ap.value(iou_thresholds=0.5)
    print("mAP 测试结果:", ap)


# 主程序
if __name__ == "__main__":
    configure_gpu()
    test_tensorflow_gpu()
    test_opencv_cuda()  # 测试 OpenCV CUDA 支持和光流算法检测
    test_random_noise()
    test_mean_average_precision()

