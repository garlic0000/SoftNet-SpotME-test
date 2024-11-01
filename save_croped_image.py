import os
import shutil
import glob
import natsort
import pickle
import dlib
import numpy as np
import cv2
from tqdm import tqdm
from pathlib import Path

CASME_sq_rawpic_root_path = "/kaggle/input/casme-sq/rawpic/rawpic"
dir_crop_root_path = "/kaggle/working/rawpic_crop_no_resize"

detector_model = "/kaggle/working/SoftNet-SpotME-test/Utils/mmod_human_face_detector.dat"


def get_rawpic_count(root_path):
    """
    递归地统计数据集中所有 .jpg 图片的数量
    遍历所有子目录，最终返回图片总数
    Args:
        root_path: 图片根目录

    Returns:图片总数
    """
    count = 0
    for sub in Path(root_path).iterdir():
        if sub.is_dir():
            for vid in sub.iterdir():
                if vid.is_dir():
                    count += len(glob.glob(os.path.join(
                        str(vid), "*.jpg")))
    return count


def crop_images(dataset_name):
    """
    对数据集中的图像进行裁剪，提取出人脸区域并保存成新图片
    使用 dlib 的 HOG 面部检测器来检测每张图片中的人脸
    对人脸进行裁剪和缩放
    裁剪后的图片以 128x128 的分辨率保存
    Args:
        dataset_name:处理的数据集名称

    Returns: 已裁剪和缩放的图片

    """
    # # 加载 HOG 人脸检测器
    # face_detector = dlib.get_frontal_face_detector()
    # 使用cnn_face_detector代替face_detector
    cnn_face_detector = dlib.cnn_face_detection_model_v1(detector_model)

    if dataset_name == 'CASME_sq':
        sum_count = get_rawpic_count(CASME_sq_rawpic_root_path)
        print("rawpic count = ", sum_count)
        with tqdm(total=sum_count) as tq:  # 进度条
            # s15, s16, s19等subject目录名称
            for sub in Path(CASME_sq_rawpic_root_path).iterdir():
                if not sub.is_dir():
                    continue
                # 把裁剪的图片保存至 'rawpic_crop'
                # 创建新目录 'rawpic_crop'
                if not os.path.exists(dir_crop_root_path):
                    os.mkdir(dir_crop_root_path)

                # 为每一个subject创建目录
                # # /kaggle/input/casme2/rawpic/rawpic/s15/15_0101disgustingteeth
                #
                #                 s_name = "casme_0{}".format(sub_item.name[1:])
                #                 v_name = "casme_0{}".format(type_item.name[0:7])
                s_name = "casme_0{}".format(sub.name[1:])
                dir_crop_sub = os.path.join(dir_crop_root_path, s_name)
                if os.path.exists(dir_crop_sub):
                    shutil.rmtree(dir_crop_sub)
                os.mkdir(dir_crop_sub)
                print()  # 输出一个空行
                print('Subject', sub.name)
                for vid in sub.iterdir():
                    if not vid.is_dir():
                        continue
                    print()  # 输出一个空行
                    print("Video", vid.name)
                    # 为每段视频创建目录
                    v_name = "casme_0{}".format(vid.name[0:7])
                    dir_crop_sub_vid = os.path.join(dir_crop_sub, v_name)
                    if os.path.exists(dir_crop_sub_vid):
                        shutil.rmtree(dir_crop_sub_vid)
                    os.mkdir(dir_crop_sub_vid)
                    # natsort 是一个第三方库，用于执行“自然排序”，
                    # 也就是按人类习惯的方式进行排序。
                    # 例如，按自然顺序，img2.jpg 会排在 img10.jpg 前面，而不是后面。
                    dir_crop_sub_vid_img_list = glob.glob(os.path.join(str(vid), "*.jpg"))
                    # 读取每张图片
                    for dir_crop_sub_vid_img in natsort.natsorted(dir_crop_sub_vid_img_list):
                        img = os.path.basename(dir_crop_sub_vid_img)  # 获取文件名，例如 'img001.jpg'
                        img_name = img[3:-4]  # 获取 '001'
                        # 读入图片
                        image = cv2.imread(dir_crop_sub_vid_img)
                        # # 运行HOG人脸检测器
                        # detected_faces = face_detector(image, 1)
                        # 使用cnn_face_detector代替face_detector
                        detected_faces = cnn_face_detector(image, 1)
                        if img_name == '001':
                            # 使用第一帧（图片名为 001）的面部作为参考框架来确定面部的裁剪边界
                            # 后续帧中将使用同样的边界
                            for face_rect in detected_faces:
                                # face_top = face_rect.top()
                                # face_bottom = face_rect.bottom()
                                # face_left = face_rect.left()
                                # face_right = face_rect.right()
                                # cnn_face_detector(image, 1) 返回的是包含 dlib.mmod_rectangle 对象的列表
                                # 使用 face_rect.rect.top()等来访问面部边界框的位置
                                face_top = face_rect.rect.top()
                                face_bottom = face_rect.rect.bottom()
                                face_left = face_rect.rect.left()
                                face_right = face_rect.rect.right()

                        face = image[face_top:face_bottom, face_left:face_right]  # 裁剪人脸区域
                        # 不调整尺寸
                        # face = cv2.resize(face, (128, 128))  # 调整尺寸为 128x128
                        # 保存图片
                        cv2.imwrite(os.path.join(dir_crop_sub_vid, "img{}.jpg").format(img_name), face)
                        tq.update()  # 更新进度


if __name__ == "__main__":
    crop_images('CASME_sq')
