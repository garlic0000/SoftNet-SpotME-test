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

CASME_sq_rawpic_root_path = "/kaggle/working/rawpic"
dir_crop_root_path = "/kaggle/working/rawpic_crop"


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
    # 加载 HOG 人脸检测器
    face_detector = dlib.get_frontal_face_detector()
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
                dir_crop_sub = os.path.join(dir_crop_root_path, sub.name)
                if os.path.exists(dir_crop_sub):
                    shutil.rmtree(dir_crop_sub)
                os.mkdir(dir_crop_sub)
                print('Subject', sub.name)
                for vid in sub.iterdir():
                    if not vid.is_dir():
                        continue
                    print("Video", vid.name)
                    # 为每段视频创建目录
                    dir_crop_sub_vid = os.path.join(dir_crop_sub, vid.name)
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
                        # 运行HOG人脸检测器
                        detected_faces = face_detector(image, 1)

                        if img_name == '001':
                            # 使用第一帧（图片名为 001）的面部作为参考框架来确定面部的裁剪边界
                            # 后续帧中将使用同样的边界
                            for face_rect in detected_faces:
                                face_top = face_rect.top()
                                face_bottom = face_rect.bottom()
                                face_left = face_rect.left()
                                face_right = face_rect.right()

                        face = image[face_top:face_bottom, face_left:face_right]  # 裁剪人脸区域
                        face = cv2.resize(face, (128, 128))  # 调整尺寸为 128x128
                        # 保存图片
                        cv2.imwrite(os.path.join(dir_crop_sub_vid, "img{}.jpg").format(img_name), face)
                        tq.update()  # 更新进度

    # 地址还没进行修改
    elif (dataset_name == 'SAMMLV'):
        if os.path.exists(dataset_name + '/SAMM_longvideos_crop'):  # Delete dir if exist and create new dir
            shutil.rmtree(dataset_name + '/SAMM_longvideos_crop')
        os.mkdir(dataset_name + '/SAMM_longvideos_crop')

        for vid in glob.glob(dataset_name + '/SAMM_longvideos/*'):
            count = 0
            dir_crop = dataset_name + '/SAMM_longvideos_crop/' + vid.split('/')[-1]

            if os.path.exists(dir_crop):  # Delete dir if exist and create new dir
                shutil.rmtree(dir_crop)
            os.mkdir(dir_crop)
            print('Video', vid.split('/')[-1])
            for dir_crop_img in natsort.natsorted(glob.glob(vid + '/*.jpg')):
                img = dir_crop_img.split('/')[-1].split('.')[0]
                count = img[-4:]  # Get img num Ex 0001,0002,...,2021
                # Load the image
                image = cv2.imread(dir_crop_img)

                # Run the HOG face detector on the image data
                detected_faces = face_detector(image, 1)

                # Loop through each face we found in the image
                if (count == '0001'):  # Use first frame as reference frame
                    for i, face_rect in enumerate(detected_faces):
                        face_top = face_rect.top()
                        face_bottom = face_rect.bottom()
                        face_left = face_rect.left()
                        face_right = face_rect.right()

                face = image[face_top:face_bottom, face_left:face_right]
                face = cv2.resize(face, (128, 128))

                cv2.imwrite(dir_crop + "/{}.jpg".format(count), face)


def load_images(dataset_name):
    images = []
    subjects = []
    subjectsVideos = []

    if dataset_name == 'CASME_sq':
        for i, dir_sub in enumerate(natsort.natsorted(Path(dir_crop_root_path).iterdir())):
            print('Subject: ' + dir_sub.name)
            subjects.append(dir_sub.name)
            subjectsVideos.append([])
            for dir_sub_vid in natsort.natsorted(dir_sub.iterdir()):
                subjectsVideos[-1].append(dir_sub_vid.name.split('_')[1][
                                          :4])  # Ex:'CASME_sq/rawpic_aligned/s15/15_0101disgustingteeth' -> '0101'
                image = []
                for dir_sub_vid_img in natsort.natsorted(glob.glob(os.path.join(str(dir_sub_vid), "*.jpg"))):
                    image.append(cv2.imread(dir_sub_vid_img, 0))  # 加载灰度图像
                images.append(np.array(image))

    elif (dataset_name == 'SAMMLV'):
        for i, dir_vid in enumerate(natsort.natsorted(glob.glob(dataset_name + "/SAMM_longvideos_crop/*"))):
            print('Subject: ' + dir_vid.split('/')[-1].split('_')[0])
            subject = dir_vid.split('/')[-1].split('_')[0]
            subjectVideo = dir_vid.split('/')[-1]
            if (subject not in subjects):  # Only append unique subject name
                subjects.append(subject)
                subjectsVideos.append([])
            subjectsVideos[-1].append(dir_vid.split('/')[-1])

            image = []
            for dir_vid_img in natsort.natsorted(glob.glob(dir_vid + "/*.jpg")):
                image.append(cv2.imread(dir_vid_img, 0))
            image = np.array(image)
            images.append(image)

    return images, subjects, subjectsVideos


def save_images_pkl(dataset_name, images, subjectsVideos, subjects):
    # 设置保存的pkl路径
    # 好像是当前路径
    pickle.dump(images, open(dataset_name + "_images_crop.pkl", "wb"))
    pickle.dump(subjectsVideos, open(dataset_name + "_subjectsVideos_crop.pkl", "wb"))
    pickle.dump(subjects, open(dataset_name + "_subjects_crop.pkl", "wb"))


def load_images_pkl(dataset_name):
    images = pickle.load(open(dataset_name + "_images_crop.pkl", "rb"))
    subjectsVideos = pickle.load(open(dataset_name + "_subjectsVideos_crop.pkl", "rb"))
    subjects = pickle.load(open(dataset_name + "_subjects_crop.pkl", "rb"))
    return images, subjectsVideos, subjects


if __name__ == "__main__":
    crop_images('CASME_sq')
