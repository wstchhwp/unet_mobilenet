from PIL import Image
import numpy as np
import os
import json
import cv2 as cv
from IPython import display
from tqdm import tqdm


# 手动破袋 未使用xml标注文件


# 0 -->生成测试样本
# 1 -->生成训练样本
use_data_flag = 1

if use_data_flag == 0:
    # +
    # test data
    # 标注的类别
    labels_path    ="/home/data/jdw_folder/deeplearning/Unet_Mnet_shoudongpodai/data_changping_pw/labels/"

    # 原图
    raw_imgs_path  ="/home/data/jdw_folder/deeplearning/Unet_Mnet_shoudongpodai/data_changping_pw/raw_imgs/"

    # 保存的resize 图 512 *512
    imgs_path      ="/home/data/jdw_folder/deeplearning/Unet_Mnet_shoudongpodai/data_changping_pw/imgs/"

    # 保存的mask图 512*512
    masks_path     ="/home/data/jdw_folder/deeplearning/Unet_Mnet_shoudongpodai/data_changping_pw/masks/"



if use_data_flag == 1:
    # data
    labels_path   = "/home/data/jdw_folder/deeplearning/Unet_Mnet_shoudongpodai/data_changping_pw/labels/"
    raw_imgs_path = "/home/data/jdw_folder/deeplearning/Unet_Mnet_shoudongpodai/data_changping_pw/raw_imgs/"
    imgs_path     = "/home/data/jdw_folder/deeplearning/Unet_Mnet_shoudongpodai/data_changping_pw/imgs/"
    masks_path    = "/home/data/jdw_folder/deeplearning/Unet_Mnet_shoudongpodai/data_changping_pw/masks/"


temp_count = 0
for label_file in tqdm(os.listdir(labels_path)):
    if label_file.endswith(".json"):
        label_file_path = labels_path + label_file
        label_jsons = json.load(open(label_file_path, "r"))
        image_name = label_jsons["imagePath"].split("/")[-1]
        image_path = raw_imgs_path + image_name

        raw_image = cv.imread(image_path)
        if raw_image is None:
            print(image_path)
            continue

        # 这里原图 就是 截取后的图，肯定不满足 1920*1080 的尺寸
        # if np.shape(raw_image)[0] != 1080 and np.shape(raw_image)[1] != 1920:
        #     print(np.shape(raw_image)[0])
        #     print(np.shape(raw_image)[1])
        #
        #     print(image_path)
        #     continue

        poly_list = []
        poly_area_list = []
        for poly in label_jsons["shapes"]:
            temp_BG = np.zeros([np.shape(raw_image)[0], np.shape(raw_image)[1]], dtype=np.uint8)
            points = np.array(poly["points"], dtype=np.int32)
            cv.fillPoly(temp_BG, [points], (1))
            poly_area = np.sum(temp_BG)
            poly_area_list.append(poly_area)
            poly_list.append(poly)

        poly_area_np = np.array(poly_area_list)
        poly_area_sort_index = np.argsort(-1 * poly_area_np)

        mask_temp = np.zeros([np.shape(raw_image)[0], np.shape(raw_image)[1]], dtype=np.uint8)
        for index_of_poly in poly_area_sort_index:
            poly_single = poly_list[index_of_poly]

            points = np.array(poly_single["points"], dtype=np.int32)
            if poly_single["label"].upper() == "PW":
                cv.fillPoly(mask_temp, [points], (1))
            if poly_single["label"].upper() == "NPW":
                cv.fillPoly(mask_temp, [points], (2))
            if poly_single["label"].upper() == "BAG":
                cv.fillPoly(mask_temp, [points], (3))
            if poly_single["label"].upper() == "BG":
                cv.fillPoly(mask_temp, [points], (0))

        resized_mask = cv.resize(mask_temp, (512, 512), cv.INTER_NEAREST)
        resized_img = cv.resize(raw_image, (512, 512), cv.INTER_CUBIC)
        if image_name.endswith(".jpeg"):
            mask_path = masks_path + image_name.replace(".jpeg", ".png")
            image_save_path = imgs_path + image_name.replace(".jpeg", ".jpg")
        if image_name.endswith(".jpg"):
            mask_path = masks_path + image_name.replace(".jpg", ".png")
            image_save_path = imgs_path + image_name
            
        cv.imwrite(image_save_path, resized_img)
        cv.imwrite(mask_path, resized_mask)

#     temp_count = temp_count + 1
#     if temp_count >= 60:
#         break
#     if 1:
#         print(mask_path)
#         print(image_save_path)
#         display.display(Image.fromarray(resized_mask*80))
#         display.display(Image.fromarray(resized_img))
# -


