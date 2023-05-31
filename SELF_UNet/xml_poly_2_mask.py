from PIL import Image
import numpy as np
import os
import json
import cv2 as cv
from IPython import display
from xml.dom.minidom import parse
import xml.dom.minidom

# 使用xml标注文件 与json标注文件，生成mask图像


# use_data_flag = 0-->使用测试样本
# use_data_flag = 1-->使用训练样本

use_data_flag = 1

# test_data
if use_data_flag == 0:

    """    
    #昌平测试集 
    labels_path = "/home/data/jdw_folder/deeplearning/Unet_Mnet_shoudongpodai/data_changping_test/labels/"
    xmls_path = "/home/data/jdw_folder/deeplearning/Unet_Mnet_shoudongpodai/data_changping_test/xml_files/"
    raw_imgs_path = "/home/data/jdw_folder/deeplearning/Unet_Mnet_shoudongpodai/data_changping_test/raw_imgs/"
    imgs_path = "/home/data/jdw_folder/deeplearning/Unet_Mnet_shoudongpodai/data_changping_test/imgs/"
    masks_path = "/home/data/jdw_folder/deeplearning/Unet_Mnet_shoudongpodai/data_changping_test/masks/"

    """    

    """
    #展厅测试集
    labels_path = "/home/data/jdw_folder/deeplearning/Unet_Mnet_shoudongpodai/data_zhanting_test/labels/"
    xmls_path = "/home/data/jdw_folder/deeplearning/Unet_Mnet_shoudongpodai/data_zhanting_test/xml_files/"
    raw_imgs_path = "/home/data/jdw_folder/deeplearning/Unet_Mnet_shoudongpodai/data_zhanting_test/raw_imgs/"
    imgs_path = "/home/data/jdw_folder/deeplearning/Unet_Mnet_shoudongpodai/data_zhanting_test/imgs/"
    masks_path = "/home/data/jdw_folder/deeplearning/Unet_Mnet_shoudongpodai/data_zhanting_test/masks/"
    """

    """  
    #福州测试集
    labels_path = "/home/data/jdw_folder/deeplearning/Unet_Mnet_shoudongpodai/data_fuzhou_test/labels/"
    xmls_path = "/home/data/jdw_folder/deeplearning/Unet_Mnet_shoudongpodai/data_fuzhou_test/xml_files/"
    raw_imgs_path = "/home/data/jdw_folder/deeplearning/Unet_Mnet_shoudongpodai/data_fuzhou_test/raw_imgs/"
    imgs_path = "/home/data/jdw_folder/deeplearning/Unet_Mnet_shoudongpodai/data_fuzhou_test/imgs/"
    masks_path = "/home/data/jdw_folder/deeplearning/Unet_Mnet_shoudongpodai/data_fuzhou_test/masks/"
    """

    """
    # 陆工测试
    labels_path = "/home/data/jdw_folder/deeplearning/Unet_Mnet_shoudongpodai/data_lugong_fuzhou/labels/"
    xmls_path = "/home/data/jdw_folder/deeplearning/Unet_Mnet_shoudongpodai/data_lugong_fuzhou/xml_files/"
    raw_imgs_path = "/home/data/jdw_folder/deeplearning/Unet_Mnet_shoudongpodai/data_lugong_fuzhou/raw_imgs/"
    imgs_path = "/home/data/jdw_folder/deeplearning/Unet_Mnet_shoudongpodai/data_lugong_fuzhou/imgs/"
    masks_path = "/home/data/jdw_folder/deeplearning/Unet_Mnet_shoudongpodai/data_lugong_fuzhou/masks/"
    """

    """    
    # 中粮-富春测试
    labels_path = "/home/data/jdw_folder/deeplearning/Unet_Mnet_shoudongpodai/data_zhongliang_fuchun_test/labels/"
    xmls_path = "/home/data/jdw_folder/deeplearning/Unet_Mnet_shoudongpodai/data_zhongliang_fuchun_test/xml_files/"
    raw_imgs_path = "/home/data/jdw_folder/deeplearning/Unet_Mnet_shoudongpodai/data_zhongliang_fuchun_test/raw_imgs/"
    imgs_path = "/home/data/jdw_folder/deeplearning/Unet_Mnet_shoudongpodai/data_zhongliang_fuchun_test/imgs/"
    masks_path = "/home/data/jdw_folder/deeplearning/Unet_Mnet_shoudongpodai/data_zhongliang_fuchun_test/masks/"
    """    
    
    # 机器人测试
    labels_path = "/home/data/jdw_folder/deeplearning/Unet_Mnet_shoudongpodai/data_jiqiren_test/labels/"
    xmls_path = "/home/data/jdw_folder/deeplearning/Unet_Mnet_shoudongpodai/data_jiqiren_test/xml_files/"
    raw_imgs_path = "/home/data/jdw_folder/deeplearning/Unet_Mnet_shoudongpodai/data_jiqiren_test/raw_imgs/"
    imgs_path = "/home/data/jdw_folder/deeplearning/Unet_Mnet_shoudongpodai/data_jiqiren_test/imgs/"
    masks_path = "/home/data/jdw_folder/deeplearning/Unet_Mnet_shoudongpodai/data_jiqiren_test/masks/"
    

# data
if use_data_flag == 1:
    """
    #展厅训练集
    labels_path = "/home/data/jdw_folder/deeplearning/Unet_Mnet_shoudongpodai/data_zhanting/labels/"
    xmls_path = "/home/data/jdw_folder/deeplearning/Unet_Mnet_shoudongpodai/data_zhanting/xml_files/"
    raw_imgs_path = "/home/data/jdw_folder/deeplearning/Unet_Mnet_shoudongpodai/data_zhanting/raw_imgs/"
    imgs_path = "/home/data/jdw_folder/deeplearning/Unet_Mnet_shoudongpodai/data_zhanting/imgs/"
    masks_path = "/home/data/jdw_folder/deeplearning/Unet_Mnet_shoudongpodai/data_zhanting/masks/"
    """

    """
    #昌平训练集
    labels_path = "/home/data/jdw_folder/deeplearning/Unet_Mnet_shoudongpodai/data_changping/labels/"
    xmls_path = "/home/data/jdw_folder/deeplearning/Unet_Mnet_shoudongpodai/data_changping/xml_files/"
    raw_imgs_path = "/home/data/jdw_folder/deeplearning/Unet_Mnet_shoudongpodai/data_changping/raw_imgs/"
    imgs_path = "/home/data/jdw_folder/deeplearning/Unet_Mnet_shoudongpodai/data_changping/imgs/"
    masks_path = "/home/data/jdw_folder/deeplearning/Unet_Mnet_shoudongpodai/data_changping/masks/"
    """
    
    """    
    #中粮、富春训练集
    labels_path = "/home/data/jdw_folder/deeplearning/Unet_Mnet_shoudongpodai/data_zhongliang_fuchun/labels/"
    xmls_path = "/home/data/jdw_folder/deeplearning/Unet_Mnet_shoudongpodai/data_zhongliang_fuchun/xml_files/"
    raw_imgs_path = "/home/data/jdw_folder/deeplearning/Unet_Mnet_shoudongpodai/data_zhongliang_fuchun/raw_imgs/"
    imgs_path = "/home/data/jdw_folder/deeplearning/Unet_Mnet_shoudongpodai/data_zhongliang_fuchun/imgs/"
    masks_path = "/home/data/jdw_folder/deeplearning/Unet_Mnet_shoudongpodai/data_zhongliang_fuchun/masks/"
    """
           
    """
    # 福州训练集
    labels_path = "/home/data/jdw_folder/deeplearning/Unet_Mnet_shoudongpodai/data_fuzhou/labels/"
    xmls_path = "/home/data/jdw_folder/deeplearning/Unet_Mnet_shoudongpodai/data_fuzhou/xml_files/"
    raw_imgs_path = "/home/data/jdw_folder/deeplearning/Unet_Mnet_shoudongpodai/data_fuzhou/raw_imgs/"
    imgs_path = "/home/data/jdw_folder/deeplearning/Unet_Mnet_shoudongpodai/data_fuzhou/imgs/"
    masks_path = "/home/data/jdw_folder/deeplearning/Unet_Mnet_shoudongpodai/data_fuzhou/masks/"
    
    """
    
    """
    # 昌平 破袋+未破袋
    labels_path = "/home/data/jdw_folder/deeplearning/Unet_Mnet_shoudongpodai/data_changping_total/labels/"
    xmls_path = "/home/data/jdw_folder/deeplearning/Unet_Mnet_shoudongpodai/data_changping_total/xml_files/"
    raw_imgs_path = "/home/data/jdw_folder/deeplearning/Unet_Mnet_shoudongpodai/data_changping_total/raw_imgs/"
    imgs_path = "/home/data/jdw_folder/deeplearning/Unet_Mnet_shoudongpodai/data_changping_total/imgs/"
    masks_path = "/home/data/jdw_folder/deeplearning/Unet_Mnet_shoudongpodai/data_changping_total/masks/"
    """
    
        
    #机器人春训练集
    labels_path = "/home/data/jdw_folder/deeplearning/Unet_Mnet_shoudongpodai/data_jiqiren/labels/"
    xmls_path = "/home/data/jdw_folder/deeplearning/Unet_Mnet_shoudongpodai/data_jiqiren/xml_files/"
    raw_imgs_path = "/home/data/jdw_folder/deeplearning/Unet_Mnet_shoudongpodai/data_jiqiren/raw_imgs/"
    imgs_path = "/home/data/jdw_folder/deeplearning/Unet_Mnet_shoudongpodai/data_jiqiren/imgs/"
    masks_path = "/home/data/jdw_folder/deeplearning/Unet_Mnet_shoudongpodai/data_jiqiren/masks/"
    
temp_count = 0

for label_file in os.listdir(labels_path):
    if label_file.endswith(".json"):
        xmls_file_path = xmls_path + label_file.replace(".json", ".xml")
        if os.path.exists(xmls_file_path):
            try:
                dom_tree = xml.dom.minidom.parse(xmls_file_path)
                collection = dom_tree.documentElement
                xml_object = collection.getElementsByTagName("object")[0]
                object_box = xml_object.getElementsByTagName("bndbox")[0]
                xmin = int(object_box.getElementsByTagName("xmin")[0].childNodes[0].data)
                ymin = int(object_box.getElementsByTagName("ymin")[0].childNodes[0].data)
                xmax = int(object_box.getElementsByTagName("xmax")[0].childNodes[0].data)
                ymax = int(object_box.getElementsByTagName("ymax")[0].childNodes[0].data)

                #             break
                label_file_path = labels_path + label_file
                label_jsons = json.load(open(label_file_path, "r"))
                image_name = label_jsons["imagePath"].split("/")[-1]
                image_path = raw_imgs_path + image_name
                raw_image = cv.imread(image_path)

            #                 print(image_path)
            #                 print(image_path)
            except:
                continue
            if raw_image is None:
                print("error image")
                continue
            #             if np.shape(raw_image)[0] != 1080 and np.shape(raw_image)[1] != 1920:
            #                 print("error image size")
            #                 continue

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
            #             print(np.shape(raw_image)[0])
            #             print(np.shape(raw_image)[1])
            #             print(ymax - ymin)
            #             print(xmax - xmin)
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

            mask = mask_temp[ymin:ymax, xmin:xmax]
            raw_image_ROI = raw_image[ymin:ymax, xmin:xmax]

            resized_mask = cv.resize(mask, (512, 512), cv.INTER_NEAREST)
            resized_img = cv.resize(raw_image_ROI, (512, 512), cv.INTER_CUBIC)
            
            # 这里统一将img 保存为jpg格式
            image_save_path = imgs_path + image_name.split(".")[0]+".jpg"
            
            print("image_name = ",image_name)
            
            # 若原图像为jpg结尾--》 mask以png结尾
            
            
            if image_name.endswith(".jpg"):
                mask_path = masks_path + image_name.replace(".jpg", ".png")
            
            
            # 若原图像为jpg结尾--》 mask以png结尾
            if image_name.endswith(".jpeg"):
                mask_path = masks_path + image_name.replace(".jpeg", ".png")
                
            cv.imwrite(mask_path, resized_mask)
            cv.imwrite(image_save_path, resized_img)

#             display.display(Image.fromarray(resized_mask*60))
# #             display.display(Image.fromarray(mask_temp * 60))
# #             display.display(Image.fromarray(raw_image))
#             display.display(Image.fromarray(resized_img))
#             print(image_save_path)
#             temp_count = temp_count + 1
#             if temp_count >= 100:
#                 break
# -
"""
image_save_path

poly_list

poly_area_np
"""

