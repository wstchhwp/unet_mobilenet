import os
import sys

import cv2

if __name__ == "__main__":

    names = os.listdir("/home/ly_jdw/Documents/Pytorch-UNet-master/data/zhuji/20")
    for sigle_name in names:
        abs_path = os.path.join("/home/ly_jdw/Documents/Pytorch-UNet-master/data/zhuji/20",sigle_name)
        img = cv2.imread(abs_path)
        img= cv2.resize(img,(512,512))
        cv2.imwrite(os.path.join("/home/ly_jdw/Documents/Pytorch-UNet-master/data/zhuji/20_resize",sigle_name.split(".")[0]+".jpg"),img)



# img = cv2.imread("/home/ly_jdw/Documents/Pytorch-UNet-master/data/imgs/20200313093411594.jpg")
# mm = img.shape
# cv2.imshow("111",img)
# cv2.waitKey(0)
#
# img_path = "/home/ly_jdw/Documents/Pytorch-UNet-master/data/imgs/"
# mask_path = "/home/ly_jdw/Documents/Pytorch-UNet-master/data/masks/"
#
# save_img_path =  "/home/ly_jdw/Documents/Pytorch-UNet-master/data/save_imgs/"
#
# mask_img_names = os.listdir("/home/ly_jdw/Documents/Pytorch-UNet-master/data/masks/")
#
#
# imgs_names = os.listdir("/home/ly_jdw/Documents/Pytorch-UNet-master/data/imgs/")
#
# for single_name in mask_img_names:
#     print("single_name:",single_name)
#     mask = cv2.imread(os.path.join(mask_path,single_name))
#
#     mask_pre_index = single_name.split(".")[0]
#
#     heigh = mask.shape[0]
#     width = mask.shape[1]
#     channel = mask.shape[2]
#
#
#     for i in imgs_names:
#         i_pre_index =i.split(".")[0]
#
#         if mask_pre_index == i_pre_index:
#
#             img_abs_path = os.path.join(img_path,i)
#             img = cv2.imread(img_abs_path)
#             img = cv2.resize(img,(width,heigh))
#             cv2.imwrite(os.path.join(save_img_path,i_pre_index+".jpg"),img)
#             # cv2.imshow("single_name",img)
#             # cv2.waitKey(10)


