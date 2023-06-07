# -*- coding: utf-8 -*-
import argparse
import logging
import os
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from utils.data_loading import BasicDataset
# from unet import UNet
import Unet_Mnet
#from utils.utils import plot_img_and_mask
import os
from tqdm import tqdm, trange
import cv2
from rknn.api import RKNN
from utils.configs import CONFIGS
from utils.calc_diff import *


def predict_img(net,
                full_img,
                device,
                scale_factor=1,
                out_threshold=0.5):
    net.eval()
    img = torch.from_numpy(BasicDataset.preprocess(full_img, scale_factor, is_mask=False))
    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(img)

        if net.n_classes > 1:
            probs = F.softmax(output, dim=1)[0]
        else:
            probs = torch.sigmoid(output)[0]

        tf = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((full_img.size[1], full_img.size[0])),
            transforms.ToTensor()
        ])
        # import pdb
        # pdb.set_trace()
        full_mask = tf(probs.cpu()).squeeze()

    if net.n_classes == 1:
        return (full_mask > out_threshold).numpy()
    else:
        return F.one_hot(full_mask.argmax(dim=0), net.n_classes).permute(2, 0, 1).numpy()


###############################################################
config_unet = CONFIGS()
if not os.path.exists(config_unet.pt_pred_mask_save_folder):
    os.makedirs(config_unet.pt_pred_mask_save_folder)

net = Unet_Mnet.Unet_Mnet_model(n_classes=config_unet.overflow_classes)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# logging.info(f'Loading model {args.model}')
# logging.info(f'Using device {device}')
net.to(device=device)
net.load_state_dict(torch.load(config_unet.pt_path, map_location=device).state_dict())
net.eval()

debug = True
good_count = 0
normal_count = 0
no_bad_count = 0
total_count = 0
result_rate_list = []
GT_rate_list = []
#overflow_rect_limit = [100, 80, 420, 415]  # x1,y1,x2,y2(比较规范的矩形框)
overflow_rect_limit = [42, 44, 422, 422]  # x1,y1,x2,y2
overflow_rect_area = (int(overflow_rect_limit[3]) - int(overflow_rect_limit[1])) * (int(overflow_rect_limit[2]) - int(overflow_rect_limit[0]))
create_all_zero_img_flag = True
overflow_cout = 0



for filename in tqdm(os.listdir(config_unet.test_overflow_images_path)):
    logging.info(f'\nPredicting image {filename} ...')
    test_image_path = os.path.join(config_unet.test_overflow_images_path, filename)
    try:
        img = Image.open(test_image_path)
    except :
        print("图像路径不正确")
        logging.info(f'\n图像路径不正确： {filename} ...')
        continue

    width, height = img.size  # 获得原图像的宽和高, 手动破袋的图像尺寸是512*512
    size = (width, height)
    if True == create_all_zero_img_flag:
        img_ones = np.ones((width, height), np.uint8)
        img_ones[overflow_rect_limit[1]:overflow_rect_limit[3], overflow_rect_limit[0]:overflow_rect_limit[2]] = 0
        create_all_zero_img_flag = False

    img = img.resize(size, resample=0)
    test_mask_path = os.path.join(config_unet.test_overflow_masks_path, filename.replace(".jpg", ".png"))
    #test_mask_path = config_unet.test_masks_path + filename.split(".")[0] + ".png"
    mask = Image.open(test_mask_path)
    mask_np = np.array(mask)
    if debug:
        mask_save = mask_np * 100
        cv2.imwrite(os.path.join(config_unet.pt_gt_mask_save_folder, filename), mask_save)


    # pt模型预测
    pt_pred_mask = predict_img(net=net, full_img=img, scale_factor=1, out_threshold=0.6, device=device)
    pred = np.argmax(pt_pred_mask, axis=0)  # 取出最大值对应的索引，即类别
    pred = pred.astype(np.uint8)
    if debug:
        pred_save = pred.copy()

    if net.n_classes > 2:
        pred[pred != 0] = 1

    #overflow_roi_pix_sum = sum(map(sum, cv2.bitwise_and(img_ones, pred)))  # map(fund, a) equals [func(i) for i in a]  and return a list
    overflow_roi_pix_sum = np.sum(cv2.bitwise_and(img_ones, pred))
    overflow_roi_pix_rate = overflow_roi_pix_sum / overflow_rect_area
    print("\n满溢比率：", overflow_roi_pix_rate)

    if debug:
        # 保存图像，用于观察
        cv2.rectangle(pred_save, (overflow_rect_limit[0], overflow_rect_limit[1]), (overflow_rect_limit[2], overflow_rect_limit[3]), 255)
        if overflow_roi_pix_rate >= 0.1:
            cv2.putText(pred_save, "overflow", (overflow_rect_limit[0], overflow_rect_limit[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.75, 255, 1)
        pred_save = pred_save * 100
        cv2.imwrite(os.path.join(config_unet.pt_pred_mask_save_folder, filename), pred_save)

    # # calc the diff between pt_pred and gt
    # normal_count, no_bad_count, total_count, result_rate, GT_rate \
    #     = calc_diff_two_label_between_pred_gt(pred, mask_np, normal_count, no_bad_count, total_count)

    # calc the diff between pt_pred and gt
    normal_count, no_bad_count, total_count, error_rate \
        = calc_diff_one_label_between_pred_gt(pred, mask_np, normal_count, no_bad_count, total_count, 0)

    if error_rate < 0.05:
        good_count = good_count + 1
    if error_rate < 0.1:
        normal_count = normal_count + 1
    if error_rate < 0.15:
        no_bad_count = no_bad_count + 1


print(no_bad_count, "±15%")
print("no_bad_ACC:", no_bad_count / total_count)
print(normal_count, "±10%")
print("normal_ACC:", normal_count / total_count)
print(good_count, "±5%")
print("good_ACC:", good_count / total_count)
print(total_count, "total")
    
#


