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

def predict_img_quantize(net,
                         full_img,
                         device,
                         scale_factor=0.5,
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

net = Unet_Mnet.Unet_Mnet_model(n_classes=config_unet.n_class)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net.to(device=device)
# net.load_state_dict(torch.jit.load(config_unet.pt_jit_path, map_location=device).state_dict())  # load jit_pt
net.load_state_dict(torch.load(config_unet.pt_path, map_location=device).state_dict())  # load pt

DEBUG = True
good_count = 0
normal_count = 0
no_bad_count = 0
total_count = 0
result_rate_list = []
GT_rate_list = []
resize_width = 512
resize_height = 512
normal_flag = True
not_normal_count = 0
for filename in tqdm(os.listdir(config_unet.test_images_path)):
    logging.info(f'\nPredicting image {filename} ...')
    test_image_path = os.path.join(config_unet.test_images_path, filename)
#     print(test_image_path)
    #img = Image.open('./7b5afbb3abe748faae67bf8580b18cfc.jpeg') #test_image_path)
    img = Image.open(test_image_path)
    width, height = img.size  # 获得原图像的宽和高, 手动破袋的图像宽高resize到512*512
    #size = (width, height)
    size = (resize_width, resize_height)
    img = img.resize(size, resample=0)
    test_mask_path = os.path.join(config_unet.test_masks_path, filename.replace(".jpg", ".png"))
    #test_mask_path = config_unet.test_masks_path + filename.split(".")[0] + ".png"
    mask = Image.open(test_mask_path)
    mask_np = np.array(mask)
    if DEBUG:
        mask_save = mask_np * 80
        cv2.imwrite(os.path.join(config_unet.pt_gt_mask_save_folder, filename), mask_save)


    # pt模型预测
    pt_pred_mask = predict_img(net=net, full_img=img, scale_factor=1, out_threshold=0.5, device=device)

    pred = np.argmax(pt_pred_mask, axis=0)
    if DEBUG:
        pred_save = pred * 100
        cv2.imwrite(os.path.join(config_unet.pt_pred_mask_save_folder, filename), pred_save)

    # calc the diff between pt_pred and gt
    normal_count, no_bad_count, total_count, result_rate, GT_rate, normal_flag \
        = calc_diff_three_label_between_pred_gt(pred, mask_np, normal_count, no_bad_count, total_count)

    if True == normal_flag:
        if abs(GT_rate - result_rate) < 0.05:
            good_count = good_count + 1
        if abs(GT_rate - result_rate) < 0.1:
            normal_count = normal_count + 1
        if abs(GT_rate - result_rate) < 0.15:
            no_bad_count = no_bad_count + 1
    else:
        not_normal_count += 1

if True == normal_flag:
    print(no_bad_count, "±15%")
    print("no_bad_ACC:", no_bad_count / total_count)
    print(normal_count, "±10%")
    print("normal_ACC:", normal_count / total_count)
    print(good_count, "±5%")
    print("good_ACC:", good_count / total_count)
    print(total_count, "total")
else:
    print("=======返回的所有像素值和小于{}不正常，可能是无桶，可能是盖着盖子".format(config_unet.pix_sum_thre))
    print("有问题的桶的个数：", not_normal_count)
#


