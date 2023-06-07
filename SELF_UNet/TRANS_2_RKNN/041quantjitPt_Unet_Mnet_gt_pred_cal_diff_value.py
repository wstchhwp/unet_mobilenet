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
import Unet_Mnet_Quant
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
                         scale_factor=1,
                         out_threshold=0.5):
    img = torch.from_numpy(BasicDataset.preprocess(full_img, scale_factor, is_mask=False))
    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(img)
        # todo: 22.7.6新加
        #output = torch.dequantize(output)

        # todo: 22.7.6註釋測試
        # if net.n_classes > 1:
        #     probs = F.softmax(output, dim=1)[0]
        # else:
        #     probs = torch.sigmoid(output)[0]
        probs = F.softmax(output, dim=1)[0]

        tf = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((full_img.size[1], full_img.size[0])),
            transforms.ToTensor()
        ])
        # import pdb
        # pdb.set_trace()
        full_mask = tf(probs.cpu()).squeeze()

    # todo: 22.7.6註釋測試
    # if net.n_classes == 1:
    #     return (full_mask > out_threshold).numpy()
    # else:
    #     return F.one_hot(full_mask.argmax(dim=0), net.n_classes).permute(2, 0, 1).numpy()
    return F.one_hot(full_mask.argmax(dim=0), 3).permute(2, 0, 1).numpy()


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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# # logging.info(f'Using device {device}')
# net.to(device=device)
# net.load_state_dict(torch.load(config_unet.pt_jit_path, map_location=device).state_dict())

torch.backends.quantized.engine = 'qnnpack'
pt_model = torch.jit.load(config_unet.pt_jit_path).eval()
net = pt_model

"""
if os.path.exists(config_unet.pt_jit_path):
    net_state_dict = torch.load(config_unet.pt_jit_path)
    model_fp32 = Unet_Mnet.Unet_Mnet_model(n_classes=3)
    model_fp32.qconfig = torch.quantization.get_default_qconfig('fbgemm')
    model_fp32_prepared = torch.quantization.prepare(model_fp32)
    model_int8 = torch.quantization.convert(model_fp32_prepared)
    nn_state_dict = net_state_dict.state_dict()
    model_int8.load_state_dict(nn_state_dict)
    net = model_int8
    net.eval()

elif os.path.exists(config_unet.pth_jit_path):
    state_dict = torch.load(config_unet.pth_jit_path)
    model_fp32 = Unet_Mnet.Unet_Mnet_model(n_classes=3)
    model_fp32.qconfig = torch.quantization.get_default_qconfig('fbgemm')
    model_fp32_prepared = torch.quantization.prepare(model_fp32)
    model_int8 = torch.quantization.convert(model_fp32_prepared)
    model_int8.load_state_dict(state_dict)
    net = model_int8
    net.eval()
#print("state_dict:", state_dict)
#print(net)
"""


# net_state_dict = torch.load(config_unet.pt_jit_path, map_location='cpu')
# model_int8 = Unet_Mnet.Unet_Mnet_model(n_classes=3)
# model_int8.qconfig = torch.quantization.get_default_qconfig('fbgemm')
# nn_state_dict = net_state_dict.state_dict()
# model_int8.load_state_dict(nn_state_dict)
# net = model_int8
# net.eval()



good_count = 0
normal_count = 0
no_bad_count = 0
total_count = 0
result_rate_list = []
GT_rate_list = []

for filename in tqdm(os.listdir(config_unet.test_images_path)):
    logging.info(f'\nPredicting image {filename} ...')
    test_image_path = config_unet.test_images_path + filename
#     print(test_image_path)
    #img = Image.open('./7b5afbb3abe748faae67bf8580b18cfc.jpeg') #test_image_path)
    img = Image.open(test_image_path)
    width, height = img.size  # 获得原图像的宽和高, 手动破袋的图像宽高是512*512
    size = (width, height)
    img = img.resize(size, resample=0)
    test_mask_path = config_unet.test_masks_path + filename.replace(".jpg", ".png")
    #test_mask_path = config_unet.test_masks_path + filename.split(".")[0] + ".png"
    mask = Image.open(test_mask_path)
    mask_np = np.array(mask)

    # pt模型预测
    pt_pred_mask = predict_img_quantize(net=net, full_img=img, scale_factor=0.5, out_threshold=0.6, device=device)
    #pt_pred_mask = predict_img(net=net, full_img=img, scale_factor=0.5, out_threshold=0.6, device=device)

    pred = np.argmax(pt_pred_mask, axis=0)
    pred_save = pred * 100
    cv2.imwrite(config_unet.pt_pred_mask_save_folder + filename, pred_save)

    # calc the diff between pt_pred and gt
    normal_count, no_bad_count, total_count, result_rate, GT_rate \
        = calc_diff_two_label_between_pred_gt(pred, mask_np, normal_count, no_bad_count, total_count)

    if abs(GT_rate - result_rate) < 0.1:
        normal_count = normal_count + 1
    if abs(GT_rate - result_rate) < 0.15:
        no_bad_count = no_bad_count + 1
#     elif (result_rate - GT_rate) < 0.15 and (result_rate - GT_rate) >= 0.1:
#         normal_count = normal_count + 1
#     if abs(GT_rate - result_rate) >= 0.1:
#         print(result_rate)
#         print(GT_rate)
#         display.display(img)
#     if GT_rate < 0.6:
#         print(result_rate)
#         print(GT_rate)
        
    
print(no_bad_count, "±15%")
print(no_bad_count/total_count)
print(normal_count, "±10%")
print(normal_count/total_count)
print(total_count, "total")
    
#


