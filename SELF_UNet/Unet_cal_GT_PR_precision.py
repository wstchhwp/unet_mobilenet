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
from utils.utils import plot_img_and_mask
import os
from tqdm import tqdm, trange

import time


from unet import UNet
from unet.self_unet_backbone import My_Unet

from unet.syx_Unet_Mnet import Unet_Mnet_model


from unet.unet_mbv2 import Unet_MobileNetV2


# todo 原　UNET 网络
# net = UNet(n_channels=3, n_classes=args.classes, bilinear=args.bilinear)
# todo 自写　UNET 网络
#net = My_Unet(num_class=4)
# todo 宋　UNET_MNET
# net = Unet_Mnet_model(n_classes=4)
# todo 自写　UNET_MNET
# net = Unet_MobileNetV2(num_classes=4)



os.environ['CUDA_VISIBLE_DEVICES'] = "0"


# +
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

        full_mask = tf(probs.cpu()).squeeze()

    if net.n_classes == 1:
        return (full_mask > out_threshold).numpy()
    else:
        return F.one_hot(full_mask.argmax(dim=0), net.n_classes).permute(2, 0, 1).numpy(), full_mask.argmax(dim=0).numpy()
    
def mask_to_image(mask: np.ndarray):
    if mask.ndim == 2:
        return Image.fromarray((mask * 255).astype(np.uint8))
    elif mask.ndim == 3:
        print(mask.ndim )
        return Image.fromarray((np.argmax(mask, axis=0) * 255 / mask.shape[0]).astype(np.uint8))
    

model_pth_path = "/home/ly_jdw/Documents/0428_Unet/SELF_UNet/data_zhanting_checkpoints/checkpoint_epoch5.pth"

# 展厅测试
test_images_path = "/home/ly_jdw/Documents/0428_Unet/SELF_UNet/data_zhanting_test/imgs/"
test_masks_path = "/home/ly_jdw/Documents/0428_Unet/SELF_UNet/data_zhanting_test/masks/"



# todo 原　UNET 网络
# net = UNet(n_channels=3, n_classes=4, bilinear=False)
# todo 自写　UNET 网络
# net = My_Unet(num_class=4)
# todo 宋　UNET_MNET
# net = Unet_Mnet_model(n_classes=4)
# todo 自写　UNET_MNET
net = Unet_MobileNetV2(num_classes=4)




# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')
print("device=",device)
# logging.info(f'Loading model {args.model}')
# logging.info(f'Using device {device}')

net.to(device=device)
# net.load_state_dict(torch.load(model_pth_path, map_location=device).state_dict())
net.load_state_dict(torch.load(model_pth_path, map_location=device))

good_count = 0
normal_count = 0
no_bad_count = 0
no_pw_count = 0
total_count = 0
result_rate_list = []
GT_rate_list = []

for filename in tqdm(os.listdir(test_images_path)):
    
    logging.info(f'\nPredicting image {filename} ...')
    test_image_path = test_images_path + filename
    
    #print("test_image_path = ", test_image_path)
    img = Image.open(test_image_path)
#     size = (1920, 1024)
#     img = img.resize(size, resample=0)
    test_mask_path = test_masks_path + filename.replace(".jpg", ".png").replace("imgs", "")
    mask = Image.open(test_mask_path)
    mask_scale_factor=1
#     mask = mask.resize((int(512 * mask_scale_factor), int(512 * mask_scale_factor)), Image.NEAREST)
    mask_np = np.array(mask)
    
    #
    start_time = time.time()
    pred_mask, mask_for_show= predict_img(net=net, full_img=img, scale_factor=1, out_threshold=0.6, device=device)
    end_time = time.time()
    print("time cost:", float(end_time - start_time) * 1000.0, "ms")
    #
    pred = (np.argmax(pred_mask, axis=0))
    pred_all = pred.copy()
    pred_all[pred_all==2] = 1
    pred_all[pred_all==3] = 1
    pred_all_area = np.sum(pred_all)
    pred_pw = pred.copy()
    pred_pw[pred_pw==2] = 0
    pred_pw[pred_pw==3] = 0
    pred_pw_area = np.sum(pred_pw)

    
    GT_all = mask_np.copy()
    GT_all[GT_all==2] = 1
    GT_all[GT_all==3] = 1
    GT_all_area = np.sum(GT_all)
    GT_pw = mask_np.copy()
    GT_pw[GT_pw==2] = 0
    GT_pw[GT_pw==3] = 0
    GT_pw_area = np.sum(GT_pw)

            
    result_rate = pred_pw_area /(pred_all_area + 1)
    result_rate_list.append(result_rate)
    
    GT_rate = GT_pw_area / (GT_all_area + 1)
    GT_rate_list.append(GT_rate)
    
    total_count = total_count + 1
    
    if abs(GT_rate - result_rate) < 0.05:
        good_count = good_count + 1
    if abs(GT_rate - result_rate) < 0.1:
        normal_count = normal_count + 1
    if abs(GT_rate - result_rate) < 0.15:
        no_bad_count = no_bad_count + 1

print("=====================================================")    
print(no_bad_count, "±15%")
print("ACC:", no_bad_count/total_count)
print("=====================================================")   
print(normal_count, "±10%")
print("ACC:", normal_count/total_count)
print("=====================================================")   
print(good_count, "±5%")
print("ACC:", good_count/total_count)
print("=====================================================")   
print(total_count, "total")
print("=====================================================")   
print("ACC:", no_pw_count/total_count)
print(no_pw_count, "no_pw_count")
print("=====================================================")   
   
