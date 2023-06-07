
"""
事项记录：
1. 在电脑端用模拟器运行target_platform='rv1126'， 比'rk1808'速度要慢将近一倍
2. rknn.config的参数设置非常非常重要，直接决定了模型精度。
3.

"""

import numpy as np
import cv2
from rknn.api import RKNN
import torchvision.models as models
import torch
import torch.nn.functional as F
import tensorflow as tf
import os
from tqdm import tqdm, trange
import logging
from PIL import Image
from utils.configs import CONFIGS
from utils.calc_diff import *


def export_pytorch_model():
    net = models.resnet18(pretrained=True)
    net.eval()
    trace_model = torch.jit.trace(net, torch.Tensor(1, 3, 224, 224))
    trace_model.save('./resnet18.pt')


def show_outputs(output):
    output_sorted = sorted(output, reverse=True)
    top5_str = '\n-----TOP 5-----\n'
    for i in range(5):
        value = output_sorted[i]
        index = np.where(output == value)
        for j in range(len(index)):
            if (i + j) >= 5:
                break
            if value > 0:
                topi = '{}: {}\n'.format(index[j], value)
            else:
                topi = '-1: 0.0\n'
            top5_str += topi
    print(top5_str)


def show_perfs(perfs):
    perfs = 'perfs: {}\n'.format(perfs)
    print(perfs)


def softmax(x):
    return np.exp(x)/sum(np.exp(x))

DEBUG = True
good_count = 0
normal_count = 0
no_bad_count = 0
total_count = 0
result_rate_list = []
GT_rate_list = []
normal_flag = True
not_normal_count = 0

if __name__ == '__main__':

    #export_pytorch_model()
    config_unet = CONFIGS()
    # 创建rknn推断后输出图像要保存的文件夹
    if not os.path.exists(config_unet.rknn_pred_mask_save_folder):
        os.makedirs(config_unet.rknn_pred_mask_save_folder)

    # Create RKNN object
    rknn = RKNN()

    ret = rknn.load_rknn(config_unet.load_rknn_path)
    if ret != 0:
        print('load' + config_unet.export_rknn_path + 'failed!')
        exit(ret)
    print('done')

    # init runtime environment
    print('--> Init runtime environment')
    ret = rknn.init_runtime(target="rv1126", device_id="c34f9d401a0607b9")
    if ret != 0:
        print('Init runtime environment failed')
        exit(ret)
    print('done')

    normal_flag = True
    for filename in tqdm(os.listdir(config_unet.test_images_path)):
        logging.info(f'\nPredicting image {filename} ...')
        test_image_path = os.path.join(config_unet.test_images_path, filename)
        # Set inputs
        # rbg和bgr的设置根据 reorder_channel='0 1 2' 进行修改
        # 图像大小要和rknn的输入相同
        img = cv2.imread(test_image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if img.shape[0] != config_unet.model_height or img.shape[2] != config_unet.model_width:
            #img = cv2.resize(img, (config_unet.model_width, config_unet.model_height), interpolation=cv2.INTER_CUBIC)
            img = cv2.resize(img, (config_unet.model_width, config_unet.model_height))


        test_mask_path = os.path.join(config_unet.test_masks_path, filename.split(".")[0] + ".png")
        mask = Image.open(test_mask_path)
        mask_np = np.array(mask)
        if DEBUG:
            mask_save = mask_np * 80
            cv2.imwrite(os.path.join(config_unet.pt_gt_mask_save_folder, filename), mask_save)

        # Inference
        print('--> Running model')
        # outputs = rknn.inference(inputs=[img], data_format='nchw', inputs_pass_through=True)#, inputs_pass_through=[1])
        outputs = rknn.inference(inputs=[img])

        # post process
        transpose_outputs = torch.Tensor(outputs[0][0])  # transpose_outputs
        softmax_confidence = F.softmax(transpose_outputs, dim=0)
        softmax_confidence_np = softmax_confidence.numpy()  # 转numpy可以在调试时查看矩阵数据排列，仅测试用
        rknn_pred_mask = F.one_hot(softmax_confidence.argmax(dim=0), config_unet.n_class).permute(2, 0, 1).numpy()
        pred = np.argmax(rknn_pred_mask, axis=0)
        # 保存图像用于查看
        pred_save = pred * 100
        cv2.imwrite(os.path.join(config_unet.rknn_pred_mask_save_folder, filename), pred_save)

        # calc the diff between rknn_pred and gt
        # 计算3种类别的差异
        normal_count, no_bad_count, total_count, result_rate, GT_rate, normal_flag \
            = calc_diff_three_label_between_pred_gt(pred, mask_np, normal_count, no_bad_count, total_count)

        # 像素值是正常的，返回计算厨余垃圾占比结果
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

    # show_outputs(softmax(np.array(outputs[0][0])))
    print('done')

    rknn.release()
