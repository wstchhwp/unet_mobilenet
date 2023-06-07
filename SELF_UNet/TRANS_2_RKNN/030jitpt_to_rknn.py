
"""
事项记录：
1. 在电脑端用模拟器运行target_platform='rv1126'， 比'rk1808'速度要慢将近一倍
2. rknn.config的参数设置非常非常重要，直接决定了模型精度。
3.

"""
import tensorflow
import numpy as np
import cv2
from rknn.api import RKNN
import torchvision.models as models
import torch
import torch.nn.functional as F
import os
from tqdm import tqdm, trange
import logging
from utils.configs import CONFIGS


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

if __name__ == '__main__':

    #export_pytorch_model()
    config_unet = CONFIGS()

    print("NOTE:=================")
    print("    To run this demo, it's recommanded to use PyTorch1.6.0 and Torchvision0.7.0")
    print("NOTE:=================")
    # Create RKNN object
    rknn = RKNN(verbose=False, verbose_file='./verbose_log.txt')

    # pre-process config
    print('--> config model')
    #rknn.config(mean_values=[[0, 0, 0]], std_values=[[1, 1, 1]], reorder_channel='0 1 2', quantized_dtype='dynamic_fixed_point-i8')#, quantized_algorithm='mmse')
    #rknn.config(channel_mean_value='123.675 116.28 103.53 58.395', reorder_channel='0 1 2')
    ## pytorch非量化转rknn配置
    rknn.config(  # 正式项目用的是该config配置
        reorder_channel='0 1 2',   # RBG
        mean_values=[[0, 0, 0]],
        std_values=[[255, 255, 255]],
        optimization_level=3,
        target_platform='rv1126',
        quantize_input_node=False,
        # output_optimize=1,
        # force_builtin_perm=False,
        )

    # # pytorch量化后转rknn配置
    # rknn.config(
    #     quantize_input_node=True,
    #     merge_dequant_layer_and_output_node=True,
    #     )

    # Load pytorch model
    print('--> Loading model')
    ret = rknn.load_pytorch(model=config_unet.pt_jit_path, input_size_list=config_unet.model_size_list)  
    if ret != 0:
        print('Load pytorch model failed!')
        exit(ret)
    print('done')

    # Build model
    print('--> Building model')
    # 加上dataset获取量化校准表，利用一些batch的图像数据集，做图像量化，生成量化表
    ret = rknn.build(do_quantization=config_unet.quantization_on, dataset=config_unet.quantization_dataset, pre_compile=config_unet.pre_compile)  # pre_compile=True,那么模型就只能在rk板子上运行
    if ret != 0:
        print('Build pytorch failed!')
        exit(ret)
    print('done')

    # Export rknn model
    print('--> Export RKNN model')
    ret = rknn.export_rknn(config_unet.export_rknn_path)
    if ret != 0:
        print('Export' + config_unet.export_rknn_path + 'failed!')
        exit(ret)
    print('done')

    # Load rknn model
    ret = rknn.load_rknn(config_unet.load_rknn_path)
    if ret != 0:
        print('load' + config_unet.load_rknn_path + 'failed!')
        exit(ret)
    print('done')

    # init runtime environment
    print('--> Init runtime environment')
    ret = rknn.init_runtime(target="rv1126", device_id="c34f9d401a0607b9")  # todo:写到config中
    if ret != 0:
        print('Init runtime environment failed')
        exit(ret)
    print('done')

    # for filename in tqdm(os.listdir(test_images_path)):
    #     logging.info(f'\nPredicting image {filename} ...')
    #     test_image_path = test_images_path + filename
    #     # Set inputs
    #     # rbg和bgr的设置根据 reorder_channel='0 1 2' 进行修改
    #     # 图像大小要和rknn的输入相同
    #     img = cv2.imread(test_image_path)
    #     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #     img = cv2.resize(img, (256, 256))
    #
    #     # Inference
    #     print('--> Running model, rknn.inference')
    #     # outputs = rknn.inference(inputs=[img], data_format='nchw', inputs_pass_through=True)#, inputs_pass_through=[1])
    #     outputs = rknn.inference(inputs=[img])
    #
    #     transpose_outputs = torch.Tensor(outputs[0][0])  # transpose_outputs
    #     softmax_confidence = F.softmax(transpose_outputs, dim=0)
    #     softmax_confidence_np = softmax_confidence.numpy()  # 转numpy可以在调试时查看矩阵数据排列，仅测试用
    #     # softmax_confidence_tmp = softmax(np.array(transpose_outputs))
    #     # softmax_confidence = multi_softmax(np.array(transpose_outputs))   # [256,480,1,3]
    #
    #     pred_mask = F.one_hot(softmax_confidence.argmax(dim=0), 3).permute(2, 0, 1).numpy()
    #     pred = (np.argmax(pred_mask, axis=0)) * 100
    #     cv2.imwrite("./py_rknn_rknn_predMask/"+filename, pred)
    #
    # # show_outputs(softmax(np.array(outputs[0][0])))
    # print('done')

    # # perf
    # print('--> Begin evaluate model performance')
    # perf_results = rknn.eval_perf(inputs=[img])
    # print('done')

    rknn.release()
