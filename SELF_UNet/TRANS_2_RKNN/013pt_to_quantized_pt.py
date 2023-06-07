"""
注意事項：
  1. 一定要用torch1.9版本， torch1.6版本不能保存量化后的网络模型，只能保存量化后的权重参数
  2.

"""

import torch.onnx
import Unet_Mnet_Quant
from utils.configs import CONFIGS
from torchvision import transforms
from utils.data_loading import BasicDataset
import torch
import torch.nn.functional as F
import cv2 as cv
import numpy as np
from PIL import Image

def predict_img_quantize(net,
                         device,
                         scale_factor=1,
                         out_threshold=0.5):
    net.eval()
    img = Image.open("2.jpg")
    width, height = img.size  # 获得原图像的宽和高, 手动破袋的图像宽高是512*512
    size = (width, height)
    full_img = img.resize(size, resample=0)
    img = torch.from_numpy(BasicDataset.preprocess(full_img, scale_factor, is_mask=False))
    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(img)
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

    # todo: 22.7.6注释测试
    # if net.n_classes == 1:
    #     return (full_mask > out_threshold).numpy()
    # else:
    #     return F.one_hot(full_mask.argmax(dim=0), net.n_classes).permute(2, 0, 1).numpy()
    return F.one_hot(full_mask.argmax(dim=0), 3).permute(2, 0, 1).numpy()


#"""
# 生成的模型是4.4M
# 权重参数模型pth -> 带网络结构的模型pt
config_unet = CONFIGS()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net_state_dict = torch.load(config_unet.pt_path, map_location='cpu')  # map_location='cpu'
model_fp32 = Unet_Mnet_Quant.Unet_Mnet_model(n_classes=3)
model_fp32.load_state_dict(net_state_dict.state_dict())
model_fp32.float()
# model must be set to eval mode for static quantization logic to work
model_fp32.eval()
# Specify quantization configuration
# 部署在x86 server上,使用 'fbgemm' ; 部署在ARM上,使用 'qnnpack'
model_fp32.qconfig = torch.quantization.get_default_qconfig('qnnpack')
#print(model_fp32.qconfig)
model_fp32_prepared = torch.quantization.prepare(model_fp32, inplace=True)

# 验证量化准备前的模型
pt_pred_prequantize_mask = predict_img_quantize(net=model_fp32_prepared, scale_factor=0.5, out_threshold=0.6, device=device)
pred = np.argmax(pt_pred_prequantize_mask, axis=0)
pred_save = pred * 100
cv.imwrite("2_mask1.jpg", pred_save)

model_int8 = torch.quantization.convert(model_fp32_prepared, inplace=True)
# 验证量化后的模型
pt_pred_afterquantize_mask = predict_img_quantize(net=model_int8, scale_factor=0.5, out_threshold=0.6, device=device)
pred = np.argmax(pt_pred_afterquantize_mask, axis=0)
pred_save = pred * 100
cv.imwrite("2_mask2.jpg", pred_save)

# torch.save(model_int8, config_unet.pt_jit_path)  #model_int8.state_dict()

input = torch.ones(1, config_unet.input_channel, config_unet.input_height, config_unet.input_width)
traced_model = torch.jit.trace(model_int8, input)
torch.jit.save(traced_model, config_unet.pt_jit_path)
print("转换完成.已生成" + config_unet.pt_jit_path)
#"""



"""
# 參考宋哥發來的網址，寫的代碼， 生成的模型是4.4M   
# 权重参数模型pth -> 带网络结构的模型pt

config_unet = CONFIGS()
net = Unet_Mnet.Unet_Mnet_model(n_classes=3)
#net.train()
#net.fuse_model()
print(net)
checkpoint = torch.load(config_unet.pt_path, map_location='cpu')  # map_location='cpu'
net.load_state_dict(checkpoint.state_dict())
# model must be set to eval mode for static quantization logic to work
net.eval()
# Specify quantization configuration
# 部署在x86 server上,使用 'fbgemm' ; 部署在在ARM上,使用 'qnnpack'
net.qconfig = torch.quantization.get_default_qat_qconfig('qnnpack')
print(net.qconfig)
net = torch.quantization.prepare_qat(net, inplace=True)
# No tuning, just straight convert with default parameters

net = torch.quantization.convert(net, inplace=True)
input = torch.ones(1, config_unet.input_channel, config_unet.input_height, config_unet.input_width)
traced_model = torch.jit.trace(net, input)
torch.jit.save(traced_model, config_unet.pt_jit_path)
print("转换完成.已生成" + config_unet.pt_jit_path)
"""


"""
# TODO:生成的模型是4.7M，直接加载运行jit_pt模型会报错，转rknn模型也会报错。
# 权重参数模型pth -> 带网络结构的模型pt

config_unet = CONFIGS()
net = Unet_Mnet.Unet_Mnet_model(n_classes=3)
print(net)
print("================================================")

checkpoint = torch.load(config_unet.pt_path, map_location='cpu')  # cpu
net.load_state_dict(checkpoint.state_dict())
net.eval()
# Specify quantization configuration
# 部署在x86 server上,使用 'fbgemm' ; 部署在在ARM上,使用 'qnnpack'
backend = "fbgemm"  # "cpu"
net.qconfig = torch.quantization.get_default_qconfig(backend)  # get_default_qat_qconfig
model_static_prepared = torch.quantization.prepare(net, inplace=False)
model_static_quantized = torch.quantization.convert(model_static_prepared, inplace=False)
model_static_quantized.to("cpu")

print(model_static_quantized)
print("================================================")
print("torch_version:", torch.__version__)

# nchw
# torch.ones(1, config_unet.input_channel, config_unet.input_height, config_unet.input_width)
input_size = torch.Tensor(1, config_unet.input_channel, config_unet.input_height, config_unet.input_width)
trace_model = torch.jit.trace(model_static_quantized, input_size)
#trace_model = torch.jit.trace(net, torch.Tensor(1, input_channel, input_width, input_height))

trace_model.save(config_unet.pt_jit_path)
print("转换完成.已生成" + config_unet.pt_jit_path)
"""