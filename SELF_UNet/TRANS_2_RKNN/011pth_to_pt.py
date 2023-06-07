# 权重参数模型pth -> 带网络结构的模型pt
import torch.onnx
import Unet_Mnet
from utils.configs import CONFIGS

config_unet = CONFIGS()
net = Unet_Mnet.Unet_Mnet_model(n_classes=3)
print(net)
checkpoint = torch.load(config_unet.pth_path, map_location='cpu')
net.load_state_dict(checkpoint)
net.eval()
print(net)
print("torch_version:", torch.__version__)

# nchw
trace_model = torch.jit.trace(net, torch.Tensor(1, config_unet.input_channel, config_unet.input_height, config_unet.input_width))
#trace_model = torch.jit.trace(net, torch.Tensor(1, input_channel, input_width, input_height))

trace_model.save(config_unet.pt_path)
print("转换完成.已生成" + config_unet.pt_path)

