# 权重参数模型pth -> 带网络结构的模型pt
import torch.onnx
#import Unet_Mnet
from utils.configs import CONFIGS

config_unet = CONFIGS()
#net = Unet_Mnet.Unet_Mnet_model(n_classes=config_unet.n_class)

# print(net)
# checkpoint = torch.load(config_unet.pt_path, map_location='cpu')
# net.load_state_dict(checkpoint.state_dict())
# net.eval()
# print(net)
# print("torch_version:", torch.__version__)

net = torch.load(config_unet.pt_path, map_location='cpu')
net.eval()

# nchw
trace_model = torch.jit.trace(net, torch.Tensor(1, config_unet.model_channel, config_unet.model_height, config_unet.model_width))
#trace_model = torch.jit.trace(net, torch.Tensor(1, input_channel, input_width, input_height))

trace_model.save(config_unet.pt_jit_path)
print("转换完成.已生成" + config_unet.pt_jit_path)

