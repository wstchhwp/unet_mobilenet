from http.server import HTTPServer, BaseHTTPRequestHandler
import json
import ssl

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from utils.data_loading import BasicDataset
import Unet_Mnet

import requests as req
from PIL import Image, ImageDraw
from io import BytesIO
import cv2


# os.environ['CUDA_VISIBLE_DEVICES'] = "0"


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


if __name__ == '__main__':

    overflow_classes = 2
    model_pth_path = "./20220725/checkpoint_epoch_117.pth"

    net = Unet_Mnet.Unet_Mnet_model(n_classes=overflow_classes)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # logging.info(f'Loading model {args.model}')
    # logging.info(f'Using device {device}')
    net.to(device=device)
    net.load_state_dict(torch.load(model_pth_path, map_location=device).state_dict())
    net.eval()

    debug = True
    create_all_zero_img_flag = True
    good_count = 0
    normal_count = 0
    no_bad_count = 0
    total_count = 0
    result_rate_list = []
    GT_rate_list = []
    roi_width = 512
    roi_height = 512

    # 全局的数据
    data = {'result': 'None !'}  # overflow
    host = ('localhost', 1212)


    #  Resquest实现GET 和 POST
    class Resquest(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(data).encode())

        def do_POST(self):

            req_datas = self.rfile.read(int(self.headers['content-length'])).decode()
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            req_datas = json.loads(req_datas)
            img_src = req_datas["image_oss_path"]
            uuid = req_datas["uuid"]
            #         response = req.get(img_src)

            error = "no error"
            xmin = 0
            xmax = 1920
            ymin = 0
            ymax = 1080

            try:
                # 获取平台对原始图像1920*1080画的垃圾桶的坐标
                req_xmin = int(req_datas["xmin"])
                req_xmax = int(req_datas["xmax"])
                req_ymin = int(req_datas["ymin"])
                req_ymax = int(req_datas["ymax"])

                # 获取平台对原始图像1920*1080画的满溢界线的坐标
                req_xmin_overflow_limit = int(req_datas["xmin_overflow_limit"])
                req_xmax_overflow_limit = int(req_datas["xmax_overflow_limit"])
                req_ymin_overflow_limit = int(req_datas["ymin_overflow_limit"])
                req_ymax_overflow_limit = int(req_datas["ymax_overflow_limit"])

                if req_xmin >= 0 and req_xmin <= req_xmax and req_xmax <= 1920:
                    if req_ymin >= 0 and req_ymin <= req_ymax and req_ymax <= 1080:
                        xmin: int = req_xmin
                        xmax: int = req_xmax
                        ymin: int = req_ymin
                        ymax: int = req_ymax

                if req_xmin_overflow_limit >= 0 and req_xmin_overflow_limit <= req_xmax_overflow_limit and req_xmax_overflow_limit <= 1920:
                    if req_ymin_overflow_limit >= 0 and req_ymin_overflow_limit <= req_ymax_overflow_limit and req_ymax_overflow_limit <= 1080:

                        xmin_overflow_limit: int = req_xmin_overflow_limit
                        xmax_overflow_limit: int = req_xmax_overflow_limit
                        ymin_overflow_limit: int = req_ymin_overflow_limit
                        ymax_overflow_limit: int = req_ymax_overflow_limit

                        # 获取图像crop之后，获取的满溢界线的4个坐标
                        xmin_crop_overflow_limit: int = int(xmin_overflow_limit - xmin) if xmin_overflow_limit >= xmin else 0
                        ymin_crop_overflow_limit: int = int(ymin_overflow_limit - ymin) if ymin_overflow_limit >= ymin else 0
                        xmax_crop_overflow_limit: int = int(xmax_overflow_limit - xmin) if xmin < xmax_overflow_limit < xmax else 0
                        ymax_crop_overflow_limit: int = int(ymax_overflow_limit - ymin) if ymin < ymax_overflow_limit < ymax else 0

                        # crop图像resize到512， 对满溢界线也做相应的变化
                        xmin_crop_overflow_limit = (xmin_crop_overflow_limit * roi_width)  // (xmax - xmin)
                        ymin_crop_overflow_limit = (ymin_crop_overflow_limit * roi_height) // (ymax - ymin)
                        xmax_crop_overflow_limit = (xmax_crop_overflow_limit * roi_width)  // (xmax - xmin)
                        ymax_crop_overflow_limit = (ymax_crop_overflow_limit * roi_height) // (ymax - ymin)

                        # 计算满溢界线对应的面积
                        overflow_rect_area = (xmax_crop_overflow_limit - xmin_crop_overflow_limit) * (ymax_crop_overflow_limit - ymin_crop_overflow_limit)
                        print("overflow_rect_area：", overflow_rect_area)

                        # 创建全1图像，把满溢界线区域置0
                        # global create_all_zero_img_flag
                        # if True == create_all_zero_img_flag:
                        #     img_ones = np.ones((roi_width, roi_height), np.uint8)
                        #     img_ones[ymin_crop_overflow_limit:ymax_crop_overflow_limit, xmin_crop_overflow_limit:xmax_crop_overflow_limit] = 0
                        #     create_all_zero_img_flag = False
                        img_ones = np.ones((roi_width, roi_height), np.uint8)
                        img_ones[ymin_crop_overflow_limit:ymax_crop_overflow_limit, xmin_crop_overflow_limit:xmax_crop_overflow_limit] = 0

                    else:
                        print("error region1")
                        error = "error region"
                else:
                    print("error region2")
                    error = "error region"
            except:
                print("no error")
                error = "no error"

            width = 0
            height = 0
            image = None
            try:
                response = req.get(img_src)
                image = Image.open(BytesIO(response.content))
                width, height = image.size  # resize((width, height))
            except:
                print("image is None")
                image = None
                data = {'result': "error", "uuid": uuid, "error": "error image path"}
            #             self.wfile.write(json.dumps(data).encode('utf-8'))
            if image is None:
                data = {'result': "error", "uuid": uuid, "error": "no image"}
            elif (width != 1920 or height != 1080):
                data = {'result': "error", "uuid": uuid, "error": "error image size"}
            else:
                roi_image = image.crop(box=(xmin, ymin, xmax, ymax))
                roi_image.save("4.jpg")
                roi_image = roi_image.resize((roi_width, roi_height), Image.NEAREST)
                pt_pred_mask = predict_img(net=net, full_img=roi_image, scale_factor=1, out_threshold=0.6,
                                           device=device)
                pred = np.argmax(pt_pred_mask, axis=0)  # 取出最大值对应的索引，即类别
                pred = pred.astype(np.uint8)
                if debug:
                    pred_save = pred.copy()

                if net.n_classes > 2:
                    pred[pred != 0] = 1

                # overflow_roi_pix_sum = sum(map(sum, cv2.bitwise_and(img_ones, pred)))  # map(fund, a) equals [func(i) for i in a]  and return a list
                overflow_roi_pix_sum = np.sum(cv2.bitwise_and(img_ones, pred))
                overflow_roi_pix_rate = overflow_roi_pix_sum / overflow_rect_area if 0 != overflow_rect_area else 0.0
                print("\n满溢比率：", overflow_roi_pix_rate)

                if debug:
                    # 保存图像，用于观察
                    cv2.rectangle(pred_save, (xmin, ymin), (xmax, ymax), 255)
                    cv2.rectangle(pred_save, (xmin_crop_overflow_limit, ymin_crop_overflow_limit), (xmax_crop_overflow_limit, ymax_crop_overflow_limit), 255)
                    image1 = ImageDraw.Draw(image)
                    roi_image1 = ImageDraw.Draw(roi_image)
                    image1.rectangle([(xmin, ymin), (xmax, ymax)], outline="red", width=5)    # 红色是垃圾桶框
                    image1.rectangle([(req_xmin_overflow_limit, req_ymin_overflow_limit), (req_xmax_overflow_limit, req_ymax_overflow_limit)], outline="green", width=5)  # 绿色是满溢界线
                    roi_image1.rectangle([(xmin_crop_overflow_limit, ymin_crop_overflow_limit), (xmax_crop_overflow_limit, ymax_crop_overflow_limit)], outline="blue", width=5)
                    #cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0,0,255))  # 红
                    #cv2.rectangle(image, (xmin_crop_overflow_limit, ymin_crop_overflow_limit), (xmax_crop_overflow_limit, ymax_crop_overflow_limit), (255,0,0))  # 蓝
                    if overflow_roi_pix_rate >= 0.1:
                        cv2.putText(pred_save, "overflow", (xmin_crop_overflow_limit, ymin_crop_overflow_limit),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, 255, 1)
                        image1.text((req_xmin_overflow_limit, req_ymin_overflow_limit), 'overflow', fill = (255, 0 ,0), width=5)
                    pred_save = pred_save * 100
                    cv2.imwrite("1.jpg", pred_save)
                    image.save("2.jpg")
                    roi_image.save("3.jpg")

                data = {'result': overflow_roi_pix_rate, "uuid": uuid, "error": error, "area": str(overflow_rect_area)}

            self.wfile.write(json.dumps(data).encode('utf-8'))


    #  开启服务器
    server = HTTPServer(host, Resquest)
    print("Starting server, listen at: %s:%s" % host)
    server.serve_forever()