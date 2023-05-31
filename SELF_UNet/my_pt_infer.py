import argparse
import logging
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from utils.data_loading import BasicDataset
from unet import UNet

import cv2


from unet.self_unet_backbone import My_Unet

from unet.unet_mbv2 import Unet_MobileNetV2


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
        return F.one_hot(full_mask.argmax(dim=0), net.n_classes).permute(2, 0, 1).numpy()


def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images')
    parser.add_argument('--model', '-m', default='MODEL.pth', metavar='FILE',
                        help='Specify the file in which the model is stored')
    parser.add_argument('--input', '-i', metavar='INPUT', nargs='+', help='Filenames of input images', required=True)
    parser.add_argument('--output', '-o', metavar='OUTPUT', nargs='+', help='Filenames of output images')
    parser.add_argument('--viz', '-v', action='store_true',
                        help='Visualize the images as they are processed')
    parser.add_argument('--no-save', '-n', action='store_true', help='Do not save the output masks')
    parser.add_argument('--mask-threshold', '-t', type=float, default=0.5,
                        help='Minimum probability value to consider a mask pixel white')
    parser.add_argument('--scale', '-s', type=float, default=0.5,
                        help='Scale factor for the input images')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')

    return parser.parse_args()


def get_output_filenames(args):
    def _generate_name(fn):
        return f'{os.path.splitext(fn)[0]}_OUT.png'

    return args.output or list(map(_generate_name, args.input))


def mask_to_image(mask: np.ndarray):
    if mask.ndim == 2:
        return Image.fromarray((mask * 255).astype(np.uint8))
    elif mask.ndim == 3:
        return Image.fromarray((np.argmax(mask, axis=0) * 255 / mask.shape[0]).astype(np.uint8))


if __name__ == '__main__':

    img_path="./data_zhanting_test/imgs"
    model_path="./data_zhanting_checkpoints/checkpoint_epoch5.pth"

    img_names = os.listdir(img_path)

    # 连背景是４个类别
    # net = UNet(n_channels=3, n_classes=4, bilinear=False)

    # net = My_Unet(num_class=4)

    net = Unet_MobileNetV2(num_classes=4)

    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')

    net.to(device=device)
    net.load_state_dict(torch.load(model_path, map_location=device))

    logging.info('Model loaded!')

    for i, filename in enumerate(img_names):

        save_infer_path = os.path.join("./data_zhanting_test/infer_result/",filename)

        logging.info(f'\nPredicting image {filename} ...')
        filename = os.path.join(img_path,filename)

        img = Image.open(filename)

        img_bgr = cv2.cvtColor(np.asarray(img),cv2.COLOR_RGB2BGR)

        mask = predict_img(net=net,
                           full_img=img,
                           scale_factor=1,
                           out_threshold=0.5,
                           device=device)


        result = mask_to_image(mask)

        result_bgr = cv2.cvtColor(np.asarray(result),cv2.COLOR_RGB2BGR)
        # plot_img_and_mask(img, mask)
        # pass

        result_bgr=np.concatenate((img_bgr,result_bgr),axis=1)
        # cv2.imwrite("1.png",result)
        cv2.imshow("result_bgr",result_bgr)
        cv2.imwrite(save_infer_path,result_bgr)
        cv2.waitKey(1)

        # result.save(out_filename)



        #
        # if not args.no_save:
        #     out_filename = out_files[i]
        #     result = mask_to_image(mask)
        #     result.save(out_filename)
        #     logging.info(f'Mask saved to {out_filename}')
        #
        # if args.viz:
        #     logging.info(f'Visualizing results for image {filename}, close to continue...')
        #     plot_img_and_mask(img, mask)
