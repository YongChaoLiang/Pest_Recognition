"""
coding: utf-8
author: Lu Shiliang
date:2023-09
"""

import argparse
import json
import sys

import cv2
import os

import numpy
import numpy as np
import torch

from pytorch_grad_cam import GradCAM, \
    ScoreCAM, \
    GradCAMPlusPlus, \
    AblationCAM, \
    XGradCAM, \
    EigenCAM, \
    EigenGradCAM, \
    LayerCAM, \
    FullGrad

from pytorch_grad_cam.utils.image import show_cam_on_image, \
    preprocess_image
from pytorch_grad_cam.ablation_layer import AblationLayerVit
import BiFormer.models.biformer as biformer
import PIL.Image as img

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use-cuda', action='store_true', default=True,
                        help='Use NVIDIA GPU acceleration')
    parser.add_argument(
        '--image-path',
        type=str,
        default='E:\\Code\\Pytorch_gpu\\Data\\pest\\classification\\test\\20\\13610.jpg',
        help='Input image path')
    parser.add_argument('--aug_smooth', action='store_true',
                        help='Apply test time augmentation to smooth the CAM')
    parser.add_argument(        '--eigen_smooth',
        action='store_true',
        help='Reduce noise by taking the first principle componenet'
        'of cam_weights*activations')

    parser.add_argument(
        '--method',
        type=str,
        default='gradcam',
        help='Can be gradcam/gradcam++/scorecam/xgradcam/ablationcam')

    args = parser.parse_args()
    args.use_cuda = args.use_cuda and torch.cuda.is_available()
    if args.use_cuda:
        print('Using GPU for acceleration')
    else:
        print('Using CPU for computation')

    return args


def reshape_transform(tensor, height=7, width=7):
    # print(tensor.size())

    result = tensor.reshape(tensor.size(0),
                            height, width, 512)
    # Bring the channels to the first dimension,
    result = result.transpose(2, 3).transpose(1, 2)
    return result


if __name__ == '__main__':
    """ python swinT_example.py -image-path <path_to_image>
    Example usage of using cam-methods on a SwinTransformers network.

    """
    args = get_args()



    methods = \
        {"gradcam": GradCAM,
         "scorecam": ScoreCAM,
         "gradcam++": GradCAMPlusPlus,
         "ablationcam": AblationCAM,
         "xgradcam": XGradCAM,
         "eigencam": EigenCAM,
         "eigengradcam": EigenGradCAM,
         "layercam": LayerCAM,
         "fullgrad": FullGrad}

    if args.method not in list(methods.keys()):
        raise Exception(f"method should be one of {list(methods.keys())}")

    model = biformer.biformer_tiny(pretrained=False, nm_class=102)
    weight_dict = torch.load('../Output/Real/best.pth')
    model.load_state_dict(weight_dict["model"], strict=True)

    if args.use_cuda:
        model = model.cuda()

    target_layers = [model.stages[3][1].norm2]

    if args.method not in methods:
        raise Exception(f"Method {args.method} not implemented")

    if args.method == "ablationcam":
        cam = methods[args.method](model=model,
                                   target_layers=target_layers,
                                   use_cuda=args.use_cuda,
                                   reshape_transform=reshape_transform,
                                   ablation_layer=AblationLayerVit())
    else:
        cam = methods[args.method](model=model,
                                   target_layers=target_layers,
                                   use_cuda=args.use_cuda,
                                   reshape_transform=reshape_transform)

    rgb_img = cv2.imread(args.image_path, 1)[:, :, ::-1]  # 通道变换转PIL同款

    rgb_img = cv2.resize(rgb_img, (224, 224))
    rgb_img = np.float32(rgb_img) / 255
    input_tensor = preprocess_image(rgb_img, mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])

    cam.batch_size = 32

    grayscale_cam = cam(input_tensor=input_tensor,
                        targets=None,
                        eigen_smooth=args.eigen_smooth,
                        aug_smooth=args.aug_smooth)


    grayscale_cam = grayscale_cam[0, :]

    img_name, ext = os.path.splitext(args.image_path)
    True_label = img_name.split("\\")[-2]
    model.cuda()
    model.eval()

    json_path = 'class_indices.json'
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)
    with open(json_path, "r") as f:
        class_indict = json.load(f)


    out_put = model(input_tensor.cuda())
    _, pred_label = out_put.topk(k=1, largest=True, sorted=True)
    pred_label = class_indict[str(int(pred_label.cpu()))]

    result = False
    print(pred_label, True_label)

    if pred_label == True_label:
        result = True

    img_name = img_name.split("\\")[-1]

    cam_image = show_cam_on_image(rgb_img, grayscale_cam)
    cv2.imwrite(f'./figure/{img_name}_{True_label}=={result}.jpg', cam_image)
    print('Finished.')
    sys.exit()
