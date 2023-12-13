# import torch
# from mmcv import Config
from mmdet.apis import init_detector, inference_detector
import numpy as np
import cv2

import argparse
import os
import os.path as osp
import time
import warnings

import mmcv
import torch
from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)
from mmdet.apis import multi_gpu_test, single_gpu_test
from mmdet.datasets import (build_dataloader, build_dataset,
                            replace_ImageToTensor)
from mmdet.models import build_detector
from mmdet.datasets.pipelines import Compose

import mmdet_custom  # noqa: F401,F403
import mmcv_custom  # noqa: F401,F403

from pytorch_grad_cam import GradCAM, EigenCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from PIL import Image

import mmcv
import matplotlib.pyplot as plt
import torch
from mmcv.runner import load_checkpoint
from mmdet.apis import init_detector
from mmdet.models import build_detector
from torch.autograd import Variable
import torchvision
from mmcv.parallel import collate, scatter


class_names = ['carina', 'ett', 'clavicle']
COLORS = np.random.uniform(0, 255, size=(len(class_names), 3))

def predict(data, model):
    outputs = model(return_loss=False, rescale=True, **data)

    boxes, classes, labels, indices = [], [], [], []

    for i in range(len(class_names)):
        boxes.append(outputs[0][i][0][:-1])
        classes.append(class_names[i])
        labels.append(i)

    boxes = np.int32(boxes)

    return boxes, classes, labels

def draw_boxes(boxes, labels, classes, image):
    for i, box in enumerate(boxes):
        color = COLORS[labels[i]]
        cv2.rectangle(
            image,
            (int(box[0]), int(box[1])),
            (int(box[2]), int(box[3])),
            color, 2
        )
        cv2.putText(image, classes[i], (int(box[0]), int(box[1] - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2,
                    lineType=cv2.LINE_AA)
    return image

def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet plot grad cam')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('image', help='image file path')

    args = parser.parse_args()

    return args

def fasterrcnn_reshape_transform(x):
    activations = []
    target_size = 80
    for i in range(len(x)):
        activations.append(torch.nn.functional.interpolate(torch.abs(x[i]), target_size, mode='bilinear'))
    activations = torch.cat(activations, axis=1)
    return activations

class FasterRCNNBoxScoreTarget:
    """ For every original detected bounding box specified in "bounding boxes",
    	assign a score on how the current bounding boxes match it,
    		1. In IOU
    		2. In the classification score.
    	If there is not a large enough overlap, or the category changed,
    	assign a score of 0.

    	The total score is the sum of all the box scores.
    """

    def __init__(self, labels, bounding_boxes, iou_threshold=0.5):
        self.labels = labels
        self.bounding_boxes = bounding_boxes
        self.iou_threshold = iou_threshold

    def __call__(self, model_outputs):
        output = torch.Tensor([0])
        if torch.cuda.is_available():
            output = output.cuda()

        if len(model_outputs["boxes"]) == 0:
            return output

        for box, label in zip(self.bounding_boxes, self.labels):
            box = torch.Tensor(box[None, :])
            if torch.cuda.is_available():
                box = box.cuda()

            ious = torchvision.ops.box_iou(box, model_outputs["boxes"])
            index = ious.argmax()
            if ious[0, index] > self.iou_threshold and model_outputs["labels"][index] == label:
                score = ious[0, index] + model_outputs["scores"][index]
                output = output + score
        return output

def main():
    args = parse_args()
    device = torch.device('cuda')
    # Load the config and pre-trained model
    cfg = Config.fromfile(args.config)
    model = init_detector(cfg, args.checkpoint)

    image = cv2.imread(args.image)
    breakpoint()
    img_norm_cfg = dict(
        mean=[126.55846604, 126.55846604, 126.55846604], std=[55.47551373, 55.47551373, 55.47551373], to_rgb=True)
    gradcam_pipeline = [
        dict(type='LoadImageFromFile'),
        dict(
            type='MultiScaleFlipAug',
            img_scale=(1280, 1280),
            flip=False,
            transforms=[
                dict(type='Resize', keep_ratio=True),
                dict(type='Normalize', **img_norm_cfg),
                dict(type='Pad', size_divisor=32),
                dict(type='ImageToTensor', keys=['img']),
                dict(type='Collect', keys=['img']),
            ])
    ]

    gradcam_pipeline = Compose(gradcam_pipeline)
    data = dict(img_prefix='test_img', img=image)
    img_info = dict(
        filename=args.image,  # The filename of the image
        height=data['img'].shape[1],  # Height of the image after resizing
        width=data['img'].shape[2],  # Width of the image after resizing
        scale_factor=1.0,  # Scale factor used during resizing
    )
    data['img_info'] = img_info
    data = gradcam_pipeline(data)

    data = scatter(collate([data], samples_per_gpu=1), [device])[0]

    # results = model(return_loss=False, rescale=True, **data)

    boxes, classes, labels = predict(data, model)
    image = draw_boxes(boxes, labels, classes, image)
    img = Image.fromarray(image)
    img.save("test.jpg")

    # breakpoint()
    cam = EigenCAM(model,
              target_layers=model.neck.fpn_convs,
              use_cuda=True,
            #   reshape_transform=fasterrcnn_reshape_transform
              )

    # breakpoint()
    targets = [FasterRCNNBoxScoreTarget(labels=labels, bounding_boxes=boxes)]
    grayscale_cam = cam(data,
                        targets=targets
                        )


    grayscale_cam = grayscale_cam[0, :]
    cam_image = show_cam_on_image(image_float_np, grayscale_cam, use_rgb=True)
    # # And lets draw the boxes again:
    image_with_bounding_boxes = draw_boxes(boxes, labels, classes, cam_image)
    img = Image.fromarray(image_with_bounding_boxes)
    img.save('test_gradcam.jpg')

if __name__=='__main__':
    main()