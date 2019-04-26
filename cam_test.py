from __future__ import division
import time
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import cv2
from utils.util import *
from utils.darknet import Darknet
import pickle as pkl
import os
import random
from utils.sort import *

def prep_image(img, inp_dim):
    orig_im = img
    dim = orig_im.shape[1], orig_im.shape[0]
    img = cv2.resize(orig_im, (inp_dim, inp_dim))
    img_ = img[:, :, ::-1].transpose((2, 0, 1)).copy()
    img_ = torch.from_numpy(img_).float().div(255.0).unsqueeze(0)
    return img_, orig_im, dim

def toRect(el):
    c1 = tuple(el[1:3].int())
    c2 = tuple(el[3:5].int())
    return [int(c1[0]), int(c1[1]), int(c2[0]), int(c2[1])]

if __name__ == '__main__':

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    cfgfile = "model/yolov3.cfg"
    weightsfile = "model/yolov3.weights"
    num_classes = 80
    tracker = Sort()
    colors = [(random.randrange(0, 255), random.randrange(0, 255), random.randrange(0, 255)) for i in range(10)]

    confidence = float(0.5)
    nms_thresh = float(0.3)
    start = 0
    CUDA = torch.cuda.is_available()

    print("CUDA availability:", CUDA)

    num_classes = 80
    bbox_attrs = 5 + num_classes

    model = Darknet(cfgfile)
    model.load_weights(weightsfile)

    model.net_info["height"] = 320
    inp_dim = int(model.net_info["height"])

    model.cuda()
    model.eval()

    cap = cv2.VideoCapture(0)

    assert cap.isOpened()

    frames = 0
    start = time.time()

    while cap.isOpened():

        ret, frame = cap.read()
        if not ret:
            print("video netu :c")
            continue

        img, orig_im, dim = prep_image(frame, inp_dim)
        im_dim = torch.FloatTensor(dim).cuda().repeat(1, 2)
        im_dim = im_dim.cuda()
        img = img.cuda()

        output = model(Variable(img), CUDA).cuda()
        output = write_results(output, confidence, num_classes, nms=True, nms_conf=nms_thresh)

        if type(output) == int:
            frames += 1
            print("FPS: {:5.2f}".format(frames / (time.time() - start)))
            cv2.imshow("frame", orig_im)
            key = cv2.waitKey(1)
            if key & 0xFF == ord('q'):
                break
            continue

        output[:, 1:5] = torch.clamp(output[:, 1:5], 0.0, float(inp_dim)) / inp_dim

        im_dim = im_dim.repeat(output.size(0), 1)
        output[:, [1, 3]] *= frame.shape[1]
        output[:, [2, 4]] *= frame.shape[0]

        tosort = list()

        for det in output:
            if det[-1] == 0:
                tosort.append(toRect(det))
        people = tracker.update(np.array(tosort))
        print("Ppl indexes: ", end="")
        for p in people:
            p = p.astype(np.int32)
            cv2.rectangle(orig_im, (p[0], p[1]), (p[2], p[3]), colors[p[-1] % 10], 4)
            print(p[-1], end=" ")
        print("", end="|")

        cv2.imshow("frame", orig_im)

        if cv2.waitKey(1) & 0xFF == 27:
            break
        frames += 1
        print("FPS: {:5.2f}".format(frames / (time.time() - start)))