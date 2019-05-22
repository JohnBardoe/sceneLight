# TODO: -deal with the pan
#       -add logic for multiple people and false detections (hard af)
#       -add trackbars for color change
#       -study the SORT output because the rectangle is shit


import time
import numpy as np
import cv2
import os
import random
import darknet.darknet as dn
from utils.sort import *
import sacn
import threading

##constants##############
maxY = 400
minY = 50
minX = 0
maxX = 640
minPan = 175
maxPan = 150
minTilt = 100
maxTilt = 150

confidence = float(0.5)
nms_thresh = float(0.3)
#########################

pan = [100, 100, 100, 100]
tilt = [100, 100, 100, 100]
indexes = [-1, -1, -1, -1]
dmx_data = list()
lightSettings = list([255, 0, 0, 0, 200])
sender = sacn.sACNsender(bind_port=6553, bind_address="10.8.220.23")

if __name__ == '__main__':
    tracker = Sort()
    colors = [(random.randrange(0, 255), random.randrange(0, 255), random.randrange(0, 255)) for i in range(10)]


    dn.set_gpu(0)
    net = dn.load_net("/home/bardoe/sources/sceneLight/model/yolov3.cfg".encode("utf-8"),
                      "/home/bardoe/sources/sceneLight/model/yolov3.weights".encode("utf-8"), 0)
    meta = dn.load_meta("/home/bardoe/sources/sceneLight/model/coco.data".encode("utf-8"))
    cap = cv2.VideoCapture(0)

    frames = 0
    start = time.time()
    sender.start()
    sender.activate_output(1)
    sender[1].multicast = True

    while cap.isOpened():
        if frames > 100:
            frames = 0
            start = time.time()
        _, frame = cap.read()

        dmx_data = list([0] * 13)
        r = dn.detect_np(net, meta, frame, confidence, confidence, nms_thresh)
        toSort = list()

        for i in r:
            if i[0] == b'person':
                toSort.append((int(i[2][0]), int(i[2][1]), int(i[2][2]), int(i[2][3])))

        people = tracker.update(np.array(toSort))
        print("Ppl indexes: ", end="")
        for p in people:
            p = p.astype(np.int32)
            cv2.circle(frame, (int(p[0]), int(p[1])), 6, colors[p[-1] % 10], -1)
            cv2.rectangle(frame, (int(p[0])//2, int(p[1])//2), (int(p[0] + p[2])//2, int(p[1] + p[3])//2), colors[p[-1] % 10], 4)
            print(p[-1], end=" ")
        print("", end="|")

        cv2.imshow("frame", frame)
        print("FPS: {:5.2f}".format(frames / (time.time() - start)))
        for i in range(4):
            dmx_data.extend(list([pan[i], tilt[i]]))
            dmx_data.extend(lightSettings)
        sender[1].dmx_data = dmx_data
        if cv2.waitKey(1) & 0xFF == 27:
            break
            sender.stop()
        frames += 1