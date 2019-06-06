# TODO: -add trackbars for color change
import time
import numpy as np
import cv2
import os
import random
import darknet.darknet as dn
from utils.sort import *
import sacn
import threading
import head
import math

##constants##############
maxY = 400
minY = 50
minX = 0
maxX = 640
minPan = 175
maxPan = 150
minTilt = 100
maxTilt = 150
shutdownTime = 2
switchOlder = False

confidence = float(0.5)
nms_thresh = float(0.3)
#########################

heads = [head.Head() for count in range(4)]
dmx_data = list()
lightSettingsOn = list([255, 0, 0, 0, 200])
lightSettingsOff = list([255, 0, 0, 0, 0])
sender = sacn.sACNsender(bind_port=6553, bind_address="10.8.220.23")


def findList(list, obj):
    counter = 0
    for i in list:
        if i == obj:
            return counter
        counter += 1
    return -1


def assignHd2Ps(index, x, y):
    global heads
    if x > maxX or x < minX or y > minY or y > maxY:
        return

    for i in range(len(heads)):
        if heads[i].index == index:
            heads[i].state = 1
            setHead(i, x, y)
            return
    for i in range(len(heads)):
        if heads[i].state == 0:
            heads[i].index = index
            heads[i].time = time.time()
            heads[i].state = 1
            setHead(i, x, y)
            return
    if switchOlder:
        oldestTime = time.time()
        bestIndex = 0
        for i in range(len(heads)):
            if heads[i].state == 1 or heads[i].state == 2:
                if heads[i].time < oldestTime:
                    oldestTime = heads[i].time
                    bestIndex = i
        heads[bestIndex].index = index
        heads[bestIndex].time = time.time()
        heads[bestIndex].state = 1
        setHead(bestIndex, x, y)


def setHead(index, x, y):
    global heads
    heads[index].pan = int(minPan - (minPan - maxPan) * (x / (minX + maxX)))
    heads[index].tilt = int(maxTilt + (maxTilt - minTilt) * (y / (minY + maxY)))


def updateHeads(indices):
    global heads
    for i in heads:
        if i.state == 2 and (time.time() - i.time) > shutdownTime:
            i.state = 0
            i.time = 0
            i.index = -1
        elif i.state == 1:
            if findList(indices, i.index) == -1:
                i.state = 2
                i.time = time.time()


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
        indices = list()
        for p in people:
            p = p.astype(np.int32)
            print(p[-1], end=" ")
            cv2.circle(frame, (int(p[0]), int(p[1])), 6, colors[p[-1] % 10], -1)
            cv2.rectangle(frame, (int(p[0]) // 2, int(p[1]) // 2), (int(p[0] + p[2]) // 2, int(p[1] + p[3]) // 2),
                          colors[p[-1] % 10], 4)
            assignHd2Ps(p[-1], int(p[0]), int(p[1]))
            indices.append(p[-1])
        updateHeads(indices)

        print("", end="|")
        cv2.imshow("frame", frame)
        print("FPS: {:5.2f}".format(frames / (time.time() - start)))
        for i in heads:
            dmx_data.extend(list([i.pan, i.tilt]))
            if i.state == 0:
                dmx_data.extend(lightSettingsOff)
            else:
                dmx_data.extend(lightSettingsOn)
        sender[1].dmx_data = dmx_data
        if cv2.waitKey(1) & 0xFF == 27:
            break
            sender.stop()
            exit()
        frames += 1
