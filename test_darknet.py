import darknet.darknet as dn
import sys, os
import pdb
import time
import cv2 as cv

dn.set_gpu(0)
net = dn.load_net("/home/bardoe/sources/sceneLight/model/yolov3.cfg".encode("utf-8"), "/home/bardoe/sources/sceneLight/model/yolov3.weights".encode("utf-8"), 0)
meta = dn.load_meta("/home/bardoe/sources/sceneLight/model/coco.data".encode("utf-8"))
frames = 0
start = time.time()
cap = cv.VideoCapture(0)
while True:
    if(frames>100000000):
        frames=0
        start=time.time()

    _, frame = cap.read()
    r = dn.detect_np(net, meta, frame)
    print(r)
    frames += 1
    cv.imshow("frame", frame)
    print("FPS: {:5.2f}".format(frames / (time.time() - start)))
    cv.waitKey(1)



