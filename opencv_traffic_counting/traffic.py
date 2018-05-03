import os
import logging
import logging.handlers
import random

import numpy as np
import skvideo.io
import cv2
import matplotlib.pyplot as plt

import utils
from pydub import AudioSegment
AudioSegment.converter = "./ffmpeg"
# without this some strange errors happen
cv2.ocl.setUseOpenCL(False)
random.seed(123)

def get_point(event, x, y, flags, param):
    global x1,y1,x2,y2,drawing,mode,cap,template,tempFlag
    image=img
    if event ==cv2.EVENT_LBUTTONDOWN:
        tempFlag=True
        drawing=True
        x1,y1=x,y
        #print(x,y)
    elif event==cv2.EVENT_LBUTTONUP:
        if drawing==True:
            drawing=False
            #print(ix, iy)
            x2,y2=x,y
            #draw_line(image,ix,iy,x,y)
            #cv2.destroyAllWindows()
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 200, 0), -1)
from pipeline import (
    PipelineRunner,
    ContourDetection,
    Visualizer,
    CsvWriter,
    VehicleCounter)

# ============================================================================
IMAGE_DIR = "./out"
VIDEO_SOURCE = "test_01.mp4"
SHAPE = (720, 1280)  # HxW

EXIT_PTS = np.array([
    #[[100, 720], [100, 530], [1100, 400], [1100, 720]] #左下，左上，右上，右下
    [[100, 720], [100, 500], [1100, 500], [1100, 720]]
])


# ============================================================================


def train_bg_subtractor(inst, cap, num=500):
    '''
        BG substractor need process some amount of frames to start giving result
    '''
    print ('Training BG Subtractor...')
    i = 0
    for frame in cap:
        inst.apply(frame, None, 0.001)
        i += 1
        if i >= num:
            return cap


def main():
    log = logging.getLogger("main")

    # creating exit mask from points, where we will be counting our vehicles
    base = np.zeros(SHAPE + (3,), dtype='uint8')
    exit_mask = cv2.fillPoly(base, EXIT_PTS, (255, 255, 255))[:, :, 0]

    # there is also bgslibrary, that seems to give better BG substruction, but
    # not tested it yet
    bg_subtractor = cv2.createBackgroundSubtractorMOG2(
        history=500, detectShadows=True)

    # processing pipline for programming conviniance
    pipeline = PipelineRunner(pipeline=[
        ContourDetection(bg_subtractor=bg_subtractor,
                         save_image=True, image_dir=IMAGE_DIR),
        # we use y_weight == 2.0 because traffic are moving vertically on video
        # use x_weight == 2.0 for horizontal.
        VehicleCounter(exit_masks=[exit_mask], y_weight=2.0),
        Visualizer(image_dir=IMAGE_DIR),
        CsvWriter(path='./', name='report.csv')
    ], log_level=logging.DEBUG)

    # Set up image source
    # You can use also CV2, for some reason it not working for me
    cap = skvideo.io.vreader(VIDEO_SOURCE)

    # skipping 500 frames to train bg subtractor
    train_bg_subtractor(bg_subtractor, cap, num=500)

    cap = skvideo.io.vreader(VIDEO_SOURCE)  #重新读入
    print(type(cap))
    _frame_number = -1
    frame_number = -1
    for frame in cap:
        if not frame.any():
            log.error("Frame capture failed, stopping...")
            break

        # real frame number
        _frame_number += 1

        # skip every 2nd frame to speed up processing
        if _frame_number % 2 != 0:
            continue

        # frame number that will be passed to pipline
        # this needed to make video from cutted frames
        frame_number += 1

        # plt.imshow(frame)
        # plt.show()
        # return

        pipeline.set_context({
            'frame': frame,
            'frame_number': frame_number,
        })
        pipeline.run()

# ============================================================================
x1,y1,x2,y2=0,0,0,0
if __name__ == "__main__":
    log = utils.init_logging()

    if not os.path.exists(IMAGE_DIR):
        log.debug("Creating image directory `%s`...", IMAGE_DIR)
        os.makedirs(IMAGE_DIR)


    cv2.namedWindow('shape',0)
    cv2.resizeWindow("enhanced", 640, 480)
    cv2.setMouseCallback('shape', get_point)
    capture = cv2.VideoCapture("test_01.mp4")
    ret, img = capture.read()

    cv2.imwrite('head.jpg', img)
    while 1:
        cv2.imshow('shape', img)
        Key=cv2.waitKey()
        if(Key==32):
            continue
        if (Key == 27):
            img=cv2.imread('head.jpg')
            cv2.imshow('shape',img)
            #break;

        if(Key==13):
            break

    cv2.destroyAllWindows()


    EXIT_PTS=np.array([
    #[[100, 720], [100, 530], [1100, 400], [1100, 720]] #左下，左上，右上，右下
    [[x1, y2], [x1, y1], [x2, y1], [x2, y2]]
    ])
    main()
