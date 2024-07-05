import os
import time
import cv2
import screeninfo
import PySpin
from include.StereoCameraController import StereoCameraController
import GrayCode
import numpy as np
import FringePattern

VISUALIZE = True

if __name__ == '__main__':
    cv2.namedWindow('projector', cv2.WINDOW_NORMAL)  # create window to fringe and graycode

    move = (0, 0)  # initialize variable to move opencv window
    width = 1024
    height = 1024
    img_resolution = (width, height)  # initialize image resolution
    path = 'C:\\Users\\bianca.rosa\PycharmProjects\\fringe_projection\\teste'
    os.makedirs(path, exist_ok=True)

    stereo_ctrl = StereoCameraController(left_serial=16378750, right_serial=16378734)  # set stereo cameras class
    # Get and print the serial numbers
    print("Models: {}".format(stereo_ctrl.get_model()))
    print("Serial: {}".format(stereo_ctrl.get_serial_numbers()))

    for m in screeninfo.get_monitors():  # verify all monitors
        if m.name == '\\\\.\\DISPLAY3':  # Search for projector connected a specific input port
            move = (m.x, m.y)  # get movement from primary display
            img_resolution = (m.width, m.height)  # get projector resolution
    cv2.setWindowProperty('projector', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)  # change window property
    cv2.moveWindow('projector', move[0], move[1])  # move projector window to projector display
    graycode = GrayCode.GrayCode(resolution=img_resolution, n_bits=8, axis=0)  # create graycode class
    images = graycode.get_images()  # get graycode images
    fringe = FringePattern.FringePattern(resolution=img_resolution, f_sin=4, steps=8)  # create FringePattern class
    fringe.create_fringe_image()
    image_f = fringe.get_image()  # get FringePattern images

    try:
        # Set camera parameters
        stereo_ctrl.set_exposure_time(16660.0)  # (us) Para n ter interf. rede elétrica (60 Hz). (1/60s = 0,016 Hz)
        stereo_ctrl.set_exposure_mode(PySpin.ExposureAuto_Off)  # Set exposure mode to manual
        stereo_ctrl.set_gain(0)  # Set gain (dB)
        stereo_ctrl.set_image_format(PySpin.PixelFormat_Mono8)  # Set image format to Mono8 (color is BRG8)
        stereo_ctrl.start_acquisition()

        cv2.namedWindow('Stereo', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Stereo', 1600, 600)

        count = 0
        k = 0
        n_img = np.concatenate((image_f, images), axis=2)
        # start_time = time.perf_counter()
        # frames = 0
        # fps = 0
        while k != 27 or count <= n_img.shape[2]:
            t1 = time.time()
            left_image, right_image = stereo_ctrl.capture_images()  # Get images
            cv2.imshow('projector', n_img[:, :, count])
            t2 = time.time()
            tt = t2 - t1
            print(f"tt: ", tt*1000)
            #frames += 1
            img_concatenate = np.concatenate((left_image, right_image), axis=1)  # concatenate to visualize
            cv2.imshow('Stereo', img_concatenate)
            print("OK")

            if stereo_ctrl.save_images(left=left_image, right=right_image, path=path, counter=count): #and k == 32:
                count += 1

            k = cv2.waitKey(1)

            # if (time.perf_counter() - start_time) > 1:
            #     fps = round(frames / (time.perf_counter() - start_time), 1)
            #     frames = 0
            #     start_time = time.perf_counter()
            # print("FPS: ", fps)

        # if k == 32:
        #     stereo_ctrl.save_images(path=path)

    finally:
        print("Camera closed")
        # Stop acquisition and cleanup resources
        stereo_ctrl.stop_acquisition()
        stereo_ctrl.cleanup()
