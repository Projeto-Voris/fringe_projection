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
SAVE = True
delay = 5*1e3
if __name__ == '__main__':
    cv2.namedWindow('projector', cv2.WINDOW_NORMAL)  # create window to fringe and graycode
    RUN = True

    move = (0, 0)  # initialize variable to move opencv window
    width = 1024
    height = 1024
    img_resolution = (width, height)  # initialize image resolution
    path = '/home/daniel/PycharmProjects/fringe_projection/teste'
    os.makedirs(path, exist_ok=True)

    stereo_ctrl = StereoCameraController(left_serial=16378750, right_serial=16378734)  # set stereo cameras class
    # Get and print the serial numbers
    print("Models: {}".format(stereo_ctrl.get_model()))
    print("Serial: {}".format(stereo_ctrl.get_serial_numbers()))

    for m in screeninfo.get_monitors():  # verify all monitors
        if m.name == 'DP-3':  # '\\\\.\\DISPLAY3':  # Search for projector connected a specific input port
            move = (m.x, m.y)  # get movement from primary display
            img_resolution = (m.width, m.height)  # get projector resolution

    cv2.setWindowProperty('projector', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)  # change window property
    cv2.moveWindow('projector', move[0], move[1])  # move projector window to projector display
    graycode = GrayCode.GrayCode(resolution=img_resolution, n_bits=8, axis=0)  # create graycode class
    images = graycode.get_images()  # get graycode images
    fringe = FringePattern.FringePattern(resolution=img_resolution, f_sin=8, steps=8)  # create FringePattern class
    fringe.create_fringe_image()
    image_f = fringe.get_image()  # get FringePattern images
    all_imgs = []

    try:
        # Set camera parameters
        stereo_ctrl.set_exposure_time(1666.0)  # (us) Para n ter interf. rede elétrica (60 Hz). (1/60s = 0,016 Hz)
        stereo_ctrl.set_exposure_mode(PySpin.ExposureAuto_Off)  # Set exposure mode to manual
        stereo_ctrl.set_gain(1)  # Set gain (dB)
        stereo_ctrl.set_image_format(PySpin.PixelFormat_Mono8)  # Set image format to Mono8 (color is BRG8)
        # stereo_ctrl.set_frame_rate(60.0)  # Set desired frame rate
        stereo_ctrl.start_acquisition()

        if VISUALIZE:
            cv2.namedWindow('Stereo', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('Stereo', 1600, 600)

        count = 0
        k = 0
        n_img = np.concatenate((image_f, images), axis=2)
        t0 = time.time()
        while RUN:
            if k == 27 or count >= n_img.shape[2] - 1:
                RUN = False

            t1 = time.time()
            cv2.imshow('projector', n_img[:, :, count])

            k = cv2.waitKey(1)

            t2 = time.time()
            stereo_ctrl.capture_images()  # Get images

            if VISUALIZE:
                img_concatenate = np.concatenate((stereo_ctrl.get_images()), axis=1)  # concatenate to visualize
                cv2.imshow('Stereo', img_concatenate)
            t3 = time.time()

            if (t3 - t0)*1e3 > delay:
                cv2.waitKey(100)
                if stereo_ctrl.save_images(path=path, counter=count):
                    count += 1
            t4 = time.time()
            print("Proj. dt: {:.2f}".format((t2 - t1) * 1000))
            print("Img.  dt: {:.2f}".format((t3 - t2) * 1000))
            print("Save  dt: {:.2f}".format((t4 - t3) * 1000))

    finally:
        print("Camera closed")
        # Stop acquisition and cleanup resources
        cv2.destroyAllWindows()
        stereo_ctrl.stop_acquisition()
        stereo_ctrl.cleanup()


