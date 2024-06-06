import cv2
import screeninfo
import PySpin
from include.StereoCameraController import StereoCameraController
import GrayCode
import numpy as np

VISUALIZE = True

if __name__ == '__main__':
    cv2.namedWindow('projector', cv2.WINDOW_NORMAL)  # create window to fringe and graycode

    move = (0, 0)  # initialize variable to move opencv window
    img_resolution = (800, 600)  # initialize image resolution

    stereo_ctrl = StereoCameraController(left_serial=16378749, right_serial=16378753)  # set stereo cameras class
    # Get and print the serial numbers
    print("Models: {}".format(stereo_ctrl.get_model()))
    print("Serial: {}".format(stereo_ctrl.get_serial_numbers()))

    for m in screeninfo.get_monitors():  # verify all monitors
        if m.name == 'DVI-D-0':  # Search for projector connected a specific input port
            move = (m.x, m.y)  # get movement from primary display
            img_resolution = (m.width, m.height)  # get projector resolution
    cv2.setWindowProperty('projector', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)  # change window property
    cv2.moveWindow('projector', move[0], move[1])  # move projector window to projector display
    graycode = GrayCode.GrayCode(resolution=img_resolution, n_bits=8, axis=0)  # create graycode class
    images = graycode.get_images()  # get graycode images

    try:
        # Set camera parameters
        stereo_ctrl.set_exposure_time(16660.0)  # (us) Para n ter interf. rede elétrica (60 Hz). (1/60s = 0,016 Hz)
        stereo_ctrl.set_exposure_mode(PySpin.ExposureAuto_Off)  # Set exposure mode to manual
        stereo_ctrl.set_gain(1)  # Set gain (dB)
        stereo_ctrl.set_image_format(PySpin.PixelFormat_Mono8)  # Set image format to Mono8 (color is BRG8)
        stereo_ctrl.start_acquisition()

        left_image, right_image = stereo_ctrl.capture_images()  # Get images
        count = 0
        k = 0
        while k != 27:
            k = cv2.waitKey(10)
            cv2.imshow('projector', left_image[:, :, count])

        if VISUALIZE:
            img_concatenate = np.concatenate((left_image, right_image), axis=1)  # concatenate to visualize
            cv2.namedWindow('Stereo', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('Stereo', int(img_concatenate.shape[1] / 2), int(img_concatenate.shape[0] / 2))
            cv2.imshow('Stereo', img_concatenate)

        if k == 32:
            stereo_ctrl.save_images(path=path)

    finally:
        print("Camera closed")
        # Stop acquisition and cleanup resources
        stereo_ctrl.stop_acquisition()
        stereo_ctrl.cleanup()
