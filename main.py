import os
import cv2
import matplotlib.pyplot as plt
import screeninfo
import PySpin
import numpy as np
from include.stereo_fringe_process import Stereo_Fringe_Process
from include.StereoCameraController import StereoCameraController


def main():
    VISUALIZE = True
    cv2.namedWindow('projector', cv2.WINDOW_NORMAL)

    move = (0, 0)
    width, height = 1024, 1024
    img_resolution = (width, height)
    # path = 'C:\\Users\\bianca.rosa\\PycharmProjects\\fringe_projection'
    path = '/home/daniel/PycharmProjects/fringe_projection/images'
    os.makedirs(path, exist_ok=True)

    stereo_ctrl = StereoCameraController(left_serial=16378750, right_serial=16378734)
    print("Models: {}".format(stereo_ctrl.get_model()))
    print("Serial: {}".format(stereo_ctrl.get_serial_numbers()))

    for m in screeninfo.get_monitors():
        if m.name == 'DP-5':
            move = (m.x, m.y)
            img_resolution = (m.width, m.height)

    cv2.setWindowProperty('projector', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.moveWindow('projector', move[0], move[1])
    stereo = Stereo_Fringe_Process(img_resolution=img_resolution, f_sin=75, steps=4)
    fringe_images = stereo.get_fr_image()
    graycode_images = stereo.get_gc_images()

    try:
        stereo_ctrl.set_exposure_time(16666.0)
        stereo_ctrl.set_exposure_mode(PySpin.ExposureAuto_Off)
        stereo_ctrl.set_gain(0)
        stereo_ctrl.set_image_format(PySpin.PixelFormat_Mono8)
        stereo_ctrl.start_acquisition()

        if VISUALIZE:
            cv2.namedWindow('Stereo', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('Stereo', 1600, 600)

        count = 0
        k = 0
        n_img = np.concatenate((fringe_images, graycode_images), axis=2)
        num_images = n_img.shape[2]

        while k != 27 and count < num_images:
            cv2.imshow('projector', n_img[:, :, count])
            k = cv2.waitKey(10)

            left_img, right_img = stereo_ctrl.capture_images()

            if VISUALIZE:
                img_concatenate = np.concatenate((left_img, right_img), axis=1)
                cv2.imshow('Stereo', img_concatenate)

            if k == 32:
                stereo.set_images(left_img, right_img, counter=count)

                if stereo_ctrl.save_images(path=path, counter=count):
                    count += 1

    finally:
        print("Camera closed")
        cv2.destroyAllWindows()
        stereo_ctrl.stop_acquisition()
        stereo_ctrl.cleanup()
        # stereo.normalize_b_w()

        #     width, height, _ = self.images_left.shape
        bl = cv2.threshold(stereo.images_left[:, :, 4], 180, 255, cv2.THRESH_BINARY)[1]
        br = cv2.threshold(stereo.images_right[:, :, 4], 180, 255, cv2.THRESH_BINARY)[1]
        stereo.calculate_phi_images()
        # white_left, white_right = stereo.normalize_white(bl, br)
        # plt.imshow(bl, cmap='gray')
        # plt.show()
        # plt.imshow(br, cmap='gray')
        # plt.show()
        stereo.calculate_qsi_images(150, 150)
        stereo.calculate_remaped_qsi_images()
        stereo.create_phase_map()
        # branco_maximo_left = np.max(images_left)
        # branco_minimo_left = np.min(images_left)
        # branco_maximo_right = np.max(images_right)
        # branco_minimo_right = np.min(images_right)
        # print("valor maximo left:", branco_maximo_left)
        # print("valor minimo left:", branco_minimo_left)
        # print("valor maximo right:", branco_maximo_right)
        # print("valor minimo right:", branco_minimo_right)


if __name__ == '__main__':
    main()
