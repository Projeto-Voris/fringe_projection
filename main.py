import os
import cv2
import screeninfo
import PySpin
import numpy as np
from include.stereo_fringe_process import Stereo_Fringe_Process
from include.StereoCameraController import StereoCameraController
import inverse_triangulation
from include.InverseTriangulation import InverseTriangulation
def main():
    VISUALIZE = True
    cv2.namedWindow('projector', cv2.WINDOW_NORMAL)

    move = (0, 0)
    width, height = 1024, 1024
    img_resolution = (width, height)
    pixel_per_fringe = 32
    steps = 8
    path = '/home/daniel/PycharmProjects/fringe_projection/images/pixel_per_fringe_{}_{}'.format(pixel_per_fringe, steps)
    os.makedirs(path, exist_ok=True)


    stereo_ctrl = StereoCameraController(left_serial=16378750, right_serial=16378734)
    print("Models: {}".format(stereo_ctrl.get_model()))
    print("Serial: {}".format(stereo_ctrl.get_serial_numbers()))

    for m in screeninfo.get_monitors():

        if m.name == 'DP-3':
            move = (m.x, m.y)
            img_resolution = (m.width, m.height)

    cv2.setWindowProperty('projector', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.moveWindow('projector', move[0], move[1])
    stereo = Stereo_Fringe_Process(img_resolution=img_resolution, px_f=pixel_per_fringe, steps=steps)
    fringe_images = stereo.get_fr_image()
    graycode_images = stereo.get_gc_images()
    k = 0

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

        if k != 27:
            #     width, height, _ = self.images_left.shape
            bl = cv2.threshold(stereo.images_left[:, :, 4], 180, 255, cv2.THRESH_BINARY)[1]
            br = cv2.threshold(stereo.images_right[:, :, 4], 180, 255, cv2.THRESH_BINARY)[1]
            # stereo.calculate_phi(stereo.images_left[:, :, :int(stereo.get_steps())])
            white_left, white_right = stereo.normalize_white(bl, br)
            # plt.imshow(bl, cmap='gray')
            # plt.show()
            # plt.imshow(br, cmap='gray')
            # plt.show()
            # qsi_left = stereo.calculate_qsi(stereo.images_left[:, :, 8:])
            # qsi_right = stereo.calculate_qsi(stereo.images_right[:, :, 8:])
            # stereo.remap_qsi_image(qsi_left, stereo.get_gc_order_v())
            # stereo.remap_qsi_image(qsi_right, stereo.get_gc_order_v())
            # stereo.plot_abs_phase_map(name='Images - px_f:{} - steps:{}'.format(pixel_per_fringe, steps))
            # stereo.plot_qsi_map(name='Images - px_f:{} - steps:{}'.format(pixel_per_fringe, steps))
            stereo.calculate_abs_phi_images(visualize=False)

        # Acquired the abs images
        abs_phi_image_left, abs_phi_image_right = stereo.calculate_abs_phi_images(visualize=False)
        # read the yaml_file
        yaml_file = '/home/daniel/PycharmProjects/fringe_projection/params/20241018_bouget.yaml'


        # zscan_1 = inverse_triangulation.inverse_triangulation()
        # # Acquired the points 3D
        # points_3d = zscan_1.points3d(x_lim=(-250, 500), y_lim=(-100, 400), z_lim=(-200, 200), xy_step=7, z_step=0.1, visualize=False)
        # # Interpolated the points and build the point cloud
        # zscan_1.fringe_zscan(left_images=abs_phi_image_left, right_images=abs_phi_image_right,yaml_file=yaml_file, points_3d=points_3d)

        # Inverse Triangulation for Fringe projection
        zscan = InverseTriangulation(yaml_file)
        zscan.points3d(x_lim=(-250, 500), y_lim=(-100, 400), z_lim=(-200, 200), xy_step=7, z_step=0.1, visualize=False)
        zscan.read_images(left_imgs=abs_phi_image_left, right_imgs=abs_phi_image_right)
        z_zcan_points = zscan.fringe_process(save_points=False, visualize=True)

if __name__ == '__main__':
    main()
