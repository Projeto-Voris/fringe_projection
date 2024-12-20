import os
import cv2
import screeninfo
import PySpin
import numpy as np
import cupy as cp
from include.stereo_fringe_process import Stereo_Fringe_Process
from include.StereoCameraController import StereoCameraController
from include.InverseTriangulation import InverseTriangulation
import octree_process


def main():
    VISUALIZE = True
    cv2.namedWindow('projector', cv2.WINDOW_NORMAL)

    move = (0, 0)
    width, height = 1024, 1024
    img_resolution = (width, height)
    pixel_per_fringe = 32
    steps = 8
    # path = '/home/daniel/PycharmProjects/fringe_projection/images/pixel_per_fringe_{}_{}'.format(pixel_per_fringe, steps)
    path = '/home/bianca/PycharmProjects/fringe_projection/images/pixel_per_fringe_{}_{}'.format(pixel_per_fringe, steps)
    os.makedirs(path, exist_ok=True)

    stereo_ctrl = StereoCameraController(left_serial=16378750, right_serial=16378734)
    print("Models: {}".format(stereo_ctrl.get_model()))
    print("Serial: {}".format(stereo_ctrl.get_serial_numbers()))

    for m in screeninfo.get_monitors():

        if m.name == 'DP-1':
            move = (m.x, m.y)
            img_resolution = (m.width, m.height)

    cv2.setWindowProperty('projector', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.moveWindow('projector', move[0], move[1])
    stereo = Stereo_Fringe_Process(img_resolution=img_resolution, px_f=pixel_per_fringe, steps=steps)
    fringe_images = stereo.get_fr_image()
    graycode_images = stereo.get_gc_images()
    k = 0

    try:
        stereo_ctrl.set_exposure_time(1666.0)
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
                # if stereo_ctrl.save_images(path=path, counter=count):
                count += 1

    finally:
        print("Camera closed")
        cv2.destroyAllWindows()
        stereo_ctrl.stop_acquisition()
        stereo_ctrl.cleanup()
        # stereo.normalize_b_w()

        # if k != 27:
        #     #     width, height, _ = self.images_left.shape
        #     bl = cv2.threshold(stereo.images_left[:, :, 4], 180, 255, cv2.THRESH_BINARY)[1]
        #     br = cv2.threshold(stereo.images_right[:, :, 4], 180, 255, cv2.THRESH_BINARY)[1]
        #     # stereo.calculate_phi(stereo.images_left[:, :, :int(stereo.get_steps())])
        #     white_left, white_right = stereo.normalize_white(bl, br)
        #     # plt.imshow(bl, cmap='gray')
        #     # plt.show()
        #     # plt.imshow(br, cmap='gray')
        #     # plt.show()
        #     # qsi_left = stereo.calculate_qsi(stereo.images_left[:, :, 8:])
        #     # qsi_right = stereo.calculate_qsi(stereo.images_right[:, :, 8:])
        #     # stereo.remap_qsi_image(qsi_left, stereo.get_gc_order_v())
        #     # stereo.remap_qsi_image(qsi_right, stereo.get_gc_order_v())
        #     # stereo.plot_abs_phase_map(name='Images - px_f:{} - steps:{}'.format(pixel_per_fringe, steps))
        #     # stereo.plot_qsi_map(name='Images - px_f:{} - steps:{}'.format(pixel_per_fringe, steps))
        #     stereo.calculate_abs_phi_images(visualize=False)

        # Acquired the abs images
        abs_phi_image_left, abs_phi_image_right = stereo.calculate_abs_phi_images(visualize=False)

        modulation_mask_left = stereo.calculate_phi(stereo.images_left[:, :, :7], visualize=False)[0]
        modulation_mask_right = stereo.calculate_phi(stereo.images_right[:, :, :7], visualize=False)[0]

        # read the yaml_file
        # yaml_file = '/home/daniel/PycharmProjects/fringe_projection/params/20241018_bouget.yaml'
        yaml_file = '/home/bianca/PycharmProjects/fringe_projection/Params/20241212_calib_daniel.yaml'

        # Inverse Triangulation for Fringe projection
        zscan = InverseTriangulation(yaml_file)

        # np.arange (min_val, max_val, step)
        x_lin = cp.arange(-250, 500, 4)
        y_lin = cp.arange(-100, 400, 4)
        z_lin = cp.arange(-200, 200, 2)
        num_splits = 10
        x_split = cp.array_split(x_lin, num_splits)
        y_split = cp.array_split(y_lin, num_splits)
        points_result = []
        count = 0
        for x_arr in x_split:
            for y_arr in y_split:
                points_3d = zscan.points3D_arrays(x_arr, y_arr, z_lin, visualize=False)
                zscan.read_images(left_imgs=abs_phi_image_left, right_imgs=abs_phi_image_right, left_mask=modulation_mask_left, right_mask=modulation_mask_right)
                z_zcan_points = zscan.fringe_process(points_3d=points_3d, save_points=False, visualize=False)
                points_result.append(z_zcan_points)
                count += 1
                print(count)

        # for i, array in enumerate(points_result):
        #     print(f"Shape do array {i}: {array.shape}")

        points_result_ar = cp.concatenate(points_result, axis=0)
        # points_result_ar_filtered = octree_process.filter_points_by_depth(points_result_ar, depth_threshold=0.001)
        # points_result_ar_filtered = np.asarray(points_result_ar_filtered.points)
        zscan.plot_3d_points(points_result_ar[:,0], points_result_ar[:,1], points_result_ar[:,2], color=None, title='Filtered Points')
        # zscan.plot_3d_points(points_result_ar_filtered[:, 0], points_result_ar_filtered[:, 1],
        #                      points_result_ar_filtered[:, 2], color=None, title='Filtered Points')
        print('wait')

if __name__ == '__main__':
    main()