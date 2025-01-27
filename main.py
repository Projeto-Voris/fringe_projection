import os
import cv2
import screeninfo
import PySpin
import numpy as np
import cupy as cp
from include.stereo_fringe_process import Stereo_Fringe_Process
from include.StereoCameraController import StereoCameraController
from include.InverseTriangulation import InverseTriangulation

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
        # Configura o trigger nas câmeras
        stereo_ctrl.configure_trigger()

        if VISUALIZE:
            cv2.namedWindow('Stereo', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('Stereo', 1600, 600)

        count = 0
        n_img = np.concatenate((fringe_images, graycode_images), axis=2)
        num_images = n_img.shape[2]

        while count < num_images:
            # Projeta as imagens de franjas e graycode
            cv2.imshow('projector', n_img[:, :, count])
            # Delay para projetar a imagem
            cv2.waitKey(200)

            # Aguarda o trigger ser ativado para capturar as imagens
            left_img, right_img = stereo_ctrl.capture_images_with_trigger()

            if VISUALIZE:
                img_concatenate = np.concatenate((left_img, right_img), axis=1)
                cv2.imshow('Stereo', img_concatenate)

            # Salva as imagens automaticamente
            stereo.set_images(left_img, right_img, counter=count)
            # Salva a imagem no diretório se quiser
            # if stereo_ctrl.save_images(path=path, counter=count):

            count += 1

    finally:
        print("Camera closed")
        cv2.destroyAllWindows()
        stereo_ctrl.stop_acquisition()
        stereo_ctrl.cleanup()

        # Acquired the images
        abs_phi_image_left, abs_phi_image_right = stereo.calculate_abs_phi_images(visualize=False)
        modulation_mask_left = stereo.calculate_phi(stereo.images_left[:, :, :7], visualize=False)[0]
        modulation_mask_right = stereo.calculate_phi(stereo.images_right[:, :, :7], visualize=False)[0]

        # read the yaml_file
        yaml_file = '/home/bianca/PycharmProjects/fringe_projection/Params/20241212_calib_daniel.yaml'

        # Inverse Triangulation for Fringe projection
        zscan = InverseTriangulation(yaml_file)

        # np.arange (min_val, max_val, step)
        x_lin = cp.arange(-250, 500, 20)
        y_lin = cp.arange(-100, 400, 20)
        z_lin = cp.arange(-500, 100, 0.1)

        # Número de dívisões do espaço
        num_splits = 10

        # Dívide o espaço em blocos para processamento de cada bloco
        x_split = cp.array_split(x_lin, num_splits)
        y_split = cp.array_split(y_lin, num_splits)

        # Lê as imagens de fase absoluto e o mapa de modularização
        zscan.read_images(left_imgs=abs_phi_image_left, right_imgs=abs_phi_image_right, left_mask=modulation_mask_left,
                          right_mask=modulation_mask_right)

        # Lista para o armazenamento do resultado dos processamentos de cada bloco
        points_result = []
        count = 0
        for x_arr in x_split:
            for y_arr in y_split:
                points_3d = zscan.points3D_arrays(x_arr, y_arr, z_lin, visualize=False)
                z_zcan_points = zscan.fringe_process(points_3d=points_3d, save_points=False, visualize=False)
                points_result.append(z_zcan_points)
                count += 1
                print(count)

        # Junção dos resultados de cada bloco
        points_result_ar = cp.concatenate(points_result, axis=0)

        # Aplicação de filtro com base no processo de octree, onde são descartados pontos que estiverem fora do limite de profundidade e muito distante da média de cada bloco
        points_result_ar_filtered = zscan.filter_points_by_depth(points_result_ar, depth_threshold=0.001)
        points_result_ar_filtered = np.asarray(points_result_ar_filtered.points)

        # Calcular o máximo e mínimo para cada eixo da nuvem de pontos filtrada para segundo prcessamento com maior refinamento
        x_min, x_max = points_result_ar_filtered[:, 0].min(), points_result_ar_filtered[:, 0].max()
        y_min, y_max = points_result_ar_filtered[:, 1].min(), points_result_ar_filtered[:, 1].max()
        z_min, z_max = points_result_ar_filtered[:, 2].min(), points_result_ar_filtered[:, 2].max()

        x_lin_refined = cp.arange(x_min, x_max, 1)
        y_lin_refined = cp.arange(y_min, y_max, 1)
        z_lin_refined = cp.arange(z_min, z_max, 0.1)
        x_split_refined = cp.array_split(x_lin_refined, num_splits)
        y_split_refined = cp.array_split(y_lin_refined, num_splits)
        points_result_refined = []
        count = 0
        for x_arr_r in x_split_refined:
            for y_arr_r in y_split_refined:
                points_3d = zscan.points3D_arrays(x_arr_r, y_arr_r, z_lin_refined, visualize=False)
                z_zcan_points = zscan.fringe_process(points_3d=points_3d, save_points=False, visualize=False)
                points_result_refined.append(z_zcan_points)
                count += 1
                print(count)

        points_result_refined_ar = cp.concatenate(points_result_refined, axis=0)
        points_result_refined_ar_filtered = zscan.filter_points_by_depth(points_result_refined_ar, depth_threshold=0.001)
        points_result_refined_ar_filtered = np.asarray(points_result_refined_ar_filtered.points)

        # Salva os pontos em arquivo .txt
        # np.savetxt('fringe_points_results.txt', points_result_refined_ar_filtered, fmt='%.6f', delimiter=' ')

        zscan.plot_3d_points(points_result_refined_ar_filtered[:, 0], points_result_refined_ar_filtered[:, 1],
                             points_result_refined_ar_filtered[:, 2], color=None, title='Filtered Points')
        print('wait')

if __name__ == '__main__':
    main()