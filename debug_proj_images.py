import os
import cv2
import matplotlib.pyplot as plt
import screeninfo
import PySpin
import numpy as np
from include.stereo_fringe_process import Stereo_Fringe_Process
from include.StereoCameraController import StereoCameraController
from include.FringePattern import FringePattern


def main():
    pixel_per_fringe = 256
    steps = 6
    path = '/home/daniel/Insync/daniel.regner@labmetro.ufsc.br/Google Drive - Shared drives/VORIS  - Equipe/Sistema de Medição 4 - Stereo Projeção Franjas/Imagens/2024-08-27/pixel_per_fringe_{}_{}'.format(pixel_per_fringe,
                                                                                                 steps)
    # fringe = FringePattern(px_f=pixel_per_fringe, steps=steps)

    left_files = sorted(os.listdir(os.path.join(path, 'left')))
    right_files = sorted(os.listdir(os.path.join(path, 'right')))
    # left_images = np.zeros((1200, 1600, len(left_files)), dtype=np.uint8)
    # right_images = np.zeros((1200, 1600, len(right_files)), dtype=np.uint8)
    stereo = Stereo_Fringe_Process(px_f=pixel_per_fringe, steps=steps)
    for counter in range(len(left_files)):
        left_image = cv2.imread(os.path.join(path, 'left', left_files[counter]), 0)
        right_image = cv2.imread(os.path.join(path, 'right', right_files[counter]), 0)
        # left_image = cv2.equalizeHist(left_image)
        # right_image = cv2.equalizeHist(right_image)
        stereo.set_images(image_left=left_image, image_right=right_image, counter=counter)
        # concatenate = np.hstack((left_image, right_image))
        # cv2.imshow('stereo', concatenate)
        # cv2.waitKey(0)
    stereo.calculate_phi_images()
    # id_r = np.where(stereo.phi_image_right == np.pi)
    # id_l = np.where(stereo.phi_image_left == np.pi)
    # id_list_l = sorted(list(zip(id_l[0], id_l[1])))
    # id_list_r = sorted(list(zip(id_r[0], id_r[1])))
    # print('left')
    # print(id_list_l)
    # print('right')
    # print(id_list_r)

    # phi_stereo = np.hstack((stereo.phi_image_left, stereo.phi_image_right))
    # cv2.imshow('phi stereo', phi_stereo)
    # cv2.waitKey(0)

    stereo.calculate_qsi_images(170, 170)
    stereo.calculate_remaped_qsi_images()
    stereo.plot_abs_phase_map(name='Images - px_f:{} - steps:{}'.format(pixel_per_fringe, steps))
    stereo.plot_phase_map(name='Images - px_f:{} - steps:{}'.format(pixel_per_fringe, steps))
    stereo.plot_qsi_map(name='Images - px_f:{} - steps:{}'.format(pixel_per_fringe, steps))

    print('wait')


if __name__ == '__main__':
    main()