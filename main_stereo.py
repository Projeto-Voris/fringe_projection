import os
import cv2
import screeninfo
import PySpin
import numpy as np
import matplotlib.pyplot as plt
from include.StereoCameraController import StereoCameraController
from include import FringePattern, GrayCode
import PhaseMap


class Main_Stereo:
    def __init__(self, img_resolution=(1600, 1200), num_images=10):
        self.width = img_resolution[0]
        self.height = img_resolution[1]
        self.remaped_qsi_image_left = []
        self.remaped_qsi_image_right = []
        self.qsi_image_left = []
        self.qsi_image_right = []
        self.phi_image_left = []
        self.phi_image_right = []
        self.images_left = np.zeros((img_resolution[1], img_resolution[0], num_images), dtype=np.uint8)
        self.images_right = np.zeros((img_resolution[1], img_resolution[0], num_images), dtype=np.uint8)
        graycode = GrayCode.GrayCode(resolution=img_resolution, n_bits=4)
        self.real_qsi_order = graycode.get_gc_order_v()

    def set_images(self, images_left, images_right):
        self.images_left = images_left
        self.images_right = images_right


    def calculate_phi_images(self):
        self.phi_image_left = PhaseMap.calculate_phi(self.images_left[:, :, :4])
        self.phi_image_right = PhaseMap.calculate_phi(self.images_right[:, :, :4])
        return self.phi_image_left, self.phi_image_right

    def calculate_qsi_images(self):
        self.qsi_image_left = PhaseMap.calculate_qsi(self.images_left[:, :, 4:])
        self.qsi_image_right = PhaseMap.calculate_qsi(self.images_right[:, :, 4:])
        return self.qsi_image_left, self.qsi_image_right

    def calculate_remaped_qsi_images(self):
        self.remaped_qsi_image_left = PhaseMap.remap_qsi_image(self.qsi_image_left, self.real_qsi_order)
        self.remaped_qsi_image_right = PhaseMap.remap_qsi_image(self.qsi_image_right, self.real_qsi_order)
        return self.remaped_qsi_image_left, self.remaped_qsi_image_right

    def calculate_abs_phi_images(self):
        abs_phi_image_left = self.phi_image_left + 2 * np.pi * self.remaped_qsi_image_left
        abs_phi_image_right = self.phi_image_right + 2 * np.pi * self.remaped_qsi_image_right
        return abs_phi_image_left, abs_phi_image_right

    def create_phase_map(self):
        abs_phi_image_left, abs_phi_image_right = self.calculate_abs_phi_images()

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 8))

        # Left Phi and Remaped QSI
        ax1.plot(self.phi_image_left[600, :], color='gray')
        ax1.set_ylabel('Phi Image left', color='gray')
        ax1.tick_params(axis='y', labelcolor='gray')

        ax1_2 = ax1.twinx()
        ax1_2.plot(self.remaped_qsi_image_left[600, :], color='red')
        ax1_2.set_ylabel('Remaped QSI Image left', color='red')
        ax1_2.tick_params(axis='y', labelcolor='red')

        # Left Abs Phi Image 1D
        ax2.plot(abs_phi_image_left[600, :], color='gray')
        ax2.set_title('Abs Phi Image left 1D')
        ax2.set_ylabel('Abs Phi Image left')

        # Left Phi Image 2D
        im3 = ax3.imshow(self.phi_image_left, cmap='gray')
        ax3.set_title('Phi Image left 2D')
        plt.colorbar(im3, ax=ax3)

        # Left Abs Phi Image 2D
        im4 = ax4.imshow(abs_phi_image_left, cmap='gray')
        ax4.set_title('Abs Phi Image left 2D')
        plt.colorbar(im4, ax=ax4)

        # Title for the whole figure
        fig.suptitle('Fase e Fase absoluta left')
        plt.tight_layout()
        plt.show()

        # Right Phi and Remaped QSI
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 8))

        ax1.plot(self.phi_image_right[600, :], color='gray')
        ax1.set_ylabel('Phi Image right', color='gray')
        ax1.tick_params(axis='y', labelcolor='gray')

        ax1_2 = ax1.twinx()
        ax1_2.plot(self.remaped_qsi_image_right[600, :], color='red')
        ax1_2.set_ylabel('Remaped QSI Image right', color='red')
        ax1_2.tick_params(axis='y', labelcolor='red')

        # Right Abs Phi Image 1D
        ax2.plot(abs_phi_image_right[600, :], color='gray')
        ax2.set_title('Abs Phi Image right 1D')
        ax2.set_ylabel('Abs Phi Image right')

        # Right Phi Image 2D
        im3 = ax3.imshow(self.phi_image_right, cmap='gray')
        ax3.set_title('Phi Image right 2D')
        plt.colorbar(im3, ax=ax3)

        # Right Abs Phi Image 2D
        im4 = ax4.imshow(abs_phi_image_right, cmap='gray')
        ax4.set_title('Abs Phi Image right 2D')
        plt.colorbar(im4, ax=ax4)

        # Title for the whole figure
        fig.suptitle('Fase e Fase absoluta right')
        plt.tight_layout()
        plt.show()


def main():
    VISUALIZE = True
    cv2.namedWindow('projector', cv2.WINDOW_NORMAL)

    move = (0, 0)
    width, height = 1024, 1024
    img_resolution = (width, height)
    path = 'C:\\Users\\bianca.rosa\\PycharmProjects\\fringe_projection'
    os.makedirs(path, exist_ok=True)
    stereo = Main_Stereo()

    stereo_ctrl = StereoCameraController(left_serial=16378750, right_serial=16378734)
    print("Models: {}".format(stereo_ctrl.get_model()))
    print("Serial: {}".format(stereo_ctrl.get_serial_numbers()))

    for m in screeninfo.get_monitors():
        if m.name == '\\\\.\\DISPLAY3':
            move = (m.x, m.y)
            img_resolution = (m.width, m.height)

    cv2.setWindowProperty('projector', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.moveWindow('projector', move[0], move[1])

    fringe = FringePattern.FringePattern(resolution=img_resolution, f_sin=16, steps=4)
    graycode = GrayCode.GrayCode(resolution=img_resolution, n_bits=5)
    fringe_images = fringe.get_image()
    graycode_images = graycode.get_images()

    try:
        stereo_ctrl.set_exposure_time(16660.0)
        stereo_ctrl.set_exposure_mode(PySpin.ExposureAuto_Off)
        stereo_ctrl.set_gain(1)
        stereo_ctrl.set_image_format(PySpin.PixelFormat_Mono8)
        stereo_ctrl.start_acquisition()

        if VISUALIZE:
            cv2.namedWindow('Stereo', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('Stereo', 1600, 600)

        count = 0
        k = 0
        n_img = np.concatenate((fringe_images, graycode_images), axis=2)
        num_images = n_img.shape[2]

        images_left = stereo.images_left
        images_right = stereo.images_right

        while k != 27 and count < num_images:
            cv2.imshow('projector', n_img[:, :, count])
            k = cv2.waitKey(10)

            left_img, right_img = stereo_ctrl.capture_images()

            if VISUALIZE:
                img_concatenate = np.concatenate((left_img, right_img), axis=1)
                cv2.imshow('Stereo', img_concatenate)

            if k == 32:
                images_left[:, :, count] = left_img
                images_right[:, :, count] = right_img

                if stereo_ctrl.save_images(path=path, counter=count):
                    count += 1

    finally:
        print("Camera closed")
        cv2.destroyAllWindows()
        stereo_ctrl.stop_acquisition()
        stereo_ctrl.cleanup()
        bl = cv2.threshold(images_left[:,:,4], 180, 255, cv2.THRESH_BINARY)[1]
        br = cv2.threshold(images_right[:,:,4], 180, 255, cv2.THRESH_BINARY)[1]
        images_left = cv2.bitwise_and(images_left, images_left, mask=bl)
        images_right = cv2.bitwise_and(images_right, images_right, mask=br)
        stereo.set_images(images_left, images_right)
        stereo.calculate_phi_images()
        stereo.calculate_qsi_images()
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
        valores_max_branco_left = []
        valores_max_branco_right = []
        for imagem in images_left[:, :, 4]:
            branco_max_left = np.max(imagem)
            valores_max_branco_left.append(branco_max_left)

        media_branco_max_left = np.mean(valores_max_branco_left)
        print("média dos brancos left:", media_branco_max_left)

        for i, valor in enumerate(valores_max_branco_left):
            print(f"O valor máximo de branco na imagem left {i + 1} é: {valor}")
        for imagem in images_right[:, :, 4]:
            branco_max_right = np.max(imagem)
            valores_max_branco_right.append(branco_max_right)

        media_branco_max_right = np.mean(valores_max_branco_right)
        print("media dos brancos right:", media_branco_max_right)

        for j, valor in enumerate(valores_max_branco_right):
            print(f"O valor máximo de branco na imagem right{j  + 1} é: {valor}")


if __name__ == '__main__':
    main()