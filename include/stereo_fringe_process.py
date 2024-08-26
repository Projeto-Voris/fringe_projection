import numpy as np
import matplotlib.pyplot as plt
import cv2
# Import classes
from include.FringePattern import FringePattern
from include.GrayCode import GrayCode


class Stereo_Fringe_Process(GrayCode, FringePattern):
    def __init__(self, img_resolution=(1024, 768), camera_resolution=(1600, 1200), px_f=16, steps=4):
        self.remaped_qsi_image_left = np.zeros(img_resolution, np.uint8)
        self.remaped_qsi_image_right = np.zeros(img_resolution, np.uint8)
        self.qsi_image_left = np.zeros(img_resolution, np.uint8)
        self.qsi_image_right = np.zeros(img_resolution, np.uint8)
        self.phi_image_left = np.zeros(img_resolution, np.uint8)
        self.phi_image_right = np.zeros(img_resolution, np.uint8)
        self.images_left = np.zeros(
            (camera_resolution[1], camera_resolution[0],
             int(steps + self.min_bits_gc(np.floor(img_resolution[0] / px_f)) + 2)), np.uint8)
        self.images_right = np.zeros(
            (camera_resolution[1], camera_resolution[0],
             int(steps + self.min_bits_gc(np.floor(img_resolution[0] / px_f)) + 2)), np.uint8)
        GrayCode.__init__(self, resolution=img_resolution, n_bits=self.min_bits_gc(np.floor(img_resolution[0] / px_f)),
                          px_f=px_f)
        FringePattern.__init__(self, resolution=img_resolution, px_f=px_f, steps=steps)

    def min_bits_gc(self, x):
        if x <= 0:
            raise ValueError("Input must be a positive integer.")
        n = 0
        power_of_2 = 1
        # Keep doubling the power_of_2 until it exceeds x
        while power_of_2 < x:
            power_of_2 *= 2
            n += 1
        return n + 1

    def normalize_white(self, mask_left, mask_right):
        # Assuming self.mask_left and self.mask_right are the masks for left and right images
        # They should have the same dimensions as the respective images

        # Apply mask to the left images and calculate the mean of max values
        masked_left_images = self.images_left[:, :, self.steps] * (mask_left == 255)
        media_branco_max_left = np.mean(masked_left_images[mask_left == 255])

        # Apply mask to the right images and calculate the mean of max values
        masked_right_images = self.images_right[:, :, self.steps] * (mask_right == 255)
        media_branco_max_right = np.mean(masked_right_images[mask_right == 255])

        print("media dos brancos right:", media_branco_max_right)

        return media_branco_max_left, media_branco_max_right

    def set_images(self, image_left, image_right, counter):
        self.images_left[:, :, counter] = image_left
        self.images_right[:, :, counter] = image_right

    def calculate_phi_images(self):
        self.phi_image_left = self.calculate_phi(self.images_left[:, :, :int(FringePattern.get_steps(self))])
        self.phi_image_right = self.calculate_phi(self.images_right[:, :, :int(FringePattern.get_steps(self))])
        # return self.phi_image_left, self.phi_image_right

    def calculate_qsi_images(self, w_left, w_right):
        self.qsi_image_left = self.calculate_qsi(self.images_left[:, :, FringePattern.get_steps(self):],
                                                 white_value=w_left)
        self.qsi_image_right = self.calculate_qsi(self.images_right[:, :, FringePattern.get_steps(self):],
                                                  white_value=w_right)
        # return qsi_image_left, qsi_image_right

    def calculate_remaped_qsi_images(self):
        self.remaped_qsi_image_left = self.remap_qsi_image(self.qsi_image_left, GrayCode.get_gc_order_v(self))
        self.remaped_qsi_image_right = self.remap_qsi_image(self.qsi_image_right, GrayCode.get_gc_order_v(self))
        # return self.remaped_qsi_image_left, self.remaped_qsi_image_right

    def calculate_abs_phi_images(self):
        # abs_phi_image_left = self.phi_image_left + 2 * np.pi * self.remaped_qsi_image_left
        abs_phi_image_right = self.phi_image_right + 2 * np.pi * np.floor(self.remaped_qsi_image_right/2)
        # abs_phi_image_right = np.unwrap(abs_phi_image_right)
        abs_phi_image_left = np.zeros((self.images_left.shape[0], self.images_left.shape[1]), np.uint8)
        # abs_phi_image_right = np.zeros((self.images_right.shape[0], self.images_right.shape[1]), np.uint8)
        for i in range(self.images_left.shape[0]):
            for j in range(self.images_left.shape[1]):
                if self.phi_image_left[i, j] <= -np.pi / 2:
                    abs_phi_image_left[i, j] = self.phi_image_left[i, j] + 2 * np.pi * np.floor(
                        (self.qsi_image_left[i, j] + 1) / 2) + np.pi

                elif -np.pi / 2 < self.phi_image_left[i, j] < np.pi / 2:
                    abs_phi_image_left[i, j] = self.phi_image_left[i, j] + 2 * np.pi * np.floor(
                        self.qsi_image_left[i, j] / 2) + np.pi

                elif self.phi_image_left[i, j] >= np.pi / 2:
                    abs_phi_image_left[i, j] = self.phi_image_left[i, j] + 2 * np.pi * (
                                np.floor((self.qsi_image_left[i, j] + 1) / 2) - 1) + np.pi

        return abs_phi_image_left, abs_phi_image_right

    def calculate_phi(self, image):

        height, width, channels = image.shape
        n = channels

        # Calcular os valores de seno e cosseno para todos os canais de uma vez
        indices = np.arange(1, n + 1)
        sin_values = np.sin(2 * np.pi * indices / n)
        cos_values = np.cos(2 * np.pi * indices / n)

        # Multiplicar a imagem pelos valores de seno e cosseno
        sin_contributions = image * sin_values
        cos_contributions = image * cos_values

        # Somar as contribuições ao longo dos canais
        sum_sin = np.sum(sin_contributions, axis=2)
        sum_cos = np.sum(cos_contributions, axis=2)

        # Calcular Phi para cada pixel
        phi_image = np.arctan2(-sum_sin, sum_cos)

        return phi_image

    def calculate_qsi(self, graycode_image, white_value):
        height, width, _ = graycode_image.shape

        # Converter os valores relevantes da imagem graycode para inteiros
        bit_values = (graycode_image[:, :, 2:] / white_value).astype(int)

        # Converter cada linha de bits em um único número inteiro
        qsi_image = np.dot(bit_values, 2 ** np.arange(bit_values.shape[-1])[::-1])

        return qsi_image

    def remap_qsi_image(self, qsi_image, real_qsi_order):
        # Cria um mapeamento dos valores originais para os novos índices
        value_to_new_index_map = {value: new_index for new_index, value in enumerate(real_qsi_order)}

        # Mapeia os valores da qsi_image usando numpy para operações vetorizadas
        remapped_qsi_image = np.vectorize(value_to_new_index_map.get)(qsi_image)

        return remapped_qsi_image

    def create_phase_map(self):
        abs_phi_image_left, abs_phi_image_right = self.calculate_abs_phi_images()

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 8))

        # Left Phi and Remaped QSI
        ax1.plot(self.phi_image_left[int(self.images_left.shape[1] / 2), :], color='gray')
        ax1.set_ylabel('Phi Image left', color='gray')
        ax1.tick_params(axis='y', labelcolor='gray')

        ax1_2 = ax1.twinx()
        ax1_2.plot(self.remaped_qsi_image_left[int(self.images_left.shape[1] / 2), :], color='red')
        ax1_2.set_ylabel('Remaped QSI Image left', color='red')
        ax1_2.tick_params(axis='y', labelcolor='red')

        ax2.plot(self.phi_image_right[int(self.images_right.shape[1] / 2), :], color='gray')
        ax2.set_ylabel('Phi Image right', color='gray')
        ax3.tick_params(axis='y', labelcolor='gray')

        ax2_2 = ax2.twinx()
        ax2_2.plot(self.remaped_qsi_image_right[int(self.images_right.shape[1] / 2), :], color='red')
        ax2_2.set_ylabel('Remaped QSI Image right', color='red')
        ax2_2.tick_params(axis='y', labelcolor='red')

        # Left Phi Image 2D
        im3 = ax3.imshow(self.phi_image_left, cmap='gray')
        ax3.set_title('Phi Image left 2D')
        plt.colorbar(im3, ax=ax3)

        # Right Phi Image 2D
        im4 = ax4.imshow(self.phi_image_right, cmap='gray')
        ax4.set_title('Phi Image Right 2D')
        plt.colorbar(im3, ax=ax4)



        # Title for the whole figure
        fig.suptitle('Fase franjas')
        plt.tight_layout()
        plt.show()

        # Right Phi and Remaped QSI
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 8))

        # Left Abs Phi Image 1D
        ax1.plot(abs_phi_image_left[int(self.images_left.shape[1] / 2), :], color='gray')
        ax1.set_title('Abs Phi Image left 1D')
        ax1.set_ylabel('Abs Phi Image left')

        ax1_2 = ax1.twinx()
        ax1_2.plot(self.remaped_qsi_image_left[int(self.images_left.shape[1] / 2), :], color='red')
        ax1_2.set_ylabel('Remaped QSI Image left', color='red')
        ax1_2.tick_params(axis='y', labelcolor='red')
        ax1.grid(True)
        # Right Abs Phi Image 1D
        ax2.plot(abs_phi_image_right[int(self.images_right.shape[1] / 2), :], color='gray')
        ax2.set_title('Abs Phi Image right 1D')
        ax2.set_ylabel('Abs Phi Image right')

        ax2_2 = ax2.twinx()
        ax2_2.plot(self.remaped_qsi_image_right[int(self.images_right.shape[1] / 2), :], color='red')
        ax2_2.set_ylabel('Remaped QSI Image right', color='red')
        ax2_2.tick_params(axis='y', labelcolor='red')
        ax2.grid(True)

        # Left Abs Phi Image 2D
        im3 = ax3.imshow(abs_phi_image_left, cmap='gray')
        ax3.set_title('Abs Phi Image left 2D')
        plt.colorbar(im3, ax=ax3)

        # Right Abs Phi Image 2D
        im4 = ax4.imshow(abs_phi_image_right, cmap='gray')
        ax4.set_title('Abs Phi Image right 2D')
        plt.colorbar(im4, ax=ax4)

        # Title for the whole figure
        fig.suptitle('Fase absoluta')
        plt.tight_layout()


        # Qsi and remapped QSI
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 8))

        # Left Abs Phi Image 2D
        im1 = ax1.imshow(self.qsi_image_left, cmap='gray')
        ax1.set_title('Abs Phi Image left 2D')
        plt.colorbar(im1, ax=ax1)

        # Right Abs Phi Image 2D
        im2 = ax2.imshow(self.qsi_image_right, cmap='gray')
        ax2.set_title('Abs Phi Image left 2D')
        plt.colorbar(im2, ax=ax2)



        # Left Abs Phi Image 2D
        im3 = ax3.imshow(self.remaped_qsi_image_left, cmap='gray')
        ax3.set_title('Abs Phi Image left 2D')
        plt.colorbar(im3, ax=ax3)

        # Right Abs Phi Image 2D
        im4 = ax4.imshow(self.remaped_qsi_image_right, cmap='gray')
        ax4.set_title('Abs Phi Image right 2D')
        plt.colorbar(im4, ax=ax4)

        # Title for the whole figure
        fig.suptitle('Qsi & Remaped QSI')
        plt.tight_layout()
        plt.show()
