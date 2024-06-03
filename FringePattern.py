import numpy as np
import cv2
import math
import matplotlib.pyplot as plt
# import fringes as frng


class FringePattern:
    def __init__(self, width=int, height=int, f_sin=float, steps=int):
        self.f_sin = f_sin
        self.f_sin_px = 2 * np.pi / float(f_sin)
        self.sin_values = []
        self.steps = 2 * np.pi / float(steps)
        self.image = np.ones((height, width, 1), dtype=np.uint8)
        # self.f = frng.Fringes()  # instantiate class


    # def fringe_params(self):

    def show_image(self):
        cv2.imshow('Image', self.image)
        cv2.waitKey(0)

    def print_image(self):
        for i in range(self.image.shape[0]):
            for j in range(self.image.shape[1]):
                print(self.image[i, j], end='')
            print('\n')
        print("finshed")

    def create_fringe_image(self):
        x = np.arange(self.image.shape[1])
        self.sin_values = np.sin(2 * np.pi * float(self.f_sin) * x / self.image.shape[1])
        self.sin_values = self.sin_values
        y = np.sin(2 * np.pi * float(self.f_sin_px) * x / self.image.shape[1])
        plt.plot(x, self.sin_values)
        plt.plot(x, y)
        plt.show()
        # for j in range(self.image.shape[1]):
        #     for i in range(self.image.shape[0]):
        #         self.image[i][j] = self.image[i][j] * y[j]
        for i in range(self.image.shape[1]):
            self.image[:, i] = ((self.sin_values[i])*255 -255).astype(np.uint8)

    # def encode_fringes_image(self):
