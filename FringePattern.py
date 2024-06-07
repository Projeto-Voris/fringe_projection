import numpy as np
import cv2
import matplotlib.pyplot as plt

class FringePattern:
    def __init__(self, width=int, height=int, f_sin=float):
        self.f_sin = f_sin
        self.f_sin_px = 2 * np.pi / float(f_sin)
        self.sin_values =[]
        self.image = np.ones((height, width, 1), dtype =np.uint8)
    def show_image(self):
        cv2.imshow('Image', self.image)
        cv2.resizeWindow('Image', (1000, 500))
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
        self.sin_values = self.sin_values + 1
        y = np.sin(2 * np.pi * float(self.f_sin_px) * x / self.image.shape[1])
        plt.plot(x, self.sin_values)
        plt.show()
        for i in range(self.image.shape[1]):
            self.image[:, i] = ((self.sin_values[i])*255/2).astype(np.uint8)