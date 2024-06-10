import numpy as np
import cv2
import matplotlib.pyplot as plt


class FringePattern:
    def __init__(self, width=800, height=600, f_sin=2, steps=4):
        self.f_sin = f_sin
        self.sin_values = []
        self.freq = steps
        self.image = np.zeros((int(height), int(width), self.freq), dtype=np.uint8)
        # self.image = np.zeros((int(height), int(width), 1), dtype=np.uint8)

    def show_image(self):
        for i in range(self.image.shape[2]):
            cv2.imshow('Image', self.image[:,:,i])
            cv2.waitKey(0)

    def print_image(self):
        for i in range(self.image.shape[0]):
            for j in range(self.image.shape[1]):
                print(self.image[i, j], end='')
            print('\n')
        print("finshed")

    def create_fringe_image(self):
        x = np.arange(self.image.shape[1])
        for n in range(self.freq):
            phase_shift = n * 2 * np.pi/4
            y = np.sin(2 * np.pi * float(self.f_sin) * x / self.image.shape[1] + phase_shift) + 1
            self.sin_values.append(y)
            plt.plot(x, y)
        plt.show()
        for k in range(len(self.sin_values)):
            for i in range(self.image.shape[0]):
                self.image[i,:,k] = (self.sin_values[k]) * 127