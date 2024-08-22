import numpy as np
import cv2
import matplotlib.pyplot as plt
class FringePattern:

    def __init__(self, resolution=(800,600), f_sin=2, steps=4):
        self.width = resolution[0]
        self.height = resolution[1]
        self.f_sin = f_sin # function sine
        self.sin_values = []
        self.steps = steps # number of time steps
        self.f_images = np.zeros((int(resolution[1]), int(resolution[0]), self.steps), dtype=np.uint8) # vector image
        self.create_fringe_image()

    def get_steps(self):
        return self.steps
    def show_image(self): # reading last shape of vector image
        for i in range(self.f_images.shape[2]):
            cv2.imshow('Image', self.f_images[:, :, i])
            cv2.waitKey(0)

    def print_image(self): # reading shape 0 e 1
        for i in range(self.f_images.shape[0]):
            for j in range(self.f_images.shape[1]):
                print(self.f_images[i, j], end='')
            print('\n')
        print("finshed")

    def create_fringe_image(self):
        x = np.arange(self.f_images.shape[1])
        for n in range(self.steps): # phase shift of n=4
            phase_shift = n * 2 * np.pi/4
            y = np.sin(2 * np.pi * float(self.f_sin) * x / self.f_images.shape[1] + phase_shift) + 1
            self.sin_values.append(y)
            # plt.plot(x, y)
        # plt.show()
        for k in range(len(self.sin_values)):
            for i in range(self.f_images.shape[0]):
                self.f_images[i, :, k] = (self.sin_values[k]) * 255 / 2

    def get_fr_image(self):
        return self.f_images