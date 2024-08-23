import numpy as np
import cv2
import matplotlib.pyplot as plt
class FringePattern:

    def __init__(self, resolution=(800, 600), px_f=20, steps=4):
        self.width = resolution[0]
        self.height = resolution[1]
        self.n_fringes = np.floor(self.width / px_f)  # function sine
        self.sin_values = []
        self.steps = steps # number of time steps
        self.fr_images = np.zeros((int(resolution[1]), int(resolution[0]), self.steps), dtype=np.uint8) # vector image
        self.create_fringe_image()

    def get_steps(self):
        return self.steps
    def show_image(self): # reading last shape of vector image
        for i in range(self.fr_images.shape[2]):
            cv2.imshow('Image', self.fr_images[:, :, i])
            cv2.waitKey(0)

    def print_image(self): # reading shape 0 e 1
        for i in range(self.fr_images.shape[0]):
            for j in range(self.fr_images.shape[1]):
                print(self.fr_images[i, j], end='')
            print('\n')
        print("finshed")

    def create_fringe_image(self):
        x = np.arange(self.fr_images.shape[1])
        for n in range(self.steps): # phase shift of n=4
            phase_shift = n * 2 * np.pi/self.steps
            y = np.sin(2 * np.pi * float(self.n_fringes) * x / self.fr_images.shape[1] + phase_shift + np.pi/2) + 1
            self.sin_values.append(y)
        for k in range(len(self.sin_values)):
            for i in range(self.fr_images.shape[0]):
                self.fr_images[i, :, k] = (self.sin_values[k]) * 255 / 2

    def get_fr_image(self):
        return self.fr_images
