import numpy as np
import cv2
import matplotlib.pyplot as plt
from numpy import matlib


class GrayCode:
    def __init__(self, resolution=(800, 600), n_bits=4):
        self.width = resolution[0]
        self.height = resolution[1]
        self.n_bits = n_bits
        self.col_prj_n = np.ceil(np.log2(self.width))
        self.row_prj_n = np.ceil(np.log2(self.height))
        # self.ref_pattern = np.concatenate(np.ones((self.height, self.width), dtype=np.uint8),
        #                                   np.zeros((self.height, self.width), dtype=np.uint8))

    def grays(self, n_bits):
        if int(n_bits) == 1 or n_bits != round(n_bits, 1) or n_bits > 26:
            raise ValueError("Number of bits must be between 1 and 26")
            # print('error')
        gray_code = np.zeros(int(np.power(2, n_bits)), np.uint8)
        gray_code[1] = 1
        T = 2
        for k in range(1, int(n_bits)):
            T2 = T + T
            gray_code[T + 0:T2] = T + np.flip(gray_code[0:T], axis=0)
            T = T2
        return gray_code

    def graycode_pattern(self, resolution, proj_n):
        gray_de = self.grays(proj_n)
        mat = np.transpose(np.flip(np.transpose(((gray_de[:, None] & (1 << np.arange(int(proj_n)))) > 0).astype(int))))
        pattern_seq = np.zeros((self.height, self.width, int(2 * proj_n)), dtype=np.uint8)
        for i in range(int(self.col_prj_n)):
            mat2 = (np.tile(mat[:, i], (resolution, 1)))
            # mat2 = np.transpose(np.resize(mat2, (self.width, self.height)))
            pattern_seq[:, :, i] = mat2
            temp = pattern_seq[:, :, i]
            pattern_seq[:, :, i] = np.ones(temp.shape, dtype=np.uint8) - temp
        pattern_seq[:, int(self.width / 2):, 0] = 0
        return pattern_seq[:, :, :self.n_bits]

    # def concatenate(self):
    #     col = self.graycode_pattern(self.width, self.col_prj_n)
    #     row = self.graycode_pattern(self.height, self.row_prj_n)
    #     return np.concatenate((self.ref_pattern, col, row), axis=2)
