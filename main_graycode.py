import numpy as np
import cv2
import GrayCode

if __name__ == '__main__':
    img_resolution = (512, 512)
    graycode = GrayCode.GrayCode(resolution=img_resolution, n_bits=4)
    col_graycode = graycode.graycode_pattern(resolution=graycode.height, proj_n=graycode.col_prj_n)*255
    # row_graycode = graycode.graycode_pattern(resolution=graycode.width, proj_n=graycode.row_prj_n)
    for i in range(col_graycode.shape[2]):
        cv2.imshow('graycode', cv2.bitwise_not(col_graycode[:, :, i]))
        cv2.waitKey(0)
    # concat = graycode.concatenate()
    # for k in range(concat.shape[2]):
    #     cv2.imshow('Concatenate', concat[:, :, k])
    #     cv2.waitKey(0)
