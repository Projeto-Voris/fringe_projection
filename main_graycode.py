import cv2
import GrayCode

if __name__ == '__main__':
    img_resolution = (1920, 1024)
    graycode = GrayCode.GrayCode(resolution=img_resolution, n_bits=8, axis=0)
    images = graycode.get_images()
    for i in range(images.shape[2]):
        print('Image: {}'.format(i))
        cv2.imshow('gray_h', images[:, :, i])
        cv2.waitKey(0)

