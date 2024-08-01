import numpy as np
import matplotlib.pyplot as plt
import FringePattern
import GrayCode

if __name__ == '__main__':
    # gera imagem
    img_resolution = (1024, 1024)
    fringe = FringePattern.FringePattern(resolution=img_resolution, f_sin=3, steps=4)
    fringe.create_fringe_image()
    image = fringe.get_image()
    graycode = GrayCode.GrayCode(resolution=img_resolution, n_bits=4)
    graycode_image = np.invert(graycode.get_images())
    graycode.show_image()

    # Determina tamanho da imagem
    height, width, channels = image.shape
    n = channels

    # Inicializar a imagem de saída
    phi_image = np.zeros((height, width))
    qsi_image = np.zeros((height, width))
    phi_image_A = np.zeros((height, width))

    # Calcular Phi para cada pixel
    for u in range(height):
        for v in range(width):
            sum_sin = sum(image[u, v, i] * np.sin(2 * np.pi * (i + 1) / n) for i in range(n))
            sum_cos = sum(image[u, v, i] * np.cos(2 * np.pi * (i + 1) / n) for i in range(n))
            qsi_image[u, v] = int(''.join(str(int(a)) for a in graycode_image[u, v, 2:]/255), 2)
            phi_image[u, v] = np.arctan2(-sum_sin, sum_cos)
            if phi_image[u, v] <= -np.pi/2:
                phi_image_A[u, v] = phi_image[u, v] + 2 * np.pi * (qsi_image[u, v] + 1) / 2 + np.pi
            elif -np.pi/2 < phi_image[u, v] < np.pi/2:
                phi_image_A[u, v] = phi_image[u, v] + 2 * np.pi * (qsi_image[u, v]/2) + np.pi
            elif phi_image[u, v] >= np.pi/2:
                phi_image_A[u, v] = phi_image[u, v] + 2 * np.pi * (((qsi_image[u, v] + 1)/2) - 1) + np.pi

    plt.subplot(1, 2, 1)
    plt.plot(phi_image_A[1, :], color='gray')

    plt.subplot(1, 2, 2)
    plt.plot(phi_image[1, :], color='gray')
    plt.show()
