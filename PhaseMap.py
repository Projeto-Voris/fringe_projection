import numpy as np
import matplotlib.pyplot as plt
import FringePattern

if __name__ == '__main__':
    img_resolution = (1024, 1024)
    fringe = FringePattern.FringePattern(resolution=img_resolution, f_sin=1, steps=4)
    fringe.create_fringe_image()
    image = fringe.get_image()

    # Determina tamanho da imagem
    height, width, channels = image.shape
    n = channels

    # Inicializar a imagem de saída
    phi_image = np.zeros((height, width))

    x = np.arange(image.shape[1])
    # Calcular Phi para cada pixel
    for u in range(height):
        for v in range(width):
            sum_sin = sum(image[u, v, i] * np.sin(2 * np.pi * (i + 1) / n) for i in range(n))
            sum_cos = sum(image[u, v, i] * np.cos(2 * np.pi * (i + 1) / n) for i in range(n))

            phi_image[u, v] = np.arctan2(-sum_sin, sum_cos)

    plt.plot(x, phi_image)
    plt.show()