import numpy as np
import matplotlib.pyplot as plt
from include import FringePattern, GrayCode

def calculate_phi(image):
    height, width, channels = image.shape
    n = channels

    # Calcular os valores de seno e cosseno para todos os canais de uma vez
    indices = np.arange(1, n + 1)
    sin_values = np.sin(2 * np.pi * indices / n)
    cos_values = np.cos(2 * np.pi * indices / n)

    # Multiplicar a imagem pelos valores de seno e cosseno
    sin_contributions = image * sin_values
    cos_contributions = image * cos_values

    # Somar as contribuições ao longo dos canais
    sum_sin = np.sum(sin_contributions, axis=2)
    sum_cos = np.sum(cos_contributions, axis=2)

    # Calcular Phi para cada pixel
    phi_image = np.arctan2(-sum_sin, sum_cos)

    return phi_image


def calculate_qsi(graycode_image):
    height, width, _ = graycode_image.shape

    # Converter os valores relevantes da imagem graycode para inteiros
    bit_values = (graycode_image[:, :, 2:] / 255).astype(int)

    # Converter cada linha de bits em um único número inteiro
    qsi_image = np.dot(bit_values, 2 ** np.arange(bit_values.shape[-1])[::-1])

    return qsi_image


def remap_qsi_image(qsi_image, real_qsi_order):
    # Cria um mapeamento dos valores originais para os novos índices
    value_to_new_index_map = {value: new_index for new_index, value in enumerate(real_qsi_order)}

    # Mapeia os valores da qsi_image usando numpy para operações vetorizadas
    remapped_qsi_image = np.vectorize(value_to_new_index_map.get)(qsi_image)

    return remapped_qsi_image

def inv_power(x):
    count = 0
    while x != 0:
        x = x // 2
        count += 1
    return count

if __name__ == '__main__':
    # Parametro inicialziação
    img_resolution = (1024, 1024)
    n_franjas = 16
    max_bits = n_franjas + 1
    n_shift = 4

    # Inicializa imagens franja e graycode
    fringe = FringePattern.FringePattern(resolution=img_resolution, f_sin=n_franjas, steps=n_shift)
    graycode = GrayCode.GrayCode(resolution=img_resolution, n_bits=4)

    # recebe em alguma variavem as imagens
    fringe_images = fringe.get_image()
    graycode_image = graycode.get_images()
    real_qsi_order = graycode.get_gc_order_v()
    fringe.show_image()
    graycode.show_image()

    # Calcula a imagem Phi
    phi_image = calculate_phi(fringe_images)

    # Calcula a imagem QSI
    qsi_image = calculate_qsi(graycode_image)
    remap_qsi_image = remap_qsi_image(qsi_image, real_qsi_order)
    abs_phi_image = np.zeros(phi_image.shape)

    for i in range(phi_image.shape[0]):
        for j in range(phi_image.shape[1]):
            if phi_image[i, j] <= -np.pi / 2:
                abs_phi_image[i, j] = phi_image[i, j] + 2 * np.pi * remap_qsi_image[i, j]
            elif -np.pi / 2 < phi_image[i, j] < np.pi / 2:
                abs_phi_image[i, j] = phi_image[i, j] + 2 * np.pi * remap_qsi_image[i, j]
            elif phi_image[i, j] >= np.pi / 2:
                abs_phi_image[i, j] = phi_image[i, j] + 2 * np.pi * remap_qsi_image[i, j]

    plt.subplot(1, 2, 1)
    plt.plot(phi_image[1, :], color='gray')

    plt.subplot(1, 2, 2)
    plt.plot(abs_phi_image[1, :], color='gray')

    # plt.subplot(1, 2, 3)
    # plt.plot(remap_qsi_image[1, :], color='gray')

    plt.figure()
    plt.subplot(2, 2, 1)
    plt.imshow(phi_image, cmap='gray')
    plt.colorbar()
    plt.subplot(2, 2, 2)
    plt.imshow(abs_phi_image, cmap='gray')
    plt.colorbar()
    plt.show()
