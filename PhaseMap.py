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


def calculate_qsi(graycode_image, white_value):
    height, width, _ = graycode_image.shape

    # Converter os valores relevantes da imagem graycode para inteiros
    bit_values = (graycode_image[:, :, 2:] / white_value).astype(int)

    # Converter cada linha de bits em um único número inteiro
    qsi_image = np.dot(bit_values, 2 ** np.arange(bit_values.shape[-1])[::-1])

    return qsi_image


def remap_qsi_image(qsi_image, real_qsi_order):
    # Cria um mapeamento dos valores originais para os novos índices
    value_to_new_index_map = {value: new_index for new_index, value in enumerate(real_qsi_order)}

    # Mapeia os valores da qsi_image usando numpy para operações vetorizadas
    remapped_qsi_image = np.vectorize(value_to_new_index_map.get)(qsi_image)

    return remapped_qsi_image



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
    remaped_qsi_image = remap_qsi_image(qsi_image, real_qsi_order)

    # Calcula a fase absoluta
    abs_phi_image = phi_image + 2 * np.pi * remaped_qsi_image

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 8))

    # First subplot with dual y-axes
    ax1.plot(phi_image[600, 500:1000], color='gray')
    ax1.set_ylabel('Phi Image', color='gray')
    ax1.tick_params(axis='y', labelcolor='gray')

    ax1_2 = ax1.twinx()
    ax1_2.plot(remaped_qsi_image[600, :], color='red')
    ax1_2.set_ylabel('Remaped QSI Image', color='red')
    ax1_2.tick_params(axis='y', labelcolor='red')

    # Second subplot
    ax2.plot(abs_phi_image[600, :], color='gray')
    ax2.set_title('Abs Phi Image 1D')
    ax2.set_ylabel('Abs Phi Image')

    # Third subplot with an image
    im3 = ax3.imshow(phi_image, cmap='gray')
    ax3.set_title('Phi Image 2D')
    fig.colorbar(im3, ax=ax3)

    # Fourth subplot with an image
    im4 = ax4.imshow(abs_phi_image, cmap='gray')
    ax4.set_title('Abs Phi Image 2D')
    fig.colorbar(im4, ax=ax4)

    # Add a title for the whole figure
    fig.suptitle('Fase e Fase absoluta')

# Show the plot
plt.tight_layout()
plt.show()