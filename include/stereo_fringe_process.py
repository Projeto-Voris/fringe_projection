import numpy as np
import matplotlib.pyplot as plt
import math
import cv2
# Import classes
from include.FringePattern import FringePattern
from include.GrayCode import GrayCode


class Stereo_Fringe_Process(GrayCode, FringePattern):
    def __init__(self, img_resolution=(1024, 768), camera_resolution=(1600, 1200), px_f=16, steps=4):
        """
            Inicializa uma instância da classe com parâmetros específicos de resolução e configuração.
            Este método inicializa as variáveis necessárias para o processamento de imagens, incluindo
            imagens remapeadas e de fase para a esquerda e direita, bem como imagens capturadas pela câmera.
        """
        self.remaped_qsi_image_left = np.zeros(img_resolution, np.uint8)
        self.remaped_qsi_image_right = np.zeros(img_resolution, np.uint8)
        self.qsi_image_left = np.zeros(img_resolution, np.uint8)
        self.qsi_image_right = np.zeros(img_resolution, np.uint8)
        self.phi_image_left = np.zeros(img_resolution, np.uint8)
        self.phi_image_right = np.zeros(img_resolution, np.uint8)

        self.images_left = np.zeros(
            (camera_resolution[1], camera_resolution[0],
             int(steps + self.min_bits_gc(np.floor(img_resolution[0] / px_f)) + 2)), np.uint8)
        self.images_right = np.zeros(
            (camera_resolution[1], camera_resolution[0],
             int(steps + self.min_bits_gc(np.floor(img_resolution[0] / px_f)) + 2)), np.uint8)
        FringePattern.__init__(self, resolution=img_resolution, px_f=px_f, steps=steps)
        GrayCode.__init__(self, resolution=img_resolution, n_bits=self.min_bits_gc(np.floor(img_resolution[0] / px_f)),
                          px_f=px_f)

    def min_bits_gc(self, x):
        """
            Calcula o número mínimo de bits necessários para representar um número em código Gray.
            Parameters:
            -----------
            x : int
                Número positivo para o qual se deseja calcular o número mínimo de bits necessários.
            Returns:
            --------
            int
                Número mínimo de bits necessários para representar o número `x` em código Gray.
        """
        if x <= 0:
            raise ValueError("Input must be a positive integer.")
        return math.ceil(math.log2(x) + 1)

    def normalize_white(self, mask_left, mask_right):
        """
            Calcula a média dos valores máximos dos pixels brancos nas imagens esquerda e direita usando máscaras.

            Parameters:
            -----------
            mask_left : numpy.ndarray
                Máscara binária aplicada à imagem esquerda, onde os pixels de interesse são marcados com o valor 255.

            mask_right : numpy.ndarray
                Máscara binária aplicada à imagem direita, onde os pixels de interesse são marcados com o valor 255.

            Returns:
            --------
            media_branco_max_left : float
                Média dos valores máximos dos pixels brancos na imagem esquerda, calculada a partir da máscara.

            media_branco_max_right : float
                Média dos valores máximos dos pixels brancos na imagem direita, calculada a partir da máscara.
        """
        # Assuming self.mask_left and self.mask_right are the masks for left and right images
        # They should have the same dimensions as the respective images

        media_branco_max_left = np.mean(self.images_left[:, :, self.steps][mask_left == 255])
        media_branco_max_right = np.mean(self.images_right[:, :, self.steps][mask_right == 255])

        print("media dos brancos right:", media_branco_max_right)

        return media_branco_max_left, media_branco_max_right

    def set_images(self, image_left, image_right, counter):
        """
            Atribui imagens para os índices especificados nas matrizes de imagens esquerda e direita.
        """
        self.images_left[:, :, counter] = image_left
        self.images_right[:, :, counter] = image_right

    def calculate_phi_images(self):
        """
            Calcula as imagens de fase (Phi) para as imagens esquerda e direita.
        """
        self.phi_image_left = self.calculate_phi(self.images_left[:, :, :int(FringePattern.get_steps(self))])
        self.phi_image_right = self.calculate_phi(self.images_right[:, :, :int(FringePattern.get_steps(self))])
        # return self.phi_image_left, self.phi_image_right

    def calculate_qsi_images(self):
        """
            Calcula as imagens QSI (Quantitative Phase Shift Imaging) para as imagens esquerda e direita.
        """
        self.qsi_image_left = self.calculate_qsi(self.images_left[:, :, FringePattern.get_steps(self):])
        self.qsi_image_right = self.calculate_qsi(self.images_right[:, :, FringePattern.get_steps(self):])
        # return qsi_image_left, qsi_image_right

    def calculate_remaped_qsi_images(self):
        """
            Calcula as imagens QSI remapeadas para as imagens esquerda e direita.
        """
        self.remaped_qsi_image_left = self.remap_qsi_image(self.qsi_image_left, GrayCode.get_gc_order_v(self))
        self.remaped_qsi_image_right = self.remap_qsi_image(self.qsi_image_right, GrayCode.get_gc_order_v(self))
        # return self.remaped_qsi_image_left, self.remaped_qsi_image_right

    def calculate_abs_phi_images(self):
        """
            Calcula as imagens de fase absoluta (phi) para os conjuntos de imagens esquerda e direita.

            Esta função gera as imagens de fase absoluta `abs_phi_image_left` e `abs_phi_image_right` a partir das imagens
            de fase `phi_image_left` e `phi_image_right`, bem como das imagens de QSI remapeadas correspondentes. As imagens
            de fase absoluta são calculadas considerando as diferentes condições de fase em relação a -π/2 e π/2, aplicando
            correções baseadas nos valores remapeados de QSI.

            O método é utilizado para garantir que as fases calculadas estejam em um intervalo contínuo e coerente para
            processamento subsequente, como na análise de padrões de fase ou reconstrução 3D.

            Parameters:
            -----------
            Não possui parâmetros de entrada.

            Returns:
            --------
            abs_phi_image_left : np.ndarray
                Uma matriz NumPy bidimensional representando a imagem de fase absoluta correspondente à imagem
                `phi_image_left`. Os valores da fase estão em radianos.

            abs_phi_image_right : np.ndarray
                Uma matriz NumPy bidimensional representando a imagem de fase absoluta correspondente à imagem
                `phi_image_right`. Os valores da fase estão em radianos.
            """
        abs_phi_image_left = np.zeros((self.images_left.shape[0], self.images_left.shape[1]), np.float32)
        abs_phi_image_right = np.zeros((self.images_right.shape[0], self.images_right.shape[1]), np.float32)
        # self.abs_phi_image_left = self.phi_image_left + 2 * np.pi * np.ceil(self.remaped_qsi_image_left / 2)
        # self.abs_phi_image_right = self.phi_image_right + 2 * np.pi * np.ceil(self.remaped_qsi_image_right / 2)

        # abs_phi_image_right = np.unwrap(abs_phi_image_right)

        # Condição para a imagem esquerda
        mask_left1 = self.phi_image_left <= -np.pi / 2
        mask_left2 = (self.phi_image_left > -np.pi / 2) & (self.phi_image_left < np.pi / 2)
        mask_left3 = self.phi_image_left >= np.pi / 2

        abs_phi_image_left = np.zeros_like(self.phi_image_left)

        abs_phi_image_left[mask_left1] = self.phi_image_left[mask_left1] + 2 * np.pi * np.floor(
            (self.remaped_qsi_image_left[mask_left1] + 1) / 2) + np.pi

        abs_phi_image_left[mask_left2] = self.phi_image_left[mask_left2] + 2 * np.pi * np.floor(
            self.remaped_qsi_image_left[mask_left2] / 2) + np.pi

        abs_phi_image_left[mask_left3] = self.phi_image_left[mask_left3] + 2 * np.pi * (
                np.floor((self.remaped_qsi_image_left[mask_left3] + 1) / 2) - 1) + np.pi

        # Condição para a imagem direita
        mask_right1 = self.phi_image_right <= -np.pi / 2
        mask_right2 = (self.phi_image_right > -np.pi / 2) & (self.phi_image_right < np.pi / 2)
        mask_right3 = self.phi_image_right >= np.pi / 2

        abs_phi_image_right = np.zeros_like(self.phi_image_right)

        abs_phi_image_right[mask_right1] = self.phi_image_right[mask_right1] + 2 * np.pi * np.floor(
            (self.remaped_qsi_image_right[mask_right1] + 1) / 2) + np.pi

        abs_phi_image_right[mask_right2] = self.phi_image_right[mask_right2] + 2 * np.pi * np.floor(
            self.remaped_qsi_image_right[mask_right2] / 2) + np.pi

        abs_phi_image_right[mask_right3] = self.phi_image_right[mask_right3] + 2 * np.pi * (
                np.floor((self.remaped_qsi_image_right[mask_right3] + 1) / 2) - 1) + np.pi

        min_value = np.min(abs_phi_image_left)
        max_value = np.max(abs_phi_image_left)
        abs_phi_image_left_remaped = 255 * (abs_phi_image_left - min_value) / (max_value - min_value)

        min_value_r = np.min(abs_phi_image_right)
        max_value_r = np.max(abs_phi_image_right)
        abs_phi_image_right_remaped = 255 * (abs_phi_image_right - min_value_r) / (max_value_r - min_value_r)

        return abs_phi_image_left_remaped, abs_phi_image_right_remaped

    def apply_mask_otsu_threshold(self, image):
        """
        Aplica Otsu thresholding a cada uma das camadas de uma imagem de múltiplos canais (franjas)
        e gera uma máscara para separar franjas do fundo.
        Parameters:
        -----------
        image : np.ndarray
            A imagem com múltiplos canais, onde cada canal é uma imagem com franjas mudando de fase.
        max_value : int
            Valor máximo a ser atribuído aos pixels acima do limiar (geralmente 255).
        Returns:
        --------
        mask : np.ndarray
            A máscara binária resultante após aplicar Otsu thresholding.
        """
        # Inicializa a máscara com zeros
        mask = np.zeros(image.shape[:2], dtype=np.uint8)

        # Aplicar Otsu thresholding a cada camada individualmente
        for i in range(image.shape[2]):
            gray_image = image[:, :, i]
            _, thresh = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            mask = np.maximum(mask, thresh)  # Combinar os thresholdings das camadas para criar a máscara

        # Expandir a máscara para o número de canais da imagem original
        mask_expanded = np.stack([mask] * image.shape[2], axis=-1)

        # Aplicar a máscara à imagem original
        masked_image = cv2.bitwise_and(image, image, mask=mask_expanded[:, :, 0])  # Usar apenas a 1ª camada da máscara

        return masked_image

    def calculate_phi(self, image):
        """
            Calcula a imagem de fase (phi) a partir de uma imagem de múltiplos canais utilizando transformações senoidais e cossenoidais.

            Esta função processa uma imagem composta por múltiplos canais (por exemplo, imagens obtidas através de projeções de padrões de fase)
            e calcula a fase correspondente para cada pixel. O cálculo é realizado aplicando funções seno e cosseno aos canais da imagem
            e combinando esses valores para obter a fase através da função arctan2.

            Parameters:
            -----------
            image : np.ndarray
                Uma matriz NumPy tridimensional (altura, largura, canais), onde cada canal representa uma amostra
                de fase em diferentes momentos ou ângulos. A função espera que os canais sejam organizados em uma
                sequência de fases.

            Returns:
            --------
            phi_image : np.ndarray
                Uma matriz NumPy bidimensional representando a imagem Phi. Cada valor de pixel na imagem corresponde
                ao ângulo Phi calculado para aquele pixel com base nas contribuições de seno e cosseno dos canais.
            """

        # Aplicar a máscara com o otsu threshold
        masked_image = self.apply_mask_otsu_threshold(image)

        # Calcular os valores de seno e cosseno para todos os canais de uma vez
        # indices = np.arange(1, image.shape[2]+1)
        # angle = 2 * np.pi * indices / image.shape[2]
        indices = np.arange(1, masked_image.shape[2] + 1)
        angle = 2 * np.pi * indices / masked_image.shape[2]

        sin_values = np.sin(angle)
        cos_values = np.cos(angle)

        # Multiplicar a imagem pelos valores de seno e cosseno
        # sin_contributions = np.sum(image * sin_values, axis=2)
        # cos_contributions = np.sum(image * cos_values, axis=2)
        sin_contributions = np.sum(masked_image * sin_values, axis=2)
        cos_contributions = np.sum(masked_image * cos_values, axis=2)

        # Calcular Phi para cada pixel
        phi_image = np.arctan2(-sin_contributions, cos_contributions)

        return phi_image

    def calculate_qsi(self, graycode_image):
        """
            Calcula a imagem QSI (Quantitative Structure Image) a partir de uma imagem codificada em graycode.

            A função converte uma imagem de código gray em uma imagem QSI. A imagem de entrada deve ter várias camadas,
            onde a primeira camada é a referência de branco e as camadas subsequentes contêm os bits do código gray.
            A função normaliza os valores de bits com relação à camada de referência de branco e, em seguida, converte
            cada conjunto de bits em um único número inteiro, gerando a imagem QSI.

            Parameters:
            -----------
            graycode_image : np.ndarray
                Uma matriz NumPy de três dimensões (altura, largura, camadas) representando a imagem de código gray.
                A primeira camada(0) contém os valores de referência de branco, enquanto as camadas subsequentes contêm
                os bits do código gray.

            Returns:
            --------
            qsi_image : np.ndarray
                Uma matriz NumPy bidimensional representando a imagem QSI calculada. Cada pixel na imagem QSI corresponde
                a um valor inteiro derivado dos bits de código gray.
            """

        # Obter o valor de branco para cada pixel (shape (X, Y))
        white_value = graycode_image[:, :, 0].astype(float)
        white_value = np.maximum(white_value, 1e-6)

        # Comparar os bits relevantes com o branco correspondente
        bit_values = graycode_image[:, :, 2:] / white_value[:, :, None]
        bit_values = (bit_values > 0.8).astype(int)
        # bit_values = (bit_values > 0.5).astype(int)

        # Converter cada linha de bits em um único número inteiro
        qsi_image = np.dot(bit_values, 2 ** np.arange(bit_values.shape[-1])[::-1])

        return qsi_image

    def save_array_to_csv(self, array, filename):
        """
        Save a 2D NumPy array to a CSV file.

        :param array: 2D numpy array
        :param filename: Output CSV filename
        """
        # Save the 2D array as a CSV file
        np.savetxt(filename, array, delimiter=',')
        print(f"Array saved to {filename}")

    def remap_qsi_image(self, qsi_image, real_qsi_order):
        """
        Remapeia os valores de uma imagem QSI de acordo com uma nova ordem QSI real.

        Esta função remapeia os valores da imagem QSI fornecida, utilizando uma ordem QSI real específica.
        O mapeamento é realizado criando um dicionário que associa os valores originais aos novos índices,
        de acordo com a ordem fornecida.

        Parameters:
        -----------
        qsi_image : np.ndarray
            Uma matriz NumPy bidimensional representando a imagem QSI original cujos valores precisam ser remapeados.

        real_qsi_order : list
            Uma lista de inteiros representando a ordem real dos valores QSI. Cada valor nesta lista corresponde
            a um valor original da imagem QSI, e a posição desse valor na lista determina o novo índice a ser
            aplicado.

        Returns:
        --------
        remapped_qsi_image : np.ndarray
            Uma matriz NumPy bidimensional com os valores remapeados de acordo com a nova ordem QSI.
            O resultado mantém a mesma forma da `qsi_image` original, mas com os valores ajustados conforme
            a nova ordem especificada.
        """
        # Cria um mapeamento dos valores originais para os novos índices
        value_to_new_index_map = {value: new_index for new_index, value in enumerate(real_qsi_order)}

        # Mapeia os valores da qsi_image usando numpy para operações vetorizadas
        remapped_qsi_image = np.vectorize(lambda x: value_to_new_index_map.get(x, 0))(qsi_image)

        return remapped_qsi_image

    def plot_1d_phase(self, ax, phi_image, remaped_qsi_image, title, ylabel):
        """
            Esta função cria dois gráficos sobrepostos no mesmo eixo. O primeiro gráfico mostra a imagem de
            fase (`phi_image`) e o segundo gráfico mostra a imagem QSI remapeada (`remaped_qsi_image`). O gráfico
            da imagem QSI remapeada é plotado em um eixo y secundário para permitir uma visualização clara das
            duas séries de dados com diferentes escalas.
        """
        ax.plot(phi_image, color='gray')
        ax.set_ylabel(ylabel, color='gray')
        ax.tick_params(axis='y', labelcolor='gray')
        ax.set_title(title)
        ax.grid(True)

        ax2 = ax.twinx()
        ax2.plot(remaped_qsi_image, color='red')
        ax2.set_ylabel('Remaped QSI Image', color='red')
        ax2.tick_params(axis='y', labelcolor='red')

    def plot_2d_image(self, ax, image, title, cmap='gray'):
        """
            Esta função exibe uma imagem 2D em um eixo específico, com a opção de definir o título e o mapa de
            cores (colormap). Também adiciona uma barra de cores ao lado da imagem para indicar a escala dos valores.
        """
        im = ax.imshow(image, cmap=cmap)
        ax.set_title(title)
        plt.colorbar(im, ax=ax)

    def plot_phase_map(self, name='Plot'):
        """
            Plota gráficos 1D e 2D das imagens de fase e QSI remapeadas.

            Esta função cria uma figura com uma grade 2x2 de subplots. Os gráficos incluem:
            - Gráficos 1D da fase e da imagem QSI remapeada para as imagens esquerda e direita,
              exibidos nos dois primeiros subplots.
            - Imagens 2D da fase para as imagens esquerda e direita, exibidas nos dois últimos subplots.
        """
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        middle_index_left = int(self.images_left.shape[1] / 2)
        middle_index_right = int(self.images_right.shape[1] / 2)

        self.plot_1d_phase(axes[0, 0], self.phi_image_left[middle_index_left, :],
                           self.remaped_qsi_image_left[middle_index_left, :], 'Phi Image left', 'Phi Image left')

        self.plot_1d_phase(axes[0, 1], self.phi_image_right[middle_index_right, :],
                           self.remaped_qsi_image_right[middle_index_right, :], 'Phi Image right', 'Phi Image right')

        self.plot_2d_image(axes[1, 0], self.phi_image_left, 'Phi Image left 2D')
        self.plot_2d_image(axes[1, 1], self.phi_image_right, 'Phi Image right 2D')

        fig.suptitle('Fase franjas - {}'.format(name))
        plt.tight_layout()
        plt.show()

    def save_array_to_csv(self, array, filename):
        """
        Save a 2D NumPy array to a CSV file.

        :param array: 2D numpy array
        :param filename: Output CSV filename
        """
        # Save the 2D array as a CSV file
        np.savetxt(filename, array, delimiter=',')
        print(f"Array saved to {filename}")

    def plot_abs_phase_map(self, name='Plot'):
        """
            Plota gráficos 1D e 2D das imagens de fase absoluta e QSI remapeada.

            Esta função cria uma figura com uma grade 2x2 de subplots para visualizar as imagens de fase
            absoluta e QSI remapeada. Os gráficos incluem:
            - Gráficos 1D das imagens de fase absoluta (`abs_phi_image_left` e `abs_phi_image_right`) e
              das imagens QSI remapeadas (`remaped_qsi_image_left` e `remaped_qsi_image_right`),
              exibidos nos dois primeiros subplots.
            - Imagens 2D das fases absolutas para as imagens esquerda e direita, exibidas nos dois últimos
              subplots.
        """
        abs_phi_image_left, abs_phi_image_right = self.calculate_abs_phi_images()
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))

        # self.save_array_to_csv(abs_phi_image_left, filename='abs_image_left_32_20241016.csv')
        # self.save_array_to_csv(abs_phi_image_right, filename='abs_image_right_32_20241016.csv')

        middle_index_left = int(self.images_left.shape[1] / 2)
        middle_index_right = int(self.images_right.shape[1] / 2)

        self.plot_1d_phase(axes[0, 0], abs_phi_image_left[middle_index_left, :],
                           self.remaped_qsi_image_left[middle_index_left, :], 'Abs Phi Image left 1D', 'Abs Phi Image left')

        self.plot_1d_phase(axes[0, 1], abs_phi_image_right[middle_index_right, :],
                           self.remaped_qsi_image_right[middle_index_right, :], 'Abs Phi Image right 1D', 'Abs Phi Image right')

        self.plot_2d_image(axes[1, 0], abs_phi_image_left, 'Abs Phi Image left 2D')
        # min_value = np.min(abs_phi_image_left)
        # max_value = np.max(abs_phi_image_left)
        # image_remaped = 255 * (abs_phi_image_left - min_value) / (max_value - min_value)
        # image_remaped = image_remaped.astype(np.uint8)
        # cv2.imwrite('abs_phi_image_left_14.png', image_remaped)
        self.plot_2d_image(axes[1, 1], abs_phi_image_right, 'Abs Phi Image right 2D')
        # min_value_r = np.min(abs_phi_image_right)
        # max_value_r = np.max(abs_phi_image_right)
        # image_remaped_r = 255 * (abs_phi_image_right - min_value_r) / (max_value_r - min_value_r)
        # image_remaped_r = image_remaped_r.astype(np.uint8)
        # cv2.imwrite('abs_phi_image_right_14.png', image_remaped_r)

        fig.suptitle('Fase absoluta {}'.format(name))

        plt.tight_layout()
        # plt.savefig('abs_phi_images.png', dpi=300, bbox_inches='tight')
        plt.show()

    def plot_qsi_map(self, name='Plot'):
        """
            Plota gráficos 2D das imagens QSI e QSI remapeada.

            Esta função cria uma figura com uma grade 2x2 de subplots para visualizar as imagens QSI
            e QSI remapeada. Os gráficos incluem:
            - Imagens 2D da QSI para as imagens esquerda e direita, exibidas nos dois primeiros subplots.
            - Imagens 2D da QSI remapeada para as imagens esquerda e direita, exibidas nos dois últimos
              subplots.
        """
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))

        self.plot_2d_image(axes[0, 0], self.qsi_image_left, 'Qsi Image left 2D')
        self.plot_2d_image(axes[0, 1], self.qsi_image_right, 'Qsi Image right 2D')
        self.plot_2d_image(axes[1, 0], self.remaped_qsi_image_left, 'Remaped Qsi Image left 2D')
        self.plot_2d_image(axes[1, 1], self.remaped_qsi_image_right, 'Remaped Qsi Image right 2D')

        fig.suptitle('Qsi & Remaped QSI {}'.format(name))
        plt.tight_layout()
        plt.show()
