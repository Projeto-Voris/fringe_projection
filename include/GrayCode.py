import numpy as np
import cv2


class GrayCode:
    def __init__(self, resolution=(512, 512), n_bits=4, axis=0, px_f=16):
        self.width = resolution[0]
        self.height = resolution[1]
        self.px_f = px_f
        self.n_bits = n_bits  # Number of bits for the Gray code, adjusted to start from 0
        self.graycode_h_seq = []  # List to store horizontal Gray code sequences
        self.graycode_v_seq = []  # List to store vertical Gray code sequences
        self.image_seq_h = np.zeros((resolution[1], resolution[0], self.n_bits + 2),
                                    dtype=np.uint8)  # Horizontal Gray code images
        self.image_seq_v = np.zeros((resolution[1], resolution[0], self.n_bits + 2),
                                    dtype=np.uint8)  # Vertical Gray code images
        self.gc_images = np.zeros((self.height, self.width, self.n_bits + 2), np.uint8)  # Combined image sequence
        # self.create_images(axis=axis)  # Generate the Gray code images based on the specified axis
        self.create_graycode_images()

    def get_gc_images(self):
        return self.gc_images  # Return the generated images

    def show_image(self):  # reading last shape of vector image
        for i in range(self.gc_images.shape[2]):
            print(i)
            cv2.imshow('Image', self.gc_images[:, :, i])
            cv2.waitKey(0)
        cv2.destroyAllWindows()

    def create_graycode_images(self):
        n_bits = np.ceil(np.log2(self.width / self.px_f / 2))
        width = [element for element in np.arange(2 ** self.n_bits, dtype=np.uint8) for _ in range(int(self.px_f / 2))]
        width_b = self.list_to_graycode_binary(width, self.n_bits)
        self.gc_images[:, :, 0] = 255

        print('ok')
        for j in range(self.width):
            for i in range(self.n_bits):
                self.gc_images[:, j, i + 2] = int(list(width_b[j])[i]) * 255
        print('wait')

    def list_to_graycode_binary(self, int_list, bit_length):
        graycode_list = []
        for n in int_list:
            graycode = n ^ (n >> 1)
            # Convert to binary string with leading zeros based on the bit length
            graycode_binary = format(graycode, f'0{bit_length}b')
            graycode_list.append(graycode_binary)
        return graycode_list

    def get_gc_order_v(self):
        # Converter a porção relevante da imagem para valores binários
        bit_values = (self.gc_images[0, :, 2:] / 255).astype(int)
        # Converter os valores binários em inteiros
        int_values = np.dot(bit_values, 2 ** np.arange(bit_values.shape[-1])[::-1])
        # Obter valores únicos e seus índices
        qsi_val, indices = np.unique(int_values, return_index=True)

        # Combinar os valores únicos e índices em uma lista de tuplas e ordenar
        sorted_indices = np.argsort(indices)
        sorted_qsi_val = qsi_val[sorted_indices]

        return sorted_qsi_val
