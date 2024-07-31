import numpy as np
import cv2


class GrayCode:
    def __init__(self, resolution=(512, 512), n_bits=4, axis=0):
        self.width = resolution[0]
        self.height = resolution[1]
        self.n_bits = n_bits  # Number of bits for the Gray code, adjusted to start from 0
        self.graycode_h_seq = []  # List to store horizontal Gray code sequences
        self.graycode_v_seq = []  # List to store vertical Gray code sequences
        self.image_seq_h = np.zeros((resolution[1], resolution[0], self.n_bits+2),
                                    dtype=np.uint8)  # Horizontal Gray code images
        self.image_seq_v = np.zeros((resolution[1], resolution[0], self.n_bits+2),
                                    dtype=np.uint8)  # Vertical Gray code images
        self.images = None  # Combined image sequence
        self.create_images(axis=axis)  # Generate the Gray code images based on the specified axis

    def create_images(self, axis=0):
        if axis == 0:  # If the axis is horizontal
            self.gray_split(axis=0)  # Generate horizontal Gray code sequence
            self.create_graycode_h_images()  # Create horizontal Gray code images
            self.images = self.image_seq_h  # Set the images attribute to horizontal images

        if axis == 1:  # If the axis is vertical
            self.gray_split(axis=1)  # Generate vertical Gray code sequence
            self.create_graycode_v_images()  # Create vertical Gray code images
            self.images = self.image_seq_v  # Set the images attribute to vertical images

        if axis == 2:  # If both axes are needed
            self.gray_split(axis=0)  # Generate horizontal Gray code sequence
            self.gray_split(axis=1)  # Generate vertical Gray code sequence
            self.create_graycode_v_images()  # Create vertical Gray code images
            self.create_graycode_h_images()  # Create horizontal Gray code images
            self.images = np.concatenate((self.image_seq_h, self.image_seq_v), axis=2)  # Combine both sets of images

    def get_images(self):
        return self.images  # Return the generated images

    def show_image(self):  # reading last shape of vector image
        for i in range(self.images.shape[2]):
            print(i)
            cv2.imshow('Image', np.invert(self.images[:, :, i]))
            cv2.waitKey(0)
        cv2.destroyAllWindows()

    def gray_split(self, axis=0):
        if axis == 0:
            size = self.width  # Use the width for horizontal splitting
            self.graycode_h_seq.append(np.zeros(size, dtype=np.uint8))  # Initialize the horizontal sequence

        else:
            size = self.height  # Use the height for vertical splitting
            self.graycode_v_seq.append(np.zeros(size, dtype=np.uint8))  # Initialize the vertical sequence

        # Check if n_bits is within the valid range and an integer
        if int(self.n_bits) == 1 or self.n_bits != round(self.n_bits, 1) or self.n_bits > 26:
            raise ValueError("Number of bits must be between 1 and 26")
        # if size % (self.n_bits) != 0:  # Ensure the size is divisible by the number of bits
        #     raise ValueError("Size must be divisible by number of bits")

        size_a = np.arange(size)  # Create a linear array of the desired length
        self.graycode_h_seq.append(np.ones(size, dtype=np.uint8))
        # self.graycode_h_seq.append(np.zeros(size, dtype=np.uint8))
        for k in range(1, int(self.n_bits+2)):  # For each bit
            n = int(np.power(2, k))  # n = 2^k
            seq = np.split(size_a, n)  # Split the previous array into n parts
            row_out = np.zeros(size, np.uint8)  # Initialize the row output
            count = 0  # Counter for Gray code

            for i in range(len(seq)):
                # if count <= 2:
                if i % 2 == 0:
                    row_out[seq[i:i + 1]] = 1  # Set alternating parts to 1
                elif i % 2 != 0:
                    row_out[seq[i:i + 1]] = 0  # Set alternating parts to 0
                count += 1
                # if count > 2:
                #     if i % 2 == 0:
                #         row_out[seq[i:i + 1]] = 0  # Continue setting parts to 0 and 1 alternately
                #     elif i % 2 != 0:
                #         row_out[seq[i:i + 1]] = 1
                #     count += 1
                # if count > 4:
                #     count = 0  # Reset count after every 4 parts

            if axis == 0:
                self.graycode_h_seq.append(row_out)  # Append to horizontal sequence
            else:
                self.graycode_v_seq.append(row_out)  # Append to vertical sequence

    def create_graycode_h_images(self):
        for k in range(self.image_seq_h.shape[2]):  # For each bit layer
            for i in range(self.image_seq_h.shape[0]):  # For each row
                self.image_seq_h[i, :, k] = self.graycode_h_seq[k] * 255  # Fill the row with Gray code sequence

    def create_graycode_v_images(self):
        for k in range(self.image_seq_v.shape[2]):  # For each bit layer
            for i in range(self.image_seq_v.shape[1]):  # For each column
                self.image_seq_v[:, i, k] = self.graycode_v_seq[k] * 255  # Fill the column with Gray code sequence
