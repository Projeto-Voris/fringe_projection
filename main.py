import FringePattern
import fringes as frng
import cv2

if __name__ == '__main__':

    fringe = FringePattern.FringePattern()
    fringe.create_fringe_image()
    fringe.show_image()

