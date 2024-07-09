import cv2
import screeninfo
import GrayCode
import FringePattern

VISUALIZE = True

if __name__ == '__main__':
    cv2.namedWindow('projector', cv2.WINDOW_NORMAL)  # create window to fringe and graycode

    move = (0, 0)  # initialize variable to move opencv window
    width = 1024
    height = 1024
    img_resolution = (width, height)  # initialize image resolution

    for m in screeninfo.get_monitors():  # verify all monitors
        if m.name == '\\\\.\\DISPLAY3':  # Search for projector connected a specific input port
            move = (m.x, m.y)  # get movement from primary display
            img_resolution = (m.width, m.height)  # get projector resolution
    cv2.setWindowProperty('projector', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)  # change window property
    cv2.moveWindow('projector', move[0], move[1])  # move projector window to projector display

    graycode = GrayCode.GrayCode(resolution=img_resolution, n_bits=8, axis=0)  # create graycode class
    images = graycode.get_images()  # get graycode images
    fringe = FringePattern.FringePattern(resolution=img_resolution, f_sin=4, steps=8)  # create FringePattern class
    fringe.create_fringe_image()
    image_f = fringe.get_image()  # get FringePattern images

    for k in range(fringe.get_image().shape[2]):  # create a fringe image in projector
        cv2.imshow('projector', fringe.get_image()[:, :, k])
        cv2.waitKey(0)

    for k in range(graycode.get_images().shape[2]):  # create a graycode image in projector
        cv2.imshow('projector', graycode.get_images()[:, :, k])
        cv2.waitKey(0)