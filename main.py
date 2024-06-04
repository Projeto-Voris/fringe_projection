import FringePattern
import fringes as frng
import cv2
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    f = frng.Fringes()  # instantiate class

    f.X = 800  # set width of the fringe patterns
    f.Y = 600  # set height of the fringe patterns
    f.H = 1 # color of fringes
    f.K = 1  # set number of sets (number of fringe patterns with different spatial frequencies).)
    f.N = 0  # set number of shifts (Steps)
    f.v = [2]  # set spatial frequencies
    f.D = 1 # Number of directions
    f.angle = 0 # Angle of fringes
    f.axis = 0 # Axis of fringe projection (0:x; 1:y)
    print(f.T) # get number of frames

    I = f.encode()  # encode fringe patterns

    # for i in range(I.shape[0]):
    #     cv2.imshow('I', I[i])
    #     cv2.waitKey(0)


    fringe = FringePattern.FringePattern(width=600, height=800, f_sin=10, steps=8)
    fringe.create_fringe_image()
    fringe.show_image()
    fringe.print_image()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
