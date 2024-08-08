from include import FringePattern

if __name__ == '__main__':

    fringe = FringePattern.FringePattern(resolution=(800, 600), f_sin=2, steps=4)
    fringe.show_image()