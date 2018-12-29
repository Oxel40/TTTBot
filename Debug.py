from tkinter import *
from PIL import ImageTk, Image
import numpy as np
from Board import *


def main():
    root = Tk()

    #size of the window
    root.geometry("600x600")
    app = Board(root)
    root.mainloop()
main()
