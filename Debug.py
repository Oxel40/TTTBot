from tkinter import *
from PIL import ImageTk, Image
import numpy as np
from Board import *
from random import random
from copy import copy
from SimpleBot import *

def main():
    root = Tk()
    #size of the window
    root.geometry("600x600")
    #Create the bot
    dumbBot = SimpleBot()
    #Create the game board
    app = Board(root, dumbBot.MakeMove)
    root.mainloop()
main()
