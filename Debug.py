from tkinter import *
from PIL import ImageTk, Image
import numpy as np
from Board import *
from random import random
from copy import copy
from SimpleBot import *
from BotAPI import *

if __name__ == "__main__"
    root = Tk()
    #size of the window
    root.geometry("600x600")
    #Create the bot
    dumbBot = SimpleBot()
    cleverBot = Bot()
    #Create the game board
    app = Board(root, cleverBot.MakeMove)
    
    root.mainloop()
    for elem in dumbBot.Decisions:
        print(elem[0])
        print(elem[1])
