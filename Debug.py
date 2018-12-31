from tkinter import *
from PIL import ImageTk, Image
import numpy as np
from Board import *
from random import random
from copy import copy
from SimpleBot import *
from BotAPI import *

if __name__ == "__main__":
    root = Tk()
    #size of the window
    root.geometry("600x600")
    #Create the bot
    dumbBot = SimpleBot()
    #Create the intelligent Bot, his name is Charlie
    cleverBot = Bot()
    if cleverBot.LoadBot("Charlie") == False:
        cleverBot.NewBot("Tom")

    #Create the game board
    app = Board(root, cleverBot.MakeMove)

    root.mainloop()
    for elem in cleverBot.Decision:
        print(elem[0])
        print(elem[1])
