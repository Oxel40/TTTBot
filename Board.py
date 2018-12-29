from tkinter import *
from tkinter.ttk import Frame, Label, Style
import numpy as np

class Board(Frame):
    def __init__(self, master=None):
        Frame.__init__(self, master)
        self.master = master
        self.init_window()
        self.GameMatrixState = np.zeros((3,3))
        self.ButtonMatrix = []
        self.Create_Buttons()
    #Creation of init_window
    def init_window(self):
        # changing the title of our master widget
        self.master.title("Tic Tac Toe")

        # allowing the widget to take the full space of the root window
        self.pack(fill=BOTH, expand=1)

    def click(self, event, row, column):
        self.ButtonMatrix[row][column].configure(text = "X")

    def Create_Buttons(self):
        for i in range(3):
            self.ButtonMatrix.append([])
            for j in range(3):
                self.ButtonMatrix[i].append(Button(self,
                                                 width = 7,
                                                 height = 3,
                                                 font = "Helvetica 32 bold"))
                self.ButtonMatrix[i][j].grid(row = i,column = j)
                self.ButtonMatrix[i][j].bind("<Button-1>",
                                             lambda event, i=i, j=j: self.click(event,i,j))
