from tkinter import Tk, Canvas, PhotoImage, mainloop
from tkinter.ttk import Frame, Label, Style
from Board import *

def main():
    root = Tk()
    GameBoard = Board(root)
    root.mainloop()
main()
