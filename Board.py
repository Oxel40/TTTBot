from tkinter import Tk, Canvas, PhotoImage, mainloop
from tkinter.ttk import Frame, Label, Style

class Board(Frame):
    def __init__(self, root):
        self.HEIGHT = 600
        self.WIDTH = 600
        self.canvas = Canvas(root, width = self.WIDTH, height = self.HEIGHT, bg = "#484848")
        self.DrawGridLines()
        self.canvas.pack()
    def DrawGridLines(self):
        line_width = 5
        self.canvas.create_line(self.WIDTH / 3,
                           self.HEIGHT,
                           self.WIDTH / 3,
                           0,
                           width = line_width)
        self.canvas.create_line(self.WIDTH * 2 / 3,
                           self.HEIGHT,
                           self.WIDTH * 2 / 3,
                           0,
                           width = line_width)
        self.canvas.create_line(0,
                           self.HEIGHT / 3,
                           self.WIDTH,
                           self.HEIGHT / 3,
                           width = line_width)
        self.canvas.create_line(0,
                           self.HEIGHT * 2 / 3,
                           self.WIDTH,
                           self.HEIGHT * 2 / 3,
                           width = line_width)
    def DrawCircle(self, x, y):
        self.canvas.create_oval((x - 1) * self.WIDTH / 3 + 25,
                                (y - 1) * self.HEIGHT / 3 + 25,
                                 x * self.WIDTH / 3 - 25,
                                 y * self.HEIGHT /3 - 25,
                                 outline = "#FF0000",
                                 width = 7)
    def DrawCross(self, x, y):
        self.canvas.create_line((x - 1) * self.WIDTH / 3 + 25,
                                (y - 1) * self.HEIGHT / 3 + 25,
                                 x * self.WIDTH / 3 - 25,
                                 y * self.HEIGHT / 3 - 25,
                                 width = 5)
        self.canvas.create_line((x - 1) * self.WIDTH / 3 + 25,
                                 y * self.HEIGHT / 3 - 25,
                                 x * self.WIDTH / 3 - 25,
                                (y - 1) * self.HEIGHT / 3 + 25,
                                 width = 5)
