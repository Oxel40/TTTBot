from tkinter import *
from PIL import ImageTk, Image

class Board(Frame):
    def __init__(self, master=None):
        Frame.__init__(self, master)
        self.master = master
        self.init_window()

    #Creation of init_window
    def init_window(self):

        # changing the title of our master widget
        self.master.title("Tic Tac Toe")

        # allowing the widget to take the full space of the root window
        self.pack(fill=BOTH, expand=1)
        cell_img = ImageTk.PhotoImage(Image.open("Bomb.png"))
        # creating a button instance
        quitButton = Button(self,
                            text="Quit",
                            width = 200,
                            height = 200)
        # placing the button on my window
        quitButton.place(x=0, y=0)
def main():
    root = Tk()

    #size of the window
    root.geometry("600x600")
    app = Board(root)
    root.mainloop()
main()
