from tkinter import *

root = Tk()
frames_list = []
btn_list = []

turn = "X"
turnLabel = Label(root, text=turn, font="Helvetica 16 bold")
turnLabel.grid(row=3, columnspan=3)

def change_turn():
    global turn
    if turn == "O":
        turn = "X"
        turnLabel.config(text=turn)
    elif turn == "X":
        turn = "O"
        turnLabel.config(text=turn)

def check_win():
    # Horizontal wins
    if btn_list[0]["text"] == btn_list[1]["text"] == btn_list[2]["text"] == "X" or btn_list[0]["text"] == btn_list[1]["text"] == btn_list[2]["text"] == "O":
        print("{} wins".format(btn_list[0]["text"]))
    elif btn_list[3]["text"] == btn_list[4]["text"] == btn_list[5]["text"] == "X" or btn_list[3]["text"] == btn_list[4]["text"] == btn_list[5]["text"] == "O":
        print("{} wins".format(btn_list[3]["text"]))
    elif btn_list[6]["text"] == btn_list[7]["text"] == btn_list[8]["text"] == "X" or btn_list[6]["text"] == btn_list[7]["text"] == btn_list[8]["text"] == "O":
        print("{} wins".format(btn_list[6]["text"]))

    # Vertical wins
    elif btn_list[0]["text"] == btn_list[3]["text"] == btn_list[6]["text"] == "X" or btn_list[0]["text"] == btn_list[3]["text"] == btn_list[6]["text"] == "O":
        print("{} wins".format(btn_list[0]["text"]))
    elif btn_list[1]["text"] == btn_list[4]["text"] == btn_list[7]["text"] == "X" or btn_list[1]["text"] == btn_list[4]["text"] == btn_list[7]["text"] == "O":
        print("{} wins".format(btn_list[1]["text"]))
    elif btn_list[2]["text"] == btn_list[5]["text"] == btn_list[8]["text"] == "X" or btn_list[2]["text"] == btn_list[5]["text"] == btn_list[8]["text"] == "O":
        print("{} wins".format(btn_list[2]["text"]))

    # Diagonal wins
    elif btn_list[0]["text"] == btn_list[4]["text"] == btn_list[8]["text"] == "X" or btn_list[0]["text"] == btn_list[4]["text"] == btn_list[8]["text"] == "O":
        print("{} wins".format(btn_list[0]["text"]))
    elif btn_list[2]["text"] == btn_list[4]["text"] == btn_list[6]["text"] == "X" or btn_list[2]["text"] == btn_list[4]["text"] == btn_list[6]["text"] == "O":
        print("{} wins".format(btn_list[2]["text"]))

    # If no one wins
    else:
        change_turn()

def process_turn(ndex):
    btn_list[ndex].config(text=turn)
    check_win()

def create_frames_and_buttons():
    ndex = 0
    i = 0
    x = 0
    for i in range(3):
        for x in range(3):
            frames_list.append(Frame(root, width = 100, height = 100))
            frames_list[ndex].propagate(False)
            frames_list[ndex].grid(row = i, column = x, sticky = "nsew", padx = 2, pady = 2)
            btn_list.append(Button(frames_list[ndex], text="", font="Helvetica 16 bold",
                   command = lambda ndex=ndex: process_turn(ndex)))
            btn_list[ndex].pack(expand=True, fill=BOTH)
            x += 1
            ndex += 1
        i += 1
    root.resizable(width=False, height=False)

create_frames_and_buttons()

root.mainloop()
