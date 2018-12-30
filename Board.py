from tkinter import *
from tkinter.ttk import Frame, Label, Style
import numpy as np
from copy import copy
from random import random

class Board(Frame):
    def __init__(self, master, BotAI):
        self._Bot_AI = BotAI

        self.Player_decisions = []

        Frame.__init__(self, master)
        self.master = master
        self.init_window()
        self.GameMatrixState = np.zeros((3,3))
        self.ButtonMatrix = []
        self.Create_Buttons()

        #flip the coin on who should start
        if(random()>0.5):
            self.Call_Bot_AI()
    #Creation of init_window
    def init_window(self):
        # changing the title of our master widget
        self.master.title("Tic Tac Toe")

        # allowing the widget to take the full space of the root window
        self.pack(fill=BOTH, expand=1)

    def click(self, event, row, column):
        self.ButtonMatrix[row][column].configure(text = "X", state = "disabled")
        playerDecision = np.zeros((3,3))
        playerDecision[row,column] = 1
        #Store the players decision, dont forget to flip the signs
        self.Player_decisions.append(
                                (copy(
                                    np.dot(self.GameMatrixState,-1)),
                                playerDecision))
        #Lets encode the player as -1 and the bot as 1
        self.GameMatrixState[row,column] = -1
        if self.Check_Win():
            return None

        #Call on the bot and let it make a move
        self.Call_Bot_AI()

        if self.Check_Win():
            return None

    def EndGame(self):
        for i in range(3):
            for j in range(3):
                self.ButtonMatrix[i][j].configure(state = "disabled")
        self.PrintAllMoves()

    def Call_Bot_AI(self):
        BotDecision = self._Bot_AI(self.GameMatrixState)
        self.GameMatrixState[BotDecision[0], BotDecision[1]] = 1
        self.ButtonMatrix[BotDecision[0]][BotDecision[1]].configure(text = "O", state = "disabled")

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

    def PrintAllMoves(self):
        for decision in self.Player_decisions:
            print(decision[0])
            print("\n")
            print(decision[1])
            print("\n"*3)

    def Check_Win(self):
        A = self.GameMatrixState
        for i in range(3):
            rowSum = 0
            columnSum = 0
            for j in range(3):
                rowSum += A[i,j]
                columnSum += A[j,i]
            if rowSum == 3 or columnSum == 3:
                self.EndGame()
                return True
            elif rowSum == -3 or columnSum == -3:
                self.EndGame()
                return True
        DiagonalSum1 = 0
        DiagonalSum2 = 0
        for i in range(3):
            DiagonalSum1 += A[i,i]
            DiagonalSum2 += A[2-i,i]
        if DiagonalSum1 == 3 or DiagonalSum2 == 3:
            self.EndGame()
            return True
        elif DiagonalSum1 == -3 or DiagonalSum2 == -3:
            self.EndGame()
            return True
