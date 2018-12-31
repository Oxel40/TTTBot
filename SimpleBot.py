import numpy as np
from random import random
from copy import copy

class SimpleBot():
    def __init__(self):
        self.Decisions = []
    def MakeMove(self, GameMatrixState):
        #Given the current state of the game,
        #given by GameMatrixState
        #Mave an edjucated move


        #or so it should be, this is a retarded bot
        GuessMatrix = np.zeros((3,3))
        for i in range(3):
            for j in range(3):
                if GameMatrixState[i,j] == 0:
                    GuessMatrix[i,j] = random()

        #get the coordinates of the button with the largest
        #probability value.
        clickCords = self.MaxCords(GuessMatrix)
        #Store the decision made by the bot in a list
        decision = np.zeros((3,3))
        decision[clickCords[0], clickCords[1]] = 1
        self.Decisions.append((copy(GameMatrixState), decision))
        #click the button with those coordinates
        return clickCords

    def MaxCords(self, GuessMatrix):
        max = 0
        maxcords = (0,0)
        for i in range(3):
            for j in range(3):
                if GuessMatrix[i,j] > max:
                    maxcords = (i,j)
                    max = GuessMatrix[i,j]
        return maxcords
    def Clear_Decisionlist(self):
        self.Decisions = []
