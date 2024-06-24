'''
Here, we're defining all the objects for a basketball game simulation. We will use reinforcement learning to determine
an optimal strategy, starting with 1v1, then we will try 5v5 (as in a NBA game). 
We will use manim to animate the game.

We will use PyTorch to train the reinforcement learning model.
'''
class Player:
    def __init__(self, size, speed, fg_dict):
        self.size = size
        self.speed = speed
        self.fg_dict = fg_dict # dictionary with player's average shooting percentages from different ranges from the basket.

class Ball:
    def __init__(self, size, speed):
        self.size = size
        self.speed = speed


class Court:
    def __init__(self, height, width):
        self.height = height
        self.width = width

class Basket:
    def __init__(self, size):
        self.size = size

class OneVsOne:
    def __init__(self, court, basket, player1, player2):
        self.court = court
        self.basket = basket
        self.player1 = player1
        self.player2 = player2
