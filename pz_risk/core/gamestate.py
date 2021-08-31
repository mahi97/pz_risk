from enum import Enum


class GameState(Enum):
    StartTurn = 0
    Card = 1
    Reinforce = 2
    Attack = 3
    Move = 4
    Fortify = 5
    EndTurn = 6