from enum import Enum


class CardType(Enum):
    Infantry = 0
    Cavalry = 1
    Artillery = 2
    Wild = 3


class Card:
    def __init__(self, node, ctype):
        self.type = ctype
        self.node = node
        self.owner = -1


CARD_FIX_SCORE = {
    CardType.Infantry: 4,
    CardType.Cavalry: 6,
    CardType.Artillery: 8,
    CardType.Wild: 10
}
