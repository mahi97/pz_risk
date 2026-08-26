from pz_risk.core.card import CardType


class Player:
    def __init__(self, pid, init_placement=0):
        self.id = pid
        self.cards = {t: [] for t in CardType}
        self.placement = init_placement
        self.deserve_card = False

    def num_cards(self):
        return sum([len(v) for v in self.cards.values()])

