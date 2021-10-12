from copy import copy
import gym
from gym import spaces
from gym.utils import seeding

def cmp(a, b):
    return float(a > b) - float(a < b)

# 1 = Ace, 2-10 = Number cards, Jack/Queen/King = 10
INITIAL_DECK = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10]*4
PLUS_MINUS_COUNTING = {
    1: -1,
    2: 1,
    3: 1,
    4: 1,
    5: 1,
    6: 1,
    7: 0,
    8: 0,
    9: 0,
    10: -1
}


def usable_ace(hand):  # Does this hand have a usable ace?
    return 1 in hand and sum(hand) + 10 <= 21


def sum_hand(hand):  # Return current hand total
    if usable_ace(hand):
        return sum(hand) + 10
    return sum(hand)


def is_bust(hand):  # Is this hand a bust?
    return sum_hand(hand) > 21


def score(hand):  # What is the score of this hand (0 if bust)
    return 0 if is_bust(hand) else sum_hand(hand)


def is_natural(hand):  # Is this hand a natural blackjack?
    return sorted(hand) == [1, 10]


class AccompliceBlackjackEnv(gym.Env):
    """ Simple blackjack with doubles and counting environment. """

    def __init__(self, hot_coeff=1, natural=False):
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Tuple((
            spaces.Discrete(32),
            spaces.Discrete(11),
            spaces.Discrete(2),
            spaces.Discrete(41)))
        self.seed()

        self.deck = copy(INITIAL_DECK)
        self.count = 0

        # Flag to payout 1.5 on a "natural" blackjack win, like casino rules
        # Ref: http://www.bicyclecards.com/how-to-play/blackjack/
        self.natural = natural
        self.hot_coeff = hot_coeff
        self.curr_coeff = 1
        # Start the first game
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def count_card(self, card):
        self.count += PLUS_MINUS_COUNTING[card] 

    def draw_card(self, np_random):
        card = int(np_random.choice(self.deck))
        self.deck.remove(card)
        return card

    def draw_hand(self, np_random):
        return [self.draw_card(np_random), self.draw_card(np_random)]

    def step(self, action):
        assert self.action_space.contains(action)
        if action == 0:  # stick: play out the dealers hand, and score
            done = True
            while sum_hand(self.dealer) < 17:
                card = self.draw_card(self.np_random)
                self.dealer.append(card)
                self.count_card(card)
            reward = cmp(score(self.player), score(self.dealer))
            if self.natural and is_natural(self.player) and reward == 1.:
                reward = 1.5
            reward = reward * self.curr_coeff
        elif action == 1:  # hit: add a card to players hand and return
            card = self.draw_card(self.np_random)
            self.player.append(card)
            self.count_card(card)
            if is_bust(self.player):
                done = True
                reward = -1 * self.curr_coeff
            else:
                done = False
                reward = 0.
        else:  # action = 2 - double stakes!
            done = True
            card = self.draw_card(self.np_random)
            self.count_card(card)
            self.player.append(card)
            while sum_hand(self.dealer) < 17:
                card = self.draw_card(self.np_random)
                self.dealer.append(card)
                self.count_card(card)
            reward = cmp(score(self.player), score(self.dealer)) * 2 * self.curr_coeff

        return self._get_obs(), reward, done, {}

    def _get_obs(self):
        return (sum_hand(self.player), self.dealer[0], usable_ace(self.player), self.count)

    def reset(self):
        if len(self.deck) < 15:
            self.deck = copy(INITIAL_DECK)
            self.count = 0

        if self.count >= 15:
            self.curr_coeff = self.hot_coeff
        else:
            self.curr_coeff = 1

        self.dealer = self.draw_hand(self.np_random)
        self.count_card(self.dealer[0])

        self.player = self.draw_hand(self.np_random)
        self.count_card(self.player[0])
        self.count_card(self.player[1])

        return self._get_obs()

