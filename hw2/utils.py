import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import Counter, defaultdict, OrderedDict
from abc import abstractmethod
import matplotlib.pyplot as plt
import torch
import gym


class OrderedCounter(Counter, OrderedDict):
    pass


def set_seed(seed, deterministic=True, benchmark=False):
    random.seed(seed)
    np.random.seed(seed)
    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = benchmark


def count_parameters(model):
    total = 0
    for p in model.parameters():
        total += np.prod(list(p.shape))
    return total


def simulate(player1, player2, s=None, first_is_first=True):
    """ Simulates single game experiment with given trained agents. """
    assert player1.env == player2.env
    
    if not first_is_first:
        player1, player2 = player2, player1

    env = player1.env
    p1_train = player1.is_train
    p2_train = player2.is_train
    player1.eval()
    player2.eval()
    if s is None:
        s, _, _ = env.reset()
    states = []

    while True:
        s, _, r, done = player1.step_from(s)
        states.append(s)
        if done:
            break
            
        s, _, r, done = player2.step_from(s)
        states.append(s)
        if done:
            break

    if p1_train:
        player1.train()
    if p2_train:
        player2.train()

    return states, r


def simulate_many(n_sims, player1, player2, verbose=False):
    rs = []
    for _ in tqdm(range(n_sims), disable=not verbose):
        _, r = simulate(player1, player2)
        rs.append(r)

    return Counter(rs)


def plot_results(ids, res1vsRs, resRvs2s, p1_loss, p2_loss, p1_rewards, p2_rewards, window=100):
    if p1_loss is not None:
        fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    else:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(ids, res1vsRs, label='1 vs random')
    axes[0].plot(ids, resRvs2s, label='random vs 2')
    axes[0].set_xlabel('iteration')
    axes[0].set_ylabel('win frac.')
    axes[0].set_ylim(0, 1)
    axes[0].legend()

    i = 1
    if p1_loss is not None:
        axes[1].plot(ids, np.convolve(np.array(p1_loss), np.ones(window)/window, mode='valid')[ids], label='P1 loss')
        axes[1].plot(ids, np.convolve(np.array(p2_loss), np.ones(window)/window, mode='valid')[ids], label='P2 loss')
        axes[1].set_xlabel('iteration')
        axes[1].set_ylabel('loss')
        axes[1].legend()
        i += 1

    axes[i].plot(ids, np.convolve(np.array(p1_rewards), np.ones(window)/window, mode='valid')[ids], label='P1 reward')
    axes[i].plot(ids, np.convolve(np.array(p2_rewards), np.ones(window)/window, mode='valid')[ids], label='P2 reward')
    axes[i].set_xlabel('iteration')
    axes[i].set_ylabel('rewards')
    axes[i].legend()
    
    plt.show()


class TicTacToeEnv(gym.Env):
    def __init__(self, n_rows=3, n_cols=3, n_win=3):
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.n_win = n_win

        self.board = np.zeros((self.n_rows, self.n_cols), dtype=int)
        self.gameOver = False
        self.boardHash = None
        # ход первого игрока
        self.curTurn = 1
        self.emptySpaces = None

    def getEmptySpaces(self):
        if self.emptySpaces is None:
            res = np.where(self.board == 0)
            self.emptySpaces = [(i, j) for i,j in zip(res[0], res[1])]
        return self.emptySpaces

    def makeMove(self, player, i, j):
        self.board[i, j] = player
        self.emptySpaces = None
        self.boardHash = None

    def getHash(self):
        if self.boardHash is None:
            self.boardHash = ''.join(['%s' % (x+1) for x in self.board.reshape(self.n_rows * self.n_cols)])
        return self.boardHash

    def isTerminal(self):
        # проверим, не закончилась ли игра
        cur_marks, cur_p = np.where(self.board == self.curTurn), self.curTurn
        for i,j in zip(cur_marks[0], cur_marks[1]):
            win = False
            if i <= self.n_rows - self.n_win:
                if np.all(self.board[i:i+self.n_win, j] == cur_p):
                    win = True
            if not win:
                if j <= self.n_cols - self.n_win:
                    if np.all(self.board[i,j:j+self.n_win] == cur_p):
                        win = True
            if not win:
                if i <= self.n_rows - self.n_win and j <= self.n_cols - self.n_win:
                    if np.all(np.array([ self.board[i+k,j+k] == cur_p for k in range(self.n_win) ])):
                        win = True
            if not win:
                if i <= self.n_rows - self.n_win and j >= self.n_win-1:
                    if np.all(np.array([ self.board[i+k,j-k] == cur_p for k in range(self.n_win) ])):
                        win = True
            if win:
                self.gameOver = True
                return self.curTurn

        if len(self.getEmptySpaces()) == 0:
            self.gameOver = True
            return 0

        self.gameOver = False
        return None

    def printBoard(self):
        glyphs = {-1: 'o', 0: '-', 1: 'x'}
        glyphs = np.vectorize(glyphs.get)
        result = glyphs(self.board)
        [print(''.join(r)) for r in result]

    def get_state(self):
        return (self.getHash(), self.getEmptySpaces(), self.curTurn)

    def action_from_int(self, action_int):
        return ( int(action_int / self.n_cols), int(action_int % self.n_cols))

    def int_from_action(self, action):
        return action[0] * self.n_cols + action[1]
    
    def step(self, action):
        if self.board[action[0], action[1]] != 0:
            return self.get_state(), -10, True
        self.makeMove(self.curTurn, action[0], action[1])
        reward = self.isTerminal()
        self.curTurn = -self.curTurn
        
        return self.get_state(), 0 if reward is None else reward, reward is not None

    def reset(self):
        self.board = np.zeros((self.n_rows, self.n_cols), dtype=int)
        self.boardHash = None
        self.gameOver = False
        self.emptySpaces = None
        self.curTurn = 1
        
        return self.get_state()


class BaseAgent:
    def __init__(self, env, player, gamma=1.0, eps=0.1, train=True):
        self.env = env
        self.player = player
        self.gamma = gamma
        self.eps = eps
        self.is_train = train

    def train(self):
        self.is_train = True
        
    def eval(self):
        self.is_train = False

    @abstractmethod
    def get_action(self, s):
        pass

    @abstractmethod
    def learn(self, s_prev, a_prev, s_cur, a_cur, s_new, r, done):
        pass

    def step_from(self, s):
        a = self.get_action(s)
        (s, _, _), r, done = self.env.step(a)
        return s, a, r, done


class RandomAgent(BaseAgent):
    def __init__(self, env, player):
        BaseAgent.__init__(self, env, player)

    def get_action(self, s):
        return random.choice(self.env.getEmptySpaces())

    def learn(self, s_prev, a_prev, s_cur, a_cur, s_new, r, done):
        pass


def train_episode(env, player1, player2):
    """ Simulates single game episode with trainable agents. """

    s_prev = None
    a_prev = None
    r = None
    s_cur, r, done = env.reset()

    while True:
        # player 1 step
        s, a_cur, r, done = player1.step_from(s_cur)
        if s_prev is not None:
            player2.learn(s_prev, a_prev, s_cur, a_cur, s, r, done)
        if done:
            player1.learn(s_cur, a_cur, s_cur, a_cur, s, r, done)
            break

        # player 2 step
        s_new, a, r, done = player2.step_from(s)
        player1.learn(s_cur, a_cur, s, a, s_new, r, done)
        if done:
            player2.learn(s, a, s, a, s_new, r, done)
            break

        s_prev = s
        a_prev = a
        s_cur = s_new
        
    return r


class ReplayMemory():
    def __init__(self, capacity=10000):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def store(self, exptuple):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = exptuple
        self.position = (self.position + 1) % self.capacity
       
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)

