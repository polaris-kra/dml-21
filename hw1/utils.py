import random
import numpy as np
import skopt
import gym
from consts import *
from doubled_blackjack import DoubledBlackjackEnv
from doubled_counting_blackjack import DoubledCountingBlackjackEnv
from accomplice_blackjack import AccompliceBlackjackEnv


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def blackjack_gym(func):
    """ Provides evaluation functions with Blackjack gym game environment. """
    
    def wrapper(*args, **kwargs):
        if 'env' in kwargs:
            return func(*args, **kwargs)
        with gym.make('Blackjack-v0', natural=True) as env:
            return func(env=env, *args, **kwargs)

    return wrapper


@blackjack_gym
def simulate(*, env, pi):
    """ Simulates single game experiment with given strategy pi. """
    
    state = env.reset()
    
    states = []
    rewards = []
    done = False

    while not done:
        sid = STATE_TO_IDX[state]
        action = pi[sid]
        state, reward, done, _ = env.step(action)
        states.append(state)
        rewards.append(reward)

    return states, rewards


@blackjack_gym
def simulate_many(*, env, n_exp, pi):
    """ Simulates multiple game experiments with given strategy pi. """
    
    G_total = 0
    l_total = 0
    
    for _ in range(n_exp):
        _, rewards = simulate(env=env, pi=pi)
        G_total += sum(rewards)
        l_total += len(rewards)

    return G_total/n_exp, l_total/n_exp 


def doubled_blackjack_gym(func):
    """ Provides evaluation functions with Blackjack with doubles gym game environment. """
    
    def wrapper(*args, **kwargs):
        if 'env' in kwargs:
            return func(*args, **kwargs)
        with DoubledBlackjackEnv(natural=True) as env:
            return func(env=env, *args, **kwargs)

    return wrapper


@doubled_blackjack_gym
def simulate_with_double(*, env, pi):
    """ Simulates single game experiment with given strategy pi. """
    
    state = env.reset()
    
    states = []
    rewards = []
    done = False

    while not done:
        sid = STATE_TO_IDX[state]
        action = pi[sid]
        state, reward, done, _ = env.step(action)
        states.append(state)
        rewards.append(reward)

    return states, rewards


@doubled_blackjack_gym
def simulate_many_with_double(*, env, n_exp, pi):
    """ Simulates multiple game experiments with given strategy pi. """
    
    G_total = 0
    l_total = 0
    
    for _ in range(n_exp):
        _, rewards = simulate_with_double(env=env, pi=pi)
        G_total += sum(rewards)
        l_total += len(rewards)

    return G_total/n_exp, l_total/n_exp 


def doubled_counting_blackjack_gym(func):
    """ Provides evaluation functions with Blackjack with doubles gym game environment. """
    
    def wrapper(*args, **kwargs):
        if 'env' in kwargs:
            return func(*args, **kwargs)
        with DoubledCountingBlackjackEnv(natural=True) as env:
            return func(env=env, *args, **kwargs)

    return wrapper


@doubled_counting_blackjack_gym
def simulate_with_double_counting(*, env, pi):
    """ Simulates single game experiment with given strategy pi. """
    
    state = env.reset()
    
    states = []
    rewards = []
    done = False

    while not done:
        sid = STATE_COUNTS_TO_IDX[state]
        action = pi[sid]
        state, reward, done, _ = env.step(action)
        states.append(state)
        rewards.append(reward)

    return states, rewards


@doubled_counting_blackjack_gym
def simulate_many_with_double_counting(*, env, n_exp, pi):
    """ Simulates multiple game experiments with given strategy pi. """
    
    G_total = 0
    l_total = 0
    
    for _ in range(n_exp):
        _, rewards = simulate_with_double_counting(env=env, pi=pi)
        G_total += sum(rewards)
        l_total += len(rewards)

    return G_total/n_exp, l_total/n_exp 


def accomplice_blackjack_gym(func):
    """ Provides evaluation functions with Blackjack with doubles gym game environment. """
    
    def wrapper(*args, **kwargs):
        if 'env' in kwargs:
            return func(*args, **kwargs)
        with AccompliceBlackjackEnv(hot_coeff=10, natural=True) as env:
            return func(env=env, *args, **kwargs)

    return wrapper


@accomplice_blackjack_gym
def simulate_with_accomplice(*, env, pi):
    """ Simulates single game experiment with given strategy pi. """
    
    state = env.reset()
    
    states = []
    rewards = []
    done = False

    while not done:
        sid = STATE_COUNTS_TO_IDX[state]
        action = pi[sid]
        state, reward, done, _ = env.step(action)
        states.append(state)
        rewards.append(reward)

    return states, rewards


@accomplice_blackjack_gym
def simulate_many_with_accomplice(*, env, n_exp, pi):
    """ Simulates multiple game experiments with given strategy pi. """
    
    G_total = 0
    l_total = 0
    
    for _ in range(n_exp):
        _, rewards = simulate_with_accomplice(env=env, pi=pi)
        G_total += sum(rewards)
        l_total += len(rewards)

    return G_total/n_exp, l_total/n_exp 


def run_optimization(param_space, static_params, opt_params, strategy_cls):
    @skopt.utils.use_named_args(param_space)
    def objective(**params):
        all_params = {**params, **static_params}

        strategy = strategy_cls(**all_params)
        strategy.fit()
        G_mean, _ = strategy.evaluate(n_exp=100000)
        
        return -G_mean

    results = skopt.forest_minimize(objective, param_space, **opt_params)
    
    return results


def get_random_V():
    V = 2.5*np.random.random(len(STATES)) - 1  # [-1, 1.5]
    V[TERMINATE_IDX:] = 0.0
    return V


def get_random_Q():
    Q = 2.5*np.random.random((len(STATES), len(ACTIONS))) - 1  # [-1, 1.5]
    Q[TERMINATE_IDX:] = 0.0
    return Q


def get_random_pi():
    return np.random.randint(len(ACTIONS), size=len(STATES))


def get_Q_by_R(R, Q_default=None):
    Q = Q_default if Q_default is not None else np.zeros((len(STATES), len(ACTIONS)))
    for s in range(len(STATES)):
        for a in range(len(ACTIONS)):
            if len(R[s][a]) > 0:
                Q[s, a] = np.mean(R[s][a])
    return Q


def guess_hand(s, is_natural=False):
    total = s[0]
    ace = s[2]
    
    if is_natural:
        assert total == 21
        assert ace == 1
        return [1, 10]
    
    if total == 21:
        return [10, 8, 3] if not ace else [1, 5, 5]
    
    hand = []
    if ace:
        assert 12 <= total <= 21
        hand.append(1)
        total -= 11

    hand.append(total)

    return hand


class BaseStrategy:
    """ Pi-strategy container with all neccesary context. """
    
    def __init__(self):
        self.pi = None

    @blackjack_gym
    def fit(self, env):
        raise NotImplemented()

    @blackjack_gym
    def play(self, env):
        return simulate(env=env, pi=self.pi)

    @blackjack_gym
    def evaluate(self, env, n_exp):
        return simulate_many(env=env, pi=self.pi, n_exp=n_exp)


class BaseDoubledStrategy:
    """ Pi-strategy container with all neccesary context. """
    
    def __init__(self):
        self.pi = None

    @doubled_blackjack_gym
    def fit(self, env):
        raise NotImplemented()

    @doubled_blackjack_gym
    def play(self, env):
        return simulate_with_double(env=env, pi=self.pi)

    @doubled_blackjack_gym
    def evaluate(self, env, n_exp):
        return simulate_many_with_double(env=env, pi=self.pi, n_exp=n_exp)


class BaseDoubledCountingStrategy:
    """ Pi-strategy container with all neccesary context. """
    
    def __init__(self):
        self.pi = None

    @doubled_counting_blackjack_gym
    def fit(self, env):
        raise NotImplemented()

    @doubled_counting_blackjack_gym
    def play(self, env):
        return simulate_with_double_counting(env=env, pi=self.pi)

    @doubled_counting_blackjack_gym
    def evaluate(self, env, n_exp):
        return simulate_many_with_double_counting(env=env, pi=self.pi, n_exp=n_exp)


class BaseAccompliceStrategy:
    """ Pi-strategy container with all neccesary context. """
    
    def __init__(self):
        self.pi = None

    @accomplice_blackjack_gym
    def fit(self, env):
        raise NotImplemented()

    @accomplice_blackjack_gym
    def play(self, env):
        return simulate_with_accomplice(env=env, pi=self.pi)

    @accomplice_blackjack_gym
    def evaluate(self, env, n_exp):
        return simulate_many_with_accomplice(env=env, pi=self.pi, n_exp=n_exp)

