import itertools
import pandas as pd
from collections import OrderedDict


# actions (simple and stake doubles)
ACTIONS = [0, 1]
DOUBLED_ACTIONS = [0, 1, 2]


# rewards
REWARDS = [-1, 0, 1, 1.5]
REWARDS_TO_IDX = {r:i for i,r in enumerate(REWARDS)}
IDX_TO_REWARDS = {i:r for r,i in REWARDS_TO_IDX.items()}


# simple states
STATES = pd.DataFrame(itertools.product(range(4, 32), range(1, 11), (0, 1)), 
                      columns=['total', 'home_first', 'usable_ace'])
STATES = STATES[(STATES['usable_ace'] == 0) | 
                ((STATES['total'] >= 12) & (STATES['total'] <= 21))].reset_index(drop=True)
TERMINATE_IDX = len(STATES[STATES['total'] <= 21])
STATE_TO_IDX = OrderedDict({tuple(s):i for i,s in enumerate(STATES.values.tolist())})
IDX_TO_STATE = OrderedDict({i:s for s,i in STATE_TO_IDX.items()})


# states with counts
STATES_COUNTS = pd.DataFrame(itertools.product(range(4, 32), range(1, 11), (0, 1), range(-20, 21)), 
                      columns=['total', 'home_first', 'usable_ace', 'count'])
STATES_COUNTS = \
    STATES_COUNTS[(STATES_COUNTS['usable_ace'] == 0) |
                  ((STATES_COUNTS['total'] >= 12) & (STATES_COUNTS['total'] <= 21))].reset_index(drop=True)
TERMINATE_IDX_COUNTS = len(STATES_COUNTS[STATES_COUNTS['total'] <= 21])
STATE_COUNTS_TO_IDX = OrderedDict({tuple(s):i for i,s in enumerate(STATES_COUNTS.values.tolist())})
IDX_TO_STATE_COUNTS = OrderedDict({i:s for s,i in STATE_COUNTS_TO_IDX.items()})

