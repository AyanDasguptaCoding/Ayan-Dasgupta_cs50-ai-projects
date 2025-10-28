import random
from collections import defaultdict


def get_q_value(self, state, action):
    """
    Return the Q-value for the state and action pair.
    If no Q-value exists yet, return 0.
    """
    state_tuple = tuple(state)
    action_tuple = tuple(action)
    return self.q.get((state_tuple, action_tuple), 0)


def update_q_value(self, state, action, old_q, reward, future_rewards):
    """
    Update the Q-value for the state and action pair
    using the Q-learning formula:
    Q(s, a) = old_q + alpha * (reward + future_rewards - old_q)
    """
    state_tuple = tuple(state)
    action_tuple = tuple(action)
    new_q = old_q + self.alpha * (reward + future_rewards - old_q)
    self.q[(state_tuple, action_tuple)] = new_q


def best_future_reward(self, state):
    """
    Return the best possible future reward for a state.
    That is, for all available actions in the state, return the maximum Q-value.
    If no Q-values exist yet, return 0.
    """
    available_actions = Nim.available_actions(state)
    if not available_actions:
        return 0
    
    max_reward = -float('inf')
    state_tuple = tuple(state)
    
    for action in available_actions:
        action_tuple = tuple(action)
        q_value = self.q.get((state_tuple, action_tuple), 0)
        if q_value > max_reward:
            max_reward = q_value
    
    return max_reward if max_reward != -float('inf') else 0


def choose_action(self, state, epsilon=False):
    """
    Choose an action to take in the current state.
    If epsilon is True, use epsilon-greedy algorithm.
    Otherwise, choose the best action available.
    """
    available_actions = Nim.available_actions(state)
    if not available_actions:
        return None
    
    state_tuple = tuple(state)
    
    if epsilon and random.random() < self.epsilon:
        # Choose random action
        return random.choice(list(available_actions))
    else:
        # Choose best action based on Q-values
        best_action = None
        best_q_value = -float('inf')
        
        for action in available_actions:
            action_tuple = tuple(action)
            q_value = self.q.get((state_tuple, action_tuple), 0)
            if q_value > best_q_value:
                best_q_value = q_value
                best_action = action
            elif q_value == best_q_value:
                # If multiple actions have same Q-value, choose randomly among them
                if random.choice([True, False]):
                    best_action = action
        
        return best_action if best_action is not None else random.choice(list(available_actions))