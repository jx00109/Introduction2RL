import numpy as np
import pandas as pd


class QLearningTable:
    def __init__(self, actions, lr=0.01, gamma=0.9, e_greedy=0.9):
        self.actions = actions
        self.lr = lr
        self.gamma = gamma
        self.epsilon = e_greedy
        self.qtable = pd.DataFrame(
            columns=self.actions
        )

    def choose_action(self, observation):
        self.check_state_exist(observation)
        if np.random.uniform() < self.epsilon:
            state_action = self.qtable.loc[observation, :]
            state_action = state_action.reindex(np.random.permutation(state_action.index))
            action = state_action.idxmax()
        else:
            action = np.random.choice(self.actions)
        return action

    def learn(self, s, a, r, s_):
        self.check_state_exist(s_)
        q_predict = self.qtable.loc[s, a]
        if s_ != 'terminal':
            q_target = r + self.gamma * self.qtable.loc[s_, :].max()  # next state is not terminal
        else:
            q_target = r  # next state is terminal
        self.qtable.loc[s, a] += self.lr * (q_target - q_predict)  # update

    def check_state_exist(self, state):
        if state not in self.qtable.index:
            self.qtable = self.qtable.append(
                pd.Series(
                    [0] * len(self.actions),
                    index=self.qtable.columns,
                    name=state,
                )
            )
