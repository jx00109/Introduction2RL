import numpy as np
import pandas as pd
import time

'''
_o_____T 
探索者在直线世界中寻找宝藏T
'''
np.random.seed(2)  # 为了结果能够复现，创建一个随机种子

N_STATES = 6  # 可能的状态数，也就是o的位置/直线的长度
ACTIONS = ['left', 'right']  # 可选的动作，该例子中只能向左或向右
EPSILON = 0.9  # greedy policy中选中最大Q对应a的概率
ALPHA = 0.1  # 学习率
GAMMA = 0.9  # 折扣率
MAX_EPISODES = 13  # maximum episodes
FRESH_TIME = 0.3  # fresh time for one move


# 初始化Q表
def build_qtable(n_states, actions):
    table = pd.DataFrame(
        np.zeros((n_states, len(actions))),  # Q表的初始值
        columns=actions,  # 列名 left or right
    )
    return table


def choose_action(state, q_table):
    state_actions = q_table.iloc[state, :]
    # 10%的概率或者该状态下所有值都是0的情况下随机选择动作
    if (np.random.uniform() > EPSILON) or ((state_actions == 0).all()):
        action_name = np.random.choice(ACTIONS)
    # 选择maxQ对应的action
    else:
        action_name = state_actions.idxmax()

    return action_name


# 根据动作得到奖励，并转移到下一个状态
def get_env_feedback(S, A):
    if A == 'right':
        if S == N_STATES - 2:  # 数组下标从0开始，所以宝藏前一个为N_STATES - 2
            S_ = 'terminal'
            R = 1
        else:
            S_ = S + 1
            R = 0
    # 往左走没有奖励
    else:
        R = 0
        if S == 0:
            S_ = S
        else:
            S_ = S - 1

    return S_, R


def updata_env(S, episode, step_counter):
    env_list = ['-'] * (N_STATES - 1) + ['T']
    if S == 'terminal':
        interaction = 'Episode %s: total_steps = %s' % (episode + 1, step_counter)
        print('\r{}'.format(interaction), end='')
        time.sleep(2)
        print('\r                                       ', end='')
    else:
        env_list[S] = 'o'
        interaction = ''.join(env_list)
        print('\r{}'.format(interaction), end='')
        time.sleep(FRESH_TIME)


def rl():
    qtable = build_qtable(N_STATES, ACTIONS)
    for episode in range(MAX_EPISODES):
        step_counter = 0
        S = 0
        is_terminated = False
        updata_env(S, episode, step_counter)
        while not is_terminated:
            A = choose_action(S, qtable)
            S_, R = get_env_feedback(S, A)
            q_predict = qtable.loc[S, A]
            if S_ != 'terminal':
                q_target = R + GAMMA * qtable.iloc[S_, :].max()
            else:
                q_target = R
                is_terminated = True
            qtable.loc[S, A] += ALPHA * (q_target - q_predict)
            S = S_

            step_counter += 1
            updata_env(S, episode, step_counter)


    return qtable


if __name__ == "__main__":
    qtable = rl()
    print('\r\nQ-Table:\n')
    print(qtable)
