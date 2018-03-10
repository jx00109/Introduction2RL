from maze_env import Maze
from RL_brain import SarsaTable
import time


def update():
    ndeath = 0
    nlive = 0

    start = time.time()
    for episode in range(100):

        # initial observation
        observation = env.reset()

        # RL choose action based on observation
        action = RL.choose_action(str(observation))

        while True:
            # fresh env
            env.render()

            # RL take action and get next observation and reward
            observation_, reward, done = env.step(action)

            # RL choose action based on next observation
            action_ = RL.choose_action(str(observation_))

            # RL learn from this transition (s, a, r, s, a) ==> Sarsa
            RL.learn(str(observation), action, reward, str(observation_), action_)

            # swap observation and action
            observation = observation_
            action = action_

            # break while loop when end of this episode
            if done:
                if reward == 1:
                    nlive += 1
                if reward == -1:
                    ndeath += 1
                print('第%d回合，死亡： %d, 成功: %d' % (episode + 1, ndeath, nlive))
                break

    stop = time.time()
    # end of game
    print('SARSA结束，耗时：%f' % (stop - start))

    env.destroy()


if __name__ == "__main__":
    env = Maze()
    RL = SarsaTable(actions=list(range(env.n_actions)))

    env.after(100, update)
    env.mainloop()
