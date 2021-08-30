"""
Sarsa is a online updating method for Reinforcement learning.

Unlike Q learning which is a offline updating method, Sarsa is updating while in the current trajectory.

You will see the sarsa is more coward when punishment is close because it cares about all behaviours,
while q learning is more brave because it only cares about maximum behaviour.
"""

from maze_env import Maze
from RL_brain import SarsaLambdaTable,SarsaTable,QLearningTable
import logging
import sys
import datetime

LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"

def update():
    st = datetime.datetime.now();
    total_count = 0
    step_counter = 0
    for episode in range(100):
        # initial observation
        logging.info("-------------Episode[%d]-------------" % episode)
        observation = env.reset()
        total_count += step_counter
        step_counter = 0

        # RL choose action based on observation
        action = RL.choose_action(str(observation))

        # initial all zero eligibility trace
        RL.OnNewEpisode()

        while True:
            # fresh env
            env.render()
            step_counter += 1
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
                break

        logging.info("StepCount:%d" % step_counter)

    # end of game
    et = datetime.datetime.now();
    logging.info('game over! LastCnt:%d, TotalCnt:%d, Time:%s' % (step_counter, total_count, str(et - st)))
    env.destroy()


if __name__ == "__main__":
    Cat = "QLearn"
    if len(sys.argv) > 1:
        Cat = sys.argv[1]

    logging.basicConfig(filename=Cat + '.log', level=logging.DEBUG, format=LOG_FORMAT)
    logging.getLogger().addHandler(logging.StreamHandler())
    env = Maze("Maze" + Cat)

    if Cat == "QLearn":
        RL = QLearningTable(actions=list(range(env.n_actions)))
    elif Cat == "Sarsa":
        RL = SarsaTable(actions=list(range(env.n_actions)))
    else:
        RL = SarsaLambdaTable(actions=list(range(env.n_actions)))


    env.after(1, update)
    logging.info("MilesDbg Start")
    env.mainloop()