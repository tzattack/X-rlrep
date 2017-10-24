from rllab.algos.dueling_dqn import DuelingDQN
import gym

env_name = 'Pendulum-v0'
env = gym.make(env_name)

algo = DuelingDQN(
    env=env,
    memory_size=3000,
    action_space=25,
    learning_rate=0.001,
    reward_decay=0.9,
    e_greedy=0.9,
    replace_target_iter=200,
    memory_size_2=500,
    batch_size=32,
)
algo.train()
