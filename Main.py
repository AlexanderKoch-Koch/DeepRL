import gym
import tensorflow as tf
import numpy as np
import random
from Experience import Experience
from Models import simple_DNN
import math

env = gym.make('MountainCar-v0')
observation = env.reset()

eta = 0.9
gamma = 0.95
eta_discount = 0.9995
input_length = env.observation_space.shape[0]
output_length = 3
online_network = simple_DNN(input_length, output_length, weight_mean=0.5)
target_network = simple_DNN(input_length, output_length, weight_mean=0.5)
rpe_network = simple_DNN(input_length, 1)
memory = Experience(10000)

tf_eta_summary = tf.Summary()
tf_steps_summary = tf.Summary()
tf_steps_summary.value.add(tag="steps", simple_value=None)
tf_eta_summary.value.add(tag="eta", simple_value=0)


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    summary_writer = tf.summary.FileWriter("./logs/MC_DDQN_9", sess.graph)

    for episode in range(2000):
        observation = env.reset()
        eta *= eta_discount
        reward_sum = 0
        for step in range(1000):
            # env.render()
            q, q_summary = online_network.predict(sess, observation, summary=True)
            reward_prediciton = rpe_network.predict(sess, observation)
            print("rpe: " + str(reward_prediciton[0]))
            summary_writer.add_summary(q_summary, episode * 200 + step)
            if random.randint(0, 100) < (eta*100):
                action = env.action_space.sample()
            else:
                action = np.argmax(q)

            observation_next, reward, done, _ = env.step(action)

            rpe_network.train(sess, observation, [reward])
            rpe = math.pow(reward_prediciton[0] - reward, 2)
            reward = reward + rpe
            print(reward)
            reward_sum += reward
            priority = rpe
            memory.store(observation, action, reward, observation_next, done, priority)
            observation = observation_next
            if done:
                tf_eta_summary.value[0].simple_value = eta
                summary_writer.add_summary(tf_eta_summary, episode)
                tf_steps_summary.value[0].simple_value = reward_sum
                summary_writer.add_summary(tf_steps_summary, episode)
                break

        for i in range(20):
            [observation, action, reward, observation_next, done, priority], index = memory.sample(prioritized_replay=True)
            q_state = online_network.predict(sess, observation)
            target = np.copy(q_state[0])

            if done:
                target[action] = reward
            else:
                q_next = target_network.predict(sess, observation_next)
                target[action] = reward + gamma * np.max(q_next[0])

            td_error = np.abs(q_state[0][action] - target[action])
            memory.update_td_error(index, td_error)
            loss = online_network.train(sess, observation, target)

        if episode % 2 == 0:
            # update weights of target network every n episodes
            target_network.assign_weights_from(sess, online_network)
