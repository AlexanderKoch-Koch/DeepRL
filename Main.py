import gym
import tensorflow as tf
import numpy as np
import random
from Experience import Experience
from Models import simple_DNN
import math
import MathAI
env = gym.make('MountainCar-v0')
observation = env.reset()

eta = 0.0
gamma = 0.9
eta_discount = 1.0#0.9995
rpe_scalar = 0.5
input_length = env.observation_space.shape[0]
output_length = 3
online_network = simple_DNN(architecture=[input_length,  20, 20, output_length], weight_mean=0.0)
target_network = simple_DNN(architecture=[input_length, 20, 20, output_length], weight_mean=0.0)
rpe_network = simple_DNN(architecture=[input_length, 10, input_length], weight_mean=0.5)
memory = Experience(10000)

tf_eta_summary = tf.Summary()
tf_reward_summary = tf.Summary()
tf_rpe_summary = tf.Summary()
tf_reward_summary.value.add(tag="Reward sum", simple_value=None)
tf_eta_summary.value.add(tag="Max distance", simple_value=0)
tf_rpe_summary.value.add(tag="Reward prediction error", simple_value=0)


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    summary_writer = tf.summary.FileWriter("./logs/MC_2", sess.graph)

    for episode in range(20000):
        observation = env.reset()
        eta *= eta_discount
        rpe_scalar *= 1.1
        reward_sum = 0
        max_distance = -0.5
        for step in range(1000):
            # env.render()
            rpe_scalar *= 1.001
            q, q_summary = online_network.predict(sess, observation, summary=True)
            state_prediciton = rpe_network.predict(sess, observation)
            #print("rpe: " + str(reward_prediciton[0]))
            summary_writer.add_summary(q_summary, episode * 200 + step)
            if random.randint(0, 100) < (eta*100):
                action = env.action_space.sample()
            else:
                action = np.argmax(q)

            observation_next, reward, done, _ = env.step(action)
            max_distance = max(max_distance, observation[0])
            reward_sum += reward
            rpe_network.train(sess, observation, observation_next)
            rpe = MathAI.sigmoid(MathAI.euclidean_distance(np.array(state_prediciton)[0], observation_next))
            tf_rpe_summary.value[0].simple_value = rpe
            summary_writer.add_summary(tf_rpe_summary, episode * 200 + step)
            reward = reward + math.fabs(observation[0] + 0.5)
            #print(reward)

            priority = rpe
            memory.store(observation, action, reward, observation_next, done, priority)
            observation = observation_next
            if done:
                tf_eta_summary.value[0].simple_value = max_distance
                summary_writer.add_summary(tf_eta_summary, episode)
                tf_reward_summary.value[0].simple_value = reward_sum
                summary_writer.add_summary(tf_reward_summary, episode)
                break

        for i in range(1):
            [observation, action, reward, observation_next, done, priority], index = memory.sample(prioritized_replay=False)
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
