import gym
import tensorflow as tf
import numpy as np
import random
from Experience import Experience
from Models import simple_DNN
import math
import MathAI
env = gym.make('CartPole-v0')
observation = env.reset()

eta = 0.6
gamma = 0.95
eta_discount = 0.999
rpe_scalar = 0.5
input_length = env.observation_space.shape[0]
output_length = 2
online_network = simple_DNN(architecture=[input_length,  20, 20, output_length], weight_mean=0.0)
target_network = simple_DNN(architecture=[input_length, 20, 20, output_length], weight_mean=0.0)
#rpe_network = simple_DNN(architecture=[input_length, 10, input_length], weight_mean=0.5)
memory = Experience(10000)

tf_eta_summary = tf.Summary()
tf_reward_summary = tf.Summary()
tf_rpe_summary = tf.Summary()
tf_reward_summary.value.add(tag="steps", simple_value=None)
tf_eta_summary.value.add(tag="eta", simple_value=0)
# tf_rpe_summary.value.add(tag="Reward prediction error", simple_value=0)


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    summary_writer = tf.summary.FileWriter("./logs/CP_DQN_Target_Prioritized_1", sess.graph)

    for episode in range(20000):
        observation = env.reset()
        eta *= eta_discount
        rpe_scalar *= 1.1
        reward_sum = 0
        max_distance = -0.5
        for step in range(1000):
            # env.render()
            # rpe_scalar *= 1.001
            q, q_summary = online_network.predict(sess, observation, summary=True)
            # state_prediciton = rpe_network.predict(sess, observation)
            #print("rpe: " + str(reward_prediciton[0]))
            summary_writer.add_summary(q_summary, episode * 200 + step)
            if random.randint(0, 100) < (eta*100):
                action = env.action_space.sample()
            else:
                action = np.argmax(q)

            observation_next, reward, done, _ = env.step(action)
            if done:
                reward = -1
            # max_distance = max(max_distance, observation[0])
            # reward_sum += reward
            #rpe_network.train(sess, observation, observation_next)
            #rpe = MathAI.sigmoid(MathAI.euclidean_distance(np.array(state_prediciton), observation_next))
            #tf_rpe_summary.value[0].simple_value = rpe
            #summary_writer.add_summary(tf_rpe_summary, episode * 200 + step)
            #reward = reward + math.fabs(observation[0] + 0.5)
            #print(reward)

            priority = 100
            memory.store(observation, action, reward, observation_next, done, priority)
            observation = observation_next
            if done:
                print(step + 1)
                #tf_eta_summary.value[0].simple_value = max_distance
                #summary_writer.add_summary(tf_eta_summary, episode)
                tf_reward_summary.value[0].simple_value = step + 1
                summary_writer.add_summary(tf_reward_summary, episode)
                tf_eta_summary.value[0].simple_value = eta
                summary_writer.add_summary(tf_eta_summary, episode)
                break

        for i in range(20):
            [observation, action, reward, observation_next, done, priority], index = memory.sample(prioritized_replay=True)
            q_state = online_network.predict(sess, observation)
            target = np.copy(q_state)

            if done:
                target[action] = reward
            else:
                q_next = target_network.predict(sess, observation_next)
                target[action] = reward + gamma * np.max(q_next)

            td_error = np.abs(q_state[action] - target[action])
            memory.update_td_error(index, td_error)
            loss = online_network.train(sess, observation, target)

        if episode % 20 == 0:
            # update weights of target network every n episodes
            target_network.assign_weights_from(sess, online_network)
