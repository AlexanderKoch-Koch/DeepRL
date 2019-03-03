import numpy as np
import tensorflow as tf
import gym
from Models import simple_DNN

env = gym.make('CartPole-v0')
gamma = 0.95
tf_input = tf.placeholder(dtype=tf.float32, shape=(4,))
tf_return = tf.placeholder(dtype=tf.float32)
tf_action = tf.placeholder(dtype=tf.int32)

l0 = tf.transpose(tf.expand_dims(tf_input, axis=1))
tf_w1 = tf.Variable(tf.truncated_normal(shape=(4, 10)))
tf_b1 = tf.Variable(tf.truncated_normal(shape=(10,)))
tf_l1 = tf.nn.relu(tf.add(tf.matmul(l0, tf_w1), tf_b1))

tf_w2 = tf.Variable(tf.truncated_normal(shape=(10, 2)))
tf_b2 = tf.Variable(tf.truncated_normal(shape=(2,)))
tf_output = tf.nn.softmax(tf.add(tf.matmul(tf_l1, tf_w2), tf_b2))

tf_log_output = tf.math.multiply(tf.log(tf_output), tf.one_hot(tf_action, on_value=tf_return, depth=2))
loss = -tf.reduce_sum(tf_log_output)

tf_optimizer = tf.train.AdamOptimizer(learning_rate=0.0005).minimize(loss)


value_function = simple_DNN(architecture=[4, 10, 10, 1])
tf_summary_steps = tf.Summary()
tf_summary_steps.value.add(tag="steps", simple_value=None)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    summary_writer = tf.summary.FileWriter("./logs/CP_AC_1")
    for episode in range(10000):
        # env.render()
        state = env.reset()

        state_value = value_function.predict(sess, state)[0]
        for step in range(200):
            actions = sess.run(tf_output, feed_dict={tf_input: state})
            action = np.random.choice([0, 1], p=actions[0])

            next_state, reward, done, _ = env.step(action)
            if done:
                reward = -10

            value_next_state = value_function.predict(sess, next_state)[0]
            td_error = reward + gamma * value_next_state - state_value

            value_function.train(sess, state, [reward + gamma * value_next_state])
            sess.run(tf_optimizer, feed_dict={tf_input: state, tf_return: td_error, tf_action: action})

            state = next_state
            state_value = value_next_state
            if done:
                print(step + 1)
                tf_summary_steps.value[0].simple_value = step + 1
                summary_writer.add_summary(tf_summary_steps, episode)
                break
