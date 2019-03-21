import gym
import numpy as np
import tensorflow as tf

env = gym.make('CartPole-v0')


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

tf_optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)  # -tf_log_output * tf_return)

tf_summary_steps = tf.Summary()
tf_summary_steps.value.add(tag="steps", simple_value=None)


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    summary_writer = tf.summary.FileWriter("./logs/CP_RE_2")
    for episode in range(10000):
        # env.render()
        state = env.reset()
        trajectory = []
        for step in range(200):
            actions = sess.run(tf_output, feed_dict={tf_input: state})
            action = np.random.choice([0, 1], p=actions[0])
            #action = np.argmax(actions)
            next_state, reward, done, _ = env.step(action)
            if done:
                reward = -1
            trajectory.append([state, action, reward, next_state])
            state = next_state
            if done:
                print(step + 1)
                tf_summary_steps.value[0].simple_value = step + 1
                summary_writer.add_summary(tf_summary_steps, episode)
                break

        episode_return = 0
        state_value = 0
        for _, _, reward, _ in reversed(trajectory):
            state_value += reward
            episode_return += state_value

        average_reward = episode_return / len(trajectory)
        episode_return = state_value
        for state, action, reward, next_state in trajectory:
            baseline = episode_return - average_reward
            _, l1 = sess.run([tf_optimizer, tf_l1], feed_dict={tf_input: state, tf_return: baseline, tf_action: action})
            episode_return -= reward
