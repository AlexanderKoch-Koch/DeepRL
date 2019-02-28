import tensorflow as tf


class simple_DNN():

    def __init__(self, input_length, output_length, weight_mean=0.0):
        self.tf_x = tf.placeholder(shape=(input_length), dtype=tf.float32, name="x")
        self.tf_y = tf.placeholder(shape=(output_length), dtype=tf.float32, name="y")

        self.l0 = tf.transpose(tf.expand_dims(self.tf_x, axis=1))
        self.w1 = tf.Variable(tf.truncated_normal(shape=(input_length, 20), mean=weight_mean))
        self.b1 = tf.Variable(tf.truncated_normal(shape=(1, 20), mean=weight_mean))
        self.l1 = tf.nn.relu(tf.add(tf.matmul(self.l0, self.w1), self.b1))

        self.w2 = tf.Variable(tf.truncated_normal(shape=(20, 20), mean=weight_mean))
        self.b2 = tf.Variable(tf.truncated_normal(shape=(1, 20), mean=weight_mean))
        self.l2 = tf.nn.relu(tf.add(tf.matmul(self.l1, self.w2), self.b2))

        self.w3 = tf.Variable(tf.truncated_normal(shape=(20, output_length), mean=weight_mean))
        self.b3 = tf.Variable(tf.truncated_normal(shape=(1, output_length), mean=weight_mean))


        self.tf_out = tf.reshape(tf.add(tf.matmul(self.l2, self.w3), self.b3), shape=(output_length, ))

        self.tf_loss = tf.losses.mean_squared_error(self.tf_y, self.tf_out)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(self.tf_loss)

        self.tf_out_summary = tf.summary.histogram("Q values", self.tf_out)

        self.copy_operation = None

    def predict(self, tf_session, input, summary=False):
        if summary:
            return tf_session.run([self.tf_out, self.tf_out_summary], feed_dict={self.tf_x: input})
        else:
            return tf_session.run([self.tf_out], feed_dict={self.tf_x: input})

    def train(self, tf_session, input, output):
        loss, _ = tf_session.run([self.tf_loss, self.optimizer], feed_dict={self.tf_x: input, self.tf_y: output})
        return loss

    def assign_weights_from(self, tf_session, source_dnn):

        if self.copy_operation is None:
            ops = []
            ops.append(self.w1.assign(source_dnn.w1))
            ops.append(self.w2.assign(source_dnn.w2))
            ops.append(self.w3.assign(source_dnn.w3))
            ops.append(self.b1.assign(source_dnn.b1))
            ops.append(self.b2.assign(source_dnn.b2))
            ops.append(self.b3.assign(source_dnn.b3))

            self.copy_operation = tf.group(ops)

        tf_session.run(self.copy_operation)


