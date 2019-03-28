import numpy as np
import tensorflow as tf
import constant as c
from tfutils import w, b, conv2d, max_pool2d
from sklearn.model_selection import train_test_split


class CnnNetwork():
    def __init__(self, num_steps, model_load_path):
        self._num_step = num_steps
        self._model_load_path = model_load_path

        self._learning_rate = c.CNN_LEARNING_RATE

        tf_config = tf.ConfigProto(allow_soft_placement=True)
        tf_config.gpu_options.allow_growth = True

        self.sess = tf.Session(config=tf_config)
        self._summary_writer = tf.summary.FileWriter(c.SUMMARY_DIR, graph=self.sess.graph)

        self.frame_size = c.FRAME_SIZE
        self._batch_size = c.CNN_BATCH_SIZE
        self.define_graph()

    def define_graph(self):
        batch_size = self._batch_size

        self.input_frame = tf.placeholder(
            tf.float32, shape=[batch_size, self.frame_size, self.frame_size, 1], name="s1"
        )
        self.input_frame_2 = tf.placeholder(
            tf.float32, shape=[batch_size, 27, 27, 1], name="s2"
        )
        self.input_frame_3 = tf.placeholder(
            tf.float32, shape=[batch_size, 11, 11, 1], name="s3"
        )
        self.rain_train = tf.placeholder(
            tf.float32, shape=[batch_size, 1], name="rain"
        )

        mid1 = conv2d(self.input_frame, w([8, 8, 1, 32]), b([32, ]))
        mid2 = max_pool2d(mid1)
        print(mid2)

        mid3 = tf.concat([mid2, self.input_frame_2], 3)
        print(mid3)

        mid4 = conv2d(mid3, w([6, 6, 32 + 1, 64]), b([64, ]))
        mid5 = max_pool2d(mid4)

        mid6 = tf.concat([mid5, self.input_frame_3], 3)

        mid7 = conv2d(mid6, w([3, 3, 64 + 1, 64]), b([64, ]))
        print(mid7)

        mid8 = tf.reshape(mid7, [-1, 9 * 9 * 64])
        dropout = tf.nn.dropout(mid8, keep_prob=0.5)

        regress = tf.nn.xw_plus_b(dropout, w([9 * 9 * 64, 1]), b([1]))

        self.loss = tf.losses.mean_squared_error(regress, self.rain_train)
        self.bias = tf.reduce_mean(tf.abs(regress - self.rain_train))

        self.optimizer = tf.train.AdamOptimizer(self._learning_rate).minimize(self.loss)

    def load_data(self):
        X = np.load(c.X_DIR)
        y = np.load(c.Y_DIR)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
        X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.1)
        return X_train, X_valid, X_test, y_train, y_valid, y_test

    def iterator(self, X, y):
        count = 0
        while count + self._batch_size <= len(X):
            data = X[count: count + self._batch_size]
            rain = y[count: count + self._batch_size]
            count += self._batch_size
            yield data, rain

    def scale(self, X: np.ndarray):
        shape = X.shape
        scale_2 = []
        scale_3 = []
        for x in X:
            scale_2.append(x[17: -17, 17: -17])
            scale_3.append(x[25: -25, 25: -25])
        scale_1 = X.reshape([self._batch_size, shape[1], shape[2], 1])
        scale_2 = np.asarray(scale_2).reshape([self._batch_size, 27, 27, 1])
        scale_3 = np.asarray(scale_3).reshape([self._batch_size, 11, 11, 1])
        return scale_1, scale_2, scale_3

    def train(self):
        X_train, X_valid, X_test, y_train, y_valid, y_test = self.load_data()
        init = tf.initialize_all_variables()

        with tf.Session() as sess:
            sess.run(init)

            for epoch in range(self._num_step):
                # Train
                step = 0
                for x_t, y_t in self.iterator(X_train, y_train):
                    scale_1, scale_2, scale_3 = self.scale(x_t)
                    y_t= y_t.reshape([self._batch_size, -1])

                    _, loss, bias = sess.run([self.optimizer, self.loss, self.bias],
                                                 feed_dict={self.input_frame: scale_1,
                                                            self.input_frame_2: scale_2,
                                                            self.input_frame_3: scale_3,
                                                            self.rain_train: y_t})
                    step += 1
                    if step % 200 == 0:
                        print("Iter {}: ".format(step))
                        print("     Loss:    {}".format(loss))
                        print("     Bias:    {}".format(bias))
                # Valid
                total_val_loss = 0.
                total_val_acc = 0.
                count = 0
                for x_v, y_v in self.iterator(X_valid, y_valid):
                    count += 1
                    scale_1, scale_2, scale_3 = self.scale(x_v)
                    y_v = y_v.reshape([self._batch_size, -1])
                    loss, bias = sess.run([self.loss, self.bias],
                                              feed_dict={self.input_frame: scale_1,
                                                         self.input_frame_2: scale_2,
                                                         self.input_frame_3: scale_3,
                                                         self.rain_train: y_v})
                    total_val_loss += loss
                    total_val_acc += bias
                print("-" * 50)
                print("Epoch {}: ".format(epoch))
                print("     Loss:    {}".format(total_val_loss / count))
                print("     Bias:    {}".format(total_val_acc / count))
                print("#" * 70)

    def test(self):
        X_train, X_valid, X_test, y_train, y_valid, y_test = self.load_data()
        average = np.average(y_train)
        avg = np.ones_like(y_train)
        avg[:] = average
        avg = avg.reshape([len(avg), -1])
        y_train = y_train.reshape([len(y_train), -1])
        loss = tf.losses.mean_squared_error(avg, y_train)
        bias = tf.reduce_mean(tf.abs(avg - y_train))
        with tf.Session() as sess:
            loss, bias = sess.run([loss, bias])
            print("Base line:")
            print("     Loss: {}".format(loss))
            print("     Bias: {}".format(bias))


if __name__ == "__main__":
    runner = CnnNetwork(300, "")
    runner.test()
    runner.train()
