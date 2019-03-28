import os, glob
import logging, datetime
import numpy as np
import pandas as pd
import tensorflow as tf
import constant as c
from PIL import Image
from tfutils import w, b, conv2d, max_pool2d
from sklearn.model_selection import train_test_split
from concurrent.futures import ThreadPoolExecutor, wait
from multiprocessing import Pool


def config_log():
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                        datefmt='%a, %d %b %Y %H:%M:%S',
                        filename=os.path.join(c.MODEL_DIR, "train.log"),
                        filemode='w')

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s  %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)


def generate_ref_path(date):
    """
    Compose the file path of ref file according to the date.
    :param date:
    :return:
    """
    year = date[2:4]
    folder = "/extend/14-17_2500_radar/" + year + "_2500_radar"
    file = glob.glob(os.path.join(folder, "cappi_ref_"+date+"*"))
    if file:
        return os.path.join(folder, file[0])
    else:
        return None


class CnnNetwork():
    def __init__(self, num_steps, model_load_path, mode="train"):
        self._npy_executor_pool = ThreadPoolExecutor(max_workers=16)

        self._num_step = num_steps
        self._model_load_path = model_load_path
        tf_config = tf.ConfigProto(allow_soft_placement=True)
        tf_config.gpu_options.allow_growth = True

        self._learning_rate = c.CNN_LEARNING_RATE

        self.sess = tf.Session(config=tf_config)
        self._summary_writer = tf.summary.FileWriter(c.SUMMARY_DIR, graph=self.sess.graph)

        self._height = c.HEIGHT
        self._width = c.WIDTH

        self._size = c.ECHO_AREA
        
        if mode == "train":
            self._drop_rate = 0.5
            self._batch_size = c.CNN_BATCH_SIZE
        else:
            self._drop_rate = 1
            self._batch_size = c.CNN_PREDICTION_SIZE
        self.frame_size = c.FRAME_SIZE
        # self._batch_size = c.CNN_BATCH_SIZE
        self.define_graph()

        self.saver = tf.train.Saver(keep_checkpoint_every_n_hours=2)
        self.sess = tf.Session(config=tf_config)
        self.sess.run(tf.global_variables_initializer())

        if model_load_path:
            self.saver.restore(self.sess, model_load_path)
            print("Model load from {}".format(model_load_path))

    def define_graph(self):
        batch_size = self._batch_size

        self.input_frame = tf.placeholder(
            tf.float32, shape=[batch_size, self.frame_size, self.frame_size, 10], name="s1"
        )
        self.input_frame_2 = tf.placeholder(
            tf.float32, shape=[batch_size, 27, 27, 10], name="s2"
        )
        self.input_frame_3 = tf.placeholder(
            tf.float32, shape=[batch_size, 11, 11, 10], name="s3"
        )
        self.rain_train = tf.placeholder(
            tf.float32, shape=[batch_size, 1], name="rain"
        )
        self.rain_train_2 = tf.placeholder(
            tf.float32, shape=[batch_size, 1], name="rain_2"
        )

        mid1 = conv2d(self.input_frame, w([8, 8, 10, 32]), b([32, ]))
        mid2 = max_pool2d(mid1)
        print(mid2)

        mid3 = tf.concat([mid2, self.input_frame_2], 3)
        print(mid3)

        mid4 = conv2d(mid3, w([6, 6, 32 + 10, 64]), b([64, ]))
        mid5 = max_pool2d(mid4)

        mid6 = tf.concat([mid5, self.input_frame_3], 3)

        mid7 = conv2d(mid6, w([3, 3, 64 + 10, 64]), b([64, ]))
        print(mid7)

        mid8 = tf.reshape(mid7, [-1, 9 * 9 * 64])
        dropout = tf.nn.dropout(mid8, keep_prob=self._drop_rate)
        print(dropout)
        regress = tf.nn.xw_plus_b(dropout, w([9 * 9 * 64, 1]), b([1]))
        print(regress)

        self.loss = tf.losses.mean_squared_error(regress, self.rain_train)
        self.bias = tf.reduce_mean(tf.abs(regress - self.rain_train))
        self.error_rate = tf.reduce_mean(tf.abs(regress - self.rain_train) / self.rain_train_2) * 100

        self.optimizer = tf.train.AdamOptimizer(self._learning_rate).minimize(self.loss)
        self.pred_test = regress

    def load_data(self):
        npys = os.listdir(c.H1_DIR)

        X = []
        y = []
        
        for npy in npys:
            pair = np.load(os.path.join(c.H1_DIR, npy))
            X.append(pair[0])
            y.append(pair[1])
            
        X = np.asarray(X)
        y = np.asarray(y)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
        X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.1)
        return X_train, X_valid, X_test, y_train, y_valid, y_test

    def read_npy(self, path, X, y, i):
        X[i], y[i] = np.load(path)

    def quick_read_npys(self, path_list):
        X = np.empty((c.CNN_BATCH_SIZE, c.ECHO_AREA, c.ECHO_AREA, 10), dtype=np.uint8)
        y = np.empty(c.CNN_BATCH_SIZE, dtype=np.float32)
        future_obj = []
        for i, path in enumerate(path_list):
            obj = self._npy_executor_pool.submit(self.read_npy, path, X, y, i)
            future_obj.append(obj)
        wait(future_obj)
        return X, y

    def iterator(self, mode="train"):
        # count = 0
        # while count + self._batch_size <= len(X):
        #     data = X[count: count + self._batch_size]
        #     rain = y[count: count + self._batch_size]
        #     count += self._batch_size
        #     yield data, rain
        target_dir = c.H1_RESAMPLE

        npys = os.listdir(target_dir)
        total = len(npys)
        n_2 = int(total * 0.7)
        n_3 = int(total * 0.8)
        if mode == "train":
            npys = npys[:n_2]
        elif mode == "valid":
            npys = npys[n_2:n_3]
        elif mode == "test":
            npys = npys[n_3:]
        elif mode == "all":
            npys = npys[:]

        npy_paths = []
        count = 0
        for npy in npys:
            count += 1
            npy_paths.append(os.path.join(target_dir, npy))
            if len(npy_paths) == c.CNN_BATCH_SIZE:
                X, y = self.quick_read_npys(npy_paths)
                yield X, y
                npy_paths = []
            # if count % 1000 == 0:
            #     print("itered", count)

    def scale(self, X: np.ndarray, mode="normal"):
        shape = X.shape
        batch_size = self._batch_size
        if mode == "predict":
            batch_size = 1
        # scale_2 = []
        # scale_3 = []
        # for x in X:
        scale_2 = X[:, 17: -17, 17: -17, :]
        scale_3 = X[:, 25: -25, 25: -25, :]
        scale_1 = X.reshape([batch_size, shape[1], shape[2], 10])
        scale_2 = np.asarray(scale_2).reshape([batch_size, 27, 27, 10])
        scale_3 = np.asarray(scale_3).reshape([batch_size, 11, 11, 10])
        return scale_1, scale_2, scale_3

    def train(self):
        # X_train, X_valid, X_test, y_train, y_valid, y_test = self.load_data()
        # print("load complete!")

        for epoch in range(self._num_step):
            # Train
            step = 0
            for x_t, y_t in self.iterator():
                scale_1, scale_2, scale_3 = self.scale(x_t)
                y_t= y_t.reshape([self._batch_size, -1])
                y_t_2 = y_t.copy()
                y_t_2[y_t_2 == 0] = 0.5
                _, loss, bias, error = self.sess.run([self.optimizer, self.loss, self.bias, self.error_rate],
                                             feed_dict={self.input_frame: scale_1,
                                                        self.input_frame_2: scale_2,
                                                        self.input_frame_3: scale_3,
                                                        self.rain_train: y_t,
                                                        self.rain_train_2: y_t_2})
                step += 1
                if step % 200 == 0:
                    logging.info(
                        "Iter {}: \n\tLoss:  {}\n\tBias:  {}\n\tError: {}".format(step, loss, bias, error))

                    # print("Iter {}: ".format(step))
                    # print("     Loss:    {}".format(loss))
                    # print("     Bias:    {}".format(bias))
                    # print("     Error:   {}".format(error))
            self.valid(epoch)

    def valid(self, epoch):
        print("valid")
        total_val_loss = 0.
        total_val_acc = 0.
        total_val_err = 0.
        count = 0
        for x_v, y_v in self.iterator(mode="valid"):
            count += 1
            scale_1, scale_2, scale_3 = self.scale(x_v)
            y_v = y_v.reshape([self._batch_size, -1])
            y_v_2 = y_v.copy()
            y_v_2[y_v_2 == 0] = 0.5
            loss, bias, error = self.sess.run([self.loss, self.bias, self.error_rate],
                                              feed_dict={self.input_frame: scale_1,
                                                         self.input_frame_2: scale_2,
                                                         self.input_frame_3: scale_3,
                                                         self.rain_train: y_v,
                                                         self.rain_train_2:y_v_2})
            total_val_loss += loss
            total_val_acc += bias
            total_val_err += error
        print("-" * 50)
        # print("Epoch {}: ".format(epoch))
        # print("     Loss:    {}".format(total_val_loss / count))
        # print("     Bias:    {}".format(total_val_acc / count))
        # print("     Error:   {}".format(total_val_err / count))
        loss = total_val_loss / count
        bias = total_val_acc / count
        error = total_val_err / count
        logging.info("Valid epoch{}: \n\tLoss:  {}\n\tBias:  {}\n\tError: {}".format(epoch, loss, bias, error))

        print("#" * 70)
        self.saver.save(self.sess, os.path.join(c.MODEL_DIR, 'model.ckpt'), global_step=epoch)

    def base_line_avg(self):
        """
        New Data set:
            Loss:  178.39854431152344
            Bias:  9.449376106262207
            Error: 781.7628173828125

        :return:
        """
        # y = []
        # for x_t, y_t in self.iterator(mode='all'):
        #     y.extend(y_t)
        y = np.load(os.path.join(c.SAVE_DIR, "y.npy"))

        average = np.average(y)
        avg = np.ones_like(y)
        avg[:] = average
        avg = avg.reshape([len(avg), -1])
        y_train = y.reshape([len(y), -1])

        y_train_2 = y_train.copy()
        y_train_2[y_train_2 == 0] = 0.5

        loss = tf.losses.mean_squared_error(avg, y_train)
        bias = tf.reduce_mean(tf.abs(avg - y_train))
        error = tf.reduce_mean(tf.abs(avg - y_train) / y_train_2) * 100
        with tf.Session() as sess:
            loss, bias, error = sess.run([loss, bias, error])
            print("Base line:")
            print("     Loss:  {}".format(loss))
            print("     Bias:  {}".format(bias))
            print("     Error: {}".format(error))

    def pixel_to_rainfall(self, dBZ):
        a = c.ZR_a
        b = c.ZR_b
        dBR = (dBZ - 10.0 * np.log10(a)) / b
        rainfall_intensity = np.power(10, dBR / 10.0)
        return rainfall_intensity

    def base_line(self):
        """
        Base line:
             Loss:  394.7578430175781
             Bias:  9.565366666134354
             Error: 281.5391674761956
        New Data set
             Loss:  454.6909484863281
             Bias:  10.312403917175013
             Error: 96.59923796656459

        :return:
        """
        # y_truth = []
        # dBZ = []
        # for x_t, y_t in self.iterator(mode='all'):
        #     for i in range(x_t.shape[0]):
        #         dBZ.append(x_t[i,c.ECHO_AREA // 2, c.ECHO_AREA // 2, -1])
        #         # rain.append(self.pixel_to_rainfall(dBZ))
        #     y_truth.extend(y_t)
        #
        # dBZ = np.asarray(dBZ)
        # rain = self.pixel_to_rainfall(dBZ)
        # rain = rain.reshape([len(rain), -1])
        #
        # y_truth = np.asarray(y_truth)

        # np.save(os.path.join(c.SAVE_DIR, "dBZ.npy"), dBZ)
        # np.save(os.path.join(c.SAVE_DIR, "y.npy"), y_truth)
        dBZ = np.load(os.path.join(c.SAVE_DIR, "dBZ.npy"))
        y_truth = np.load(os.path.join(c.SAVE_DIR, "y.npy"))
        rain = self.pixel_to_rainfall(dBZ)
        rain = rain.reshape([len(rain), -1])

        y_truth = y_truth.reshape([len(y_truth), -1])

        y_truth_2 = y_truth.copy()
        y_truth_2[y_truth_2 == 0] = 0.5

        loss = tf.losses.mean_squared_error(rain, y_truth)
        bias = tf.reduce_mean(tf.abs(rain - y_truth))
        error = tf.reduce_mean(tf.abs(rain - y_truth) / y_truth_2) * 100

        with tf.Session() as sess:
            loss, bias, error = sess.run([loss, bias, error])
            logging.info("Base line: \n\tLoss:  {}\n\tBias:  {}\n\tError: {}".format(loss, bias, error))
            # print("     Loss:  {}".format(loss))
            # print("     Bias:  {}".format(bias))
            # print("     Error: {}".format(error))

    def test(self):
        total_val_loss = 0.
        total_val_acc = 0.
        count = 0
        for x_v, y_v in self.iterator(mode="test"):
            count += 1
            scale_1, scale_2, scale_3 = self.scale(x_v)
            y_v = y_v.reshape([self._batch_size, -1])
            loss, bias = self.sess.run([self.loss, self.bias],
                                  feed_dict={self.input_frame: scale_1,
                                             self.input_frame_2: scale_2,
                                             self.input_frame_3: scale_3,
                                             self.rain_train: y_v})
            total_val_loss += loss
            total_val_acc += bias
        print("-" * 50)
        print("Test: ")
        print("     Loss:    {}".format(total_val_loss / count))
        print("     Bias:    {}".format(total_val_acc / count))
        print("#" * 70)

    def predict_rain_map(self, refs):
        print("Start prediction")
        rain_map = np.zeros((self._height, self._width))
        size = self._size
        center = [size//2, size//2]
        p_size = self._batch_size
        targets = []
        centers = []
        # print(refs.shape)
        while 0<= center[0]-size//2 <= center[0]+size//2+1 <= 700:
            center[1] = size//2
            while 0 <= center[1]-size//2 <= center[1]+size//2+1 < 900:
                # print(center[0] - c.ECHO_AREA // 2, center[0] + c.ECHO_AREA // 2 + 1)
                target = refs[:,
                              center[0] - size // 2 : center[0] + size // 2 + 1,
                              center[1] - size // 2 : center[1] + size // 2 + 1,
                              :]
                targets.append(target[0])
                centers.append(center.copy())
                if len(targets) == p_size:
                    targets = np.asarray(targets)
                    # print(targets.shape)
                    scale_1, scale_2, scale_3 = self.scale(targets)
                    preds, *_ = self.sess.run([self.pred_test],
                                              feed_dict={self.input_frame: scale_1,
                                                         self.input_frame_2: scale_2,
                                                         self.input_frame_3: scale_3,
                                                         self.rain_train: np.zeros((self._batch_size, 1))})
                    for i in range(len(centers)):
                        c = centers[i]
                        p = preds[i]
                        rain_map[c[0], c[1]] = p[0]
                        # print(c, p)
                    # print(datetime.datetime.now(), centers[0], preds[0])
                    targets = []
                    centers = []
                center[1] += 1
            center[0] += 1
        rain_map[rain_map < 0] = 0
        # print(rain_map)
        print("Done")
        return rain_map.astype(np.uint8)

    def do_predict(self, target, center, rain_map):
        scale_1, scale_2, scale_3 = self.scale(target)
        preds, *_ = self.sess.run([self.pred_test],
                                  feed_dict={self.input_frame: scale_1,
                                             self.input_frame_2: scale_2,
                                             self.input_frame_3: scale_3,
                                             self.rain_train: np.zeros((self._batch_size, 1))})
        for c, p in zip(center, preds):
            rain_map[c[0], c[1]] = p
        print(datetime.datetime.now(), center[0], preds[0])

    def read_ref(self, date, nums=1):
        """
        Extract radar echo area according to the aws position.
        :param date: 12 character date
        :param nums: batch size
        :return: 61 * 61 radar echo map with shape [1, size, size, 10]
        """
        date_list = pd.date_range(end=date, periods=10, freq="6T")
        ref_paths = []
        for date in date_list:
            date = date.strftime("%Y%m%d%H%M")
            ref_path = generate_ref_path(date)
            if not ref_path:
                # print(ref_path, "not exist")
                return None
            ref_paths.append(ref_path)
        # refs = []
        data = np.empty([1, c.HEIGHT, c.WIDTH, 10])
        for i, ref_path in enumerate(ref_paths):
            ref = np.fromfile(ref_path, dtype=np.uint8).reshape(700, 900)
            ref[ref >= 80] = 0
            ref[ref <= 15] = 0
            # refs.append(ref)
            data[:, :, :, i] = ref
        print("Read", ref_paths[-1])
        return data


def do_prediction(start, end):
    runner = CnnNetwork(300, "/extend/rain_data/Save/Model/model.ckpt-10",
                        mode="test")
    print("loaded!")
    times = pd.date_range(start, end, freq="6T")
    if not os.path.exists(c.RESULT_DIR):
        os.makedirs(c.RESULT_DIR)
    for t in times:
        date = t.strftime("%Y%m%d%H%M")
        refs = runner.read_ref(date)
        if refs is not None:
            rain = runner.predict_rain_map(refs)
            rain = Image.fromarray(rain)
            rain.save(os.path.join(c.RESULT_DIR, date+".png"))
            print("Done for", date)


if __name__ == "__main__":
    # runner = CnnNetwork(300, "")
    # config_log()
    # runner.base_line()
    # runner.base_line_avg()
    # runner.train()
    # from PIL import Image
    # runner = CnnNetwork(300, "/extend/rain_data/Save/Model/model.ckpt-10",
    #                     mode="test")
    # print("loaded!")
    # refs = runner.read_ref("201809161400")
    # rain = runner.predict_rain_map(refs)
    # print(rain.shape)
    # rain = Image.fromarray(rain)
    # rain.save("result.png")
    do_prediction("201809161300", "201809170000")

