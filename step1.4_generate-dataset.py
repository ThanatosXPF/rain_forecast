import os
import pymysql
import numpy as np
import constant as c


class Extractor:
    def __init__(self):
        self.db = pymysql.connect("localhost", "root", "ices702", "Precipitation",
                             cursorclass=pymysql.cursors.SSCursor, charset="utf8")
        self.cursor = self.db.cursor()

        self._range = [(0), (0, 5), (5, 10), (10, 15), (15, 30), (30)]
        self._limit = [30000, 30000, 30000, 30000, 30000, 30000]

    def extract(self):
        # refs = np.ndarray(shape=(c.CNN_BATCH_SIZE, c.ECHO_AREA, c.ECHO_AREA, 10))
        # rains = []
        index = 0
        count = 1
        for i, r in enumerate(self._range) :
            self.db = pymysql.connect("localhost", "root", "ices702", "Precipitation",
                             cursorclass=pymysql.cursors.SSCursor, charset="utf8")
            self.cursor = self.db.cursor()
            if r == 0:
                sql = "SELECT ref, rain FROM hour_rain WHERE rain = {};".format(r)
            elif r == 30:
                sql = "SELECT ref, rain FROM hour_rain WHERE rain > {};".format(r)
            else:
                sql = "SELECT ref, rain FROM hour_rain WHERE rain >= {} and rain < {};".format(r[0], r[1])
            print(sql)
            self.cursor.execute(sql)
            print('executed!')
            for j in range(self._limit[i]):
                data = self.cursor.fetchone()
                # for data in self.cursor:
                if data is None:
                    break
                # print(data)
                ref = np.loads(data[0])
                rain = data[1]
                # refs[index, :, :, :] = ref[0]
                # rains.append(rain)
                self.save(ref, rain, count)
                count += 1

                # index += 1
                # if index == c.CNN_BATCH_SIZE:
                #     self.save(refs, rains, count)
                #     print("extract", count)
                #     refs = np.ndarray(shape=(c.CNN_BATCH_SIZE, c.ECHO_AREA,
                #                              c.ECHO_AREA, 10))
                #     rains = []
                #     count += 1
                #     index = 0

    def save(self, refs, rains, count):
        # rains = np.asarray(rains).reshape([len(rains), -1])
        data = [refs, rains]
        np.save(os.path.join(c.H1_RESAMPLE, "{}.npy".format(count)), data)
        print("extract", count)


if __name__ == '__main__':
    ex = Extractor()
    ex.extract()

