"""
This step is designed to generate data set of rainfall.
First traverse all the aws files and judge whether this aws station locate in
valid area which base on radar coverage. Then extract a predetermined size of
radar echo map from the corresponding ref file.
"""
import os
import numpy as np
import pandas as pd
import constant as c
import pymysql


def resolve_aws_name(aws):
    """
    Resolve the file name of aws file and extract the time and the number of
    records.
    :param aws:
    :return date, aws_num:
    """
    _, date, aws_num = aws.split("_")
    aws_num = aws_num[:-4]
    return date, int(aws_num)


def generate_ref_path(date):
    """
    Compose the file path of ref file according to the date.
    :param date:
    :return:
    """
    year = date[2:4]
    folder = "/extend/14-17_2500_radar/" + year + "_2500_radar"
    return folder + "/cappi_ref_" + date + "_2500_0.ref"


def read_ref_region(date, aws_lo, aws_la, nums=1):
    """
    Extract radar echo area according to the aws position.
    :param date: 12 character date
    :param aws_lo: longitude of the aws station
    :param aws_la: latitude of the aws station
    :return: 61 * 61 radar echo map with shape [1, size, size, 10]
    """
    date_list = pd.date_range(end=date, periods=10, freq="6T")
    ref_paths = []
    for date in date_list:
        date = date.strftime("%Y%m%d%H%M")
        ref_path = generate_ref_path(date)
        if not os.path.exists(ref_path):
            return None
        ref_paths.append(ref_path)
    # refs = []
    data = np.empty([nums, c.ECHO_AREA, c.ECHO_AREA, 10])
    for i, ref_path in enumerate(ref_paths):
        ref = np.fromfile(ref_path, dtype=np.uint8).reshape(700, 900)
        ref = ref[aws_lo - c.ECHO_AREA // 2: aws_lo + c.ECHO_AREA // 2 + 1,
                  aws_la - c.ECHO_AREA // 2: aws_la + c.ECHO_AREA // 2 + 1]

        ref[ref >= 80] = 0
        ref[ref <= 15] = 0
        # refs.append(ref)
        data[:, :, :, i] = ref
    return data


def aws_valid(aws_longitude, aws_latitude):
    """
    Find whether this aws station is in the valid area.
    :param aws_longitude: longitude of aws
    :param aws_latitude: latitude of aws
    :return: Boolean
    """
    if c.VALID_LONGITUDE[0] <= aws_longitude <= c.VALID_LONGITUDE[1] and \
            c.VALID_LATITUDE[0] <= aws_latitude <= c.VALID_LATITUDE[1]:

        return True
    else:
        return False


def aws_data_iterator(aws_path):
    """
    A iterator of aws records and yield rainfall and aws position.
    5 min Rain
    :param aws_path: path to a aws record file
    :return:
    """
    AWSdata = np.loadtxt(aws_path)
    for line in AWSdata:
        rain = float(line[-8])
        aws_longitude = int(line[2])
        aws_latitude = int(line[1])
        if aws_valid(aws_longitude, aws_latitude) and rain >= 0:
            yield rain, aws_longitude, aws_latitude


def extract_data():
    """
    Main method of this script.
    :return:
    """

    db = pymysql.connect("10.249.178.165", "remote", "ices702", "Precipitation", charset="utf8")
    cursor = db.cursor()

    count = 0
    zero = 0
    AWS_path = c.AWS_PATH
    for root, _, files in os.walk(AWS_path):
        for aws in files:
            if not aws.endswith(".txt"):
                continue
            date, aws_num = resolve_aws_name(aws)

            if aws_num == 0 or date[2:4] <= "13" or int(date[-2:]) %6 != 0:
                continue

            aws_path = os.path.join(root, aws)

            for rain, aws_long, aws_la in aws_data_iterator(aws_path):

                ref = read_ref_region(date, aws_long, aws_la)

                if ref is None:
                    # print("miss ref")
                    continue
                date = pd.to_datetime(date).strftime("%Y-%m-%d %H:%M:00")
                data = (ref.dumps(), rain, date)
                # print(data[0])

                cursor.execute("SELECT COUNT(*) FROM hour_rain WHERE ref=%s and rain=%s and date=%s", data)
                results = cursor.fetchall()
                print(results)
                if len(results) == 0:
                    cursor.execute("insert into hour_rain value (%s,%s,%s)", data)
                    db.commit()
                    print("hit! {}".format(data[2]))
                    count += 1
                    if count % 100 == 0:
                        print("{} extracted!".format(count))
                else:
                    print("pass {}".format(data[2]))





if __name__ == "__main__":
    extract_data()
