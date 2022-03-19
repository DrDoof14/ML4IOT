import tensorflow as tf
import pandas as pd
import numpy as np
import argparse
import datetime
import time
# import adafruit_dht
import os
import struct


def norm(t, t_min, t_max):
    return (t - t_min) / (t_max - t_min)


parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, help='specify the csv input file')
parser.add_argument('--output', type=str, help='specify output file name with tfrecord extension')
parser.add_argument('--normalize', action="store_true", help='for normalizing the humidity and temperature')
args = parser.parse_args()

cols = ['date', 'time']
conv = pd.read_csv(args.input, usecols=cols).values  # reading the csv file

# code to convert the date and time to POSIX fornmat
timestampList = []
for i in conv:
#     temp = i[0] + ' ' + i[1]
#     element = datetime.datetime.strptime(temp, "%d/%m/%Y %H:%M:%S")
#     element = time.mktime(element.timetuple())
    posix_timestamp = np.int64(time.mktime(time.strptime(row[DATE_CN] + " " + row[TIME_CN], "%d/%m/%Y %H:%M:%S")))
    timestampList.append(posix_timestamp)

new_csv = pd.read_csv(args.input)
new_csv = new_csv.drop(['date', 'time'], axis=1)
new_csv.insert(0, "datetime", timestampList)
if args.normalize:
    print("normalizing")
    temperature = new_csv.loc[:, 'temperature']
    humidity = new_csv.loc[:, 'humidity']
    norm_temperature = []
    norm_humidity = []
    for i in temperature:
        norm_temperature.append(norm(i, 0, 50))
    for i in humidity:
        norm_humidity.append(norm(i, 20, 90))
    Data = {'datetime': new_csv.loc[:, 'datetime'], 'temperature': norm_temperature, 'humidity': norm_humidity}
    norm_csv = pd.DataFrame(data=Data)
    new_csv = norm_csv.values
else:
    new_csv = new_csv.values

mapping={}

with tf.io.TFRecordWriter(args.output) as writer:
    for row in new_csv:
        # bytes---------------------------------------------------------------
        #         posix_value = row[0]
        #         pos_byte = bytearray(struct.pack("f", posix_value))
        #         pos_byte=bytes(pos_byte)

        #         temperature_value = row[1]
        #         temp_byte = bytearray(struct.pack("f", temperature_value))
        #         temp_byte=bytes(temp_byte)

        #         humidity_value = row[2]
        #         hum_byte = bytearray(struct.pack("f", humidity_value))
        #         hum_byte=bytes(hum_byte)

        # float---------------------------------------------------------------------

        #         posix_value = row[0]
        #         temperature_value = row[1]
        #         humidity_value = row[2]

        # int------------------------------------------------------------------------------

        posix_value = row[0]
        temperature_value = row[1]
        humidity_value = row[2]
        mapping["datetime"] = tf.train.Feature(int64_list=tf.train.Int64List(value=[posix_value]))
        if args.normalize:
            
            mapping["temperature"] = tf.train.Feature(float_list=tf.train.FloatList(value=[temperature_value]))
            mapping["humidity"] = tf.train.Feature(float_list=tf.train.FloatList(value=[humidity_value]))
            
#             x_feature = tf.train.Feature(
#                 int64_list=tf.train.Int64List(value=[posix_value, temperature_value, humidity_value]))
#             mapping = {'int': x_feature}  # we don't use the y_feature since we only have one datatype



        else:
            mapping["temperature"] = tf.train.Feature(int64_list=tf.train.Int64List(value=[temperature_value]))
            mapping["humidity"] = tf.train.Feature(int64_list=tf.train.Int64List(value=[humidity_value]))
#             x_feature = tf.train.Feature(
#                 int64_list=tf.train.Int64List(value=[posix_value, temperature_value, humidity_value]))
#             mapping = {'int': x_feature}  # we don't use the y_feature since we only have one datatype

        example = tf.train.Example(features=tf.train.Features(feature=mapping))
        writer.write(example.SerializeToString())

print(os.path.getsize(args.output), "B")
