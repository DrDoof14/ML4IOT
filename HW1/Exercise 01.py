import tensorflow as tf
import pandas as pd
import numpy as np
import argparse
import datetime
import time
# import adafruit_dht
import os


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
    temp = i[0] + ' ' + i[1]
    element = datetime.datetime.strptime(temp, "%d/%m/%Y %H:%M:%S")
    element = time.mktime(element.timetuple())
    timestampList.append(element)

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

with tf.io.TFRecordWriter(args.output) as writer:
    for row in new_csv:
        posix_value = row[0]
        temperature_value = row[1]
        humidity_value = row[2]
        print(type(temperature_value))
        if args.normalize:
            x_feature=tf.train.Feature(int64_list=tf.train.Int64List(value=[int(posix_value)]))
            y_feature = tf.train.Feature(float_list=tf.train.FloatList(value=[temperature_value, humidity_value]))
            mapping = {'integer': x_feature,'float': y_feature}
        else:
            x_feature = tf.train.Feature(
                int64_list=tf.train.Int64List(value=[int(temperature_value), int(humidity_value)]))
            y_feature = tf.train.Feature(float_list=tf.train.FloatList(value=[posix_value]))
            mapping = {'integer': x_feature,
                       'float': y_feature}
        example = tf.train.Example(features=tf.train.Features(feature=mapping))
        writer.write(example.SerializeToString())

print(os.path.getsize(args.output), "B")
