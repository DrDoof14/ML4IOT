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

# Code to get humidity & Tempreture from sendor and save in a CSV file
# period = 60
# freq = 5
# num_samples = int(period // freq)
# dht_device = adafruit_dht.DHT11(D4)
# for i in range(num_samples):
#     now = datetime.datetime.now()
#     temperature = dht_device.temperature
#     humidity = dht_device.humidity
#     print('{:02}/{:02}/{:04},{:02}:{:02}:{:02},{:},{:}'.format(now.day, now.month, now.year, now.hour, now.minute,
#                                                                now.second, temperature, humidity), file=fp)
#     time.sleep(args.f)
# file = pd.read_csv('rt.txt', header=None)
# file.columns = ['date', 'time', 'temperature', 'humidity']
# file.to_csv('rt.csv', index=None)

cols = ['date', 'time']
conv = pd.read_csv(args.input, usecols=cols).values  # reading the csv file

# code to convert the date and time to POSIX fornmat
timestamp = []
for i in conv:
    temp = i[0] + ' ' + i[1]
    element = datetime.datetime.strptime(temp, "%d/%m/%Y %H:%M:%S")
    tmp_tuple = element.timetuple()
    timestamp.append(time.mktime(tmp_tuple))

new_csv = pd.read_csv("rt.csv")
new_csv = new_csv.drop(['date', 'time'], axis=1)
new_csv.insert(0, "datetime", timestamp)
if args.normalize:
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
        if args.normalize:
            x_feature = tf.train.Feature(
                float_list=tf.train.FloatList(value=[posix_value,temperature_value, humidity_value]))
            mapping = {'float': x_feature}
        else:
            x_feature = tf.train.Feature(
                int64_list=tf.train.Int64List(value=[int(temperature_value), int(humidity_value)]))
            y_feature = tf.train.Feature(float_list=tf.train.FloatList(value=[posix_value]))
            mapping = {'integer': x_feature,
                       'float': y_feature}
        example = tf.train.Example(features=tf.train.Features(feature=mapping))
        writer.write(example.SerializeToString())

print(os.path.getsize(args.output), "B")
