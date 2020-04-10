# coding: utf-8
import base64
import csv
import pandas as pd
import math

# 0	negative
# 2	neutral
# 4	positive

data_0 = open('../data/data_0.txt', 'a', encoding='utf-8')
data_2 = open('../data/data_2.txt', 'a', encoding='utf-8')
data_4 = open('../data/data_4.txt', 'a', encoding='utf-8')
index = 0

with open("../dataset/training.1600000.processed.noemoticon.csv", "r", encoding='ISO-8859-1') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        try:
            kinds = line[0]
            content = line[5]
            # print(description)
            # for kind in eval(kinds):
            #     print(math.floor(kind / 10000))
            kind = int(kinds)
            if kind == 0:
                content = content.replace('\n', ' ')
                data_0.write(str(content) + '\n')
            elif kind == 2:
                content = content.replace('\n', ' ')
                data_2.write(str(content) + '\n')
            elif kind == 4:
                content = content.replace('\n', ' ')
                data_4.write(str(content) + '\n')
            print(index)
            index += 1

        except:
            pass


# a = 'UHJvdmlkZXMgaW5mb3JtYXRpb24gdGVjaG5vbG9neSBzZXJ2aWNlcw=='
# print(base64.b64decode(a))

