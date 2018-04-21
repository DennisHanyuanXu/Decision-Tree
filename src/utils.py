#!/bin/env python
#-*- encoding: utf-8 -*-

import csv
import conf


def count(dataset):
    rst = dict()

    for data in dataset:
        label = data[-1]
        if label not in rst.keys(): 
            rst[label] = 0
        rst[label] += 1
    return rst


def convert_type(s):
    s = s.strip()

    try:
        return float(s) if '.' in s else int(s)
    except ValueError:
        return s


def load_dataset():
    reader = csv.reader(open(conf.filepath, 'rt'))

    headings = dict()
    for i, heading in enumerate(next(reader)):
        headings[i] = str(heading)

    dataset = [[convert_type(item) for item in row] for row in reader]
    return (headings, dataset)
