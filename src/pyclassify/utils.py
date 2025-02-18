from math import sqrt
import yaml
import os
import csv

def distance(point1: list[float], point2: list[float]):
    n = len(point1)
    sum = 0

    for i in range(n):
        sum += (point1[i] - point2[i])**2
    return sqrt(sum)

def majority_vote(neighbors: list[int]):
    unique_labels = set(neighbors)
    max_label = None
    max_count = 0

    for label in unique_labels:
        tmp = 0
        for element in neighbors:
            if element == label:
                tmp += 1

        if tmp >  max_count:
            max_count = tmp
            max_label = label 

    return max_label

def read_config(file):
   filepath = os.path.abspath(f'{file}.yaml')
   with open(filepath, 'r') as stream:
      kwargs = yaml.safe_load(stream)
   return kwargs

def read_file(pth):
    features = []
    labels = []
    with open(pth, "r") as file:
        lines = file.readlines()

    for line in lines:
        feature = line.strip().split(",")[:-1]
        new_feature = []
        for x in feature:
            new_feature.append(float(x))
        label = line.strip().split(",")[-1]
        features.append(new_feature)
        if label == "g":
            labels.append(0)
        #Per generalizzare a binary classification
        else:
            labels.append(1)

    return features, labels

    

