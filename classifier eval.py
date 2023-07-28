import numpy as np
import pandas as pd
import joblib, pickle, json, os

stats_file = r"C:\Users\Felix\Dropbox\My PC (LAPTOP-J41MAND4)\Users\Felix\Documents\Philosophy+\Cognitive science\Aptima Coding\Classifyers\simulation stats\2023-07-28 15.22.02.159564stats.pkl"
with open(stats_file, 'rb') as file:
    stats = pickle.load(file)

for key, item in stats.items():
    if None in item:
        print(item)

accuracy = stats['accuracy']
num_events = stats['event_num']

total_events = 0
for x in num_events:
    if x == 'new_file' or x is None:
        pass
    else:
        total_events += x

non_val = 0
CR = 0

for x in accuracy:
    if x is None:
        non_val += 1
    elif x == 'new_file':
        pass
    elif x == 1 or x == 0:
        CR += x

predictions = stats['predictions']
labels = stats['labels']

type1 = 0
type2 = 0
total_CR_labels = 0

for i, label in enumerate(labels):
    if label == 'new_file':
        continue
    pred = predictions[i]
    if label == 0 and pred == 1:
        type1 += 1
    elif label == 1:
        total_CR_labels += 1
        if pred == 0:
            type2 += 1

acc_percent = (CR/total_events) * 100

type1_rate = (type1/total_events) * 100
type2_rate = (type2/total_events) * 100
precision = (CR/total_CR_labels) * 100

print(f"""
Accuracy: {acc_percent}, Precision: {precision}, T1_rate: {type1_rate}, T2_rate: {type2_rate}

""")
