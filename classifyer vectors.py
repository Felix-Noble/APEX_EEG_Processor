from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score
from sklearn.inspection import permutation_importance
from sklearn.svm import SVC
import numpy as np
import pandas as pd
import json, os, joblib, time
from APEX_EEG_Processor import EEG_Processor
from datetime import datetime

def MOT_classifyer_sort_files(files):
    countA, countB = 0, 0
    groupA, groupB = [], []
    for file in files:
        if 'CRCT' in file:
            groupA.append(file)
            countA += 1
        elif 'MISS' in file:
            groupB.append(file)
            countB += 1

    return groupA, groupB, countA, countB

def get_vector(df, coords, channels, ch_types=None, i_range=(0, 1000)):
    if ch_types is None and len([x for x in channels if x[0] != 'E']) > 0:
        raise ValueError ('Must pass ch_types list when adding non_eeg channels')
    vec_store = []
    columns = [ch for ch in df.columns if ch in channels]
    for i, ch in enumerate(columns):
        electro_comp = np.array(df.loc[i_range[0]:i_range[1], ch], dtype=np.float64)
        if ch_types[i] == 'eeg' and ch_types is not None:
            spatial_comp = np.array(coords[ch], dtype=np.float64)
            combined_comp = np.concatenate((spatial_comp, electro_comp))
            vec_store.append(combined_comp)
        else:
            vec_store.append(electro_comp)

    output_vector = vec_store[0]
    for vec in vec_store[1:]:
        output_vector = np.concatenate((output_vector, vec))
    """
    This method works fine for offline classification, 
    but need to find a faster way for live decoding
    """
    # print(output_vector.shape)

    return output_vector
def feature_space(channels, ch_types, i_range):
    f_space = []
    for i, ch in enumerate(channels):
        if ch_types[i] == 'eeg':
            f_space += [f'{ch}xyz' * 3]
        for x in range(i_range[0], i_range[1] + 1):
            f_space.append(f'{ch} t_index-{x}')

    return f_space
def make_axis(groupA, groupB, index_range, f_space=False):
    combined_groups = groupA + groupB
    X = []
    y = []
    for file in combined_groups:
        df = pd.read_csv(file)
        vector = get_vector(df, coords, channels, i_range=index_range, f_space=f_space)
        X.append(vector)

    for file in groupA:
        y.append(1)
    for file in groupB:
        y.append(0)

    return X, y

def fit(X_train, y_train, classifier=SVC()):
    pipe = make_pipeline(StandardScaler(), classifier)
    pipe.fit(X_train, y_train)

    return pipe

read = r"E:\Aptima Data complete 07.07.23\epoch flash -1.5 to +5.5"
AP = EEG_Processor(read_dir=read, csv_files='subj')
sfiles = AP.subj_csv_files

xyz = r'C:\Users\hcnla\Documents\Scripts and tools\APEX_EEG_Processor\Classifier\helpers\Electrode xyz.json'
with open(xyz, 'r') as file:
    coords = json.load(file)

channel_select = r"C:\Users\hcnla\Documents\Scripts and tools\APEX_EEG_Processor\Classifier\helpers\channels.json"
with open(channel_select, 'r') as file:
    channels = json.load(file)
channel_names = channels['names']
channel_types = channels['type']

#channels = [f'E{x}' for x in range(1,129)]

index_range = (100, 600)
index_ranges = []
# for x in range(1, 7000):
#     index_ranges.append((x - 1, x))
#
# index_ranges = [(0, 2000)]
# save_axis, save_class, save_eval = True, False, True
save_axis, save_class, save_eval = True, True, True
# print(sfiles.keys())
arr = {}
for subj in sfiles.keys():
    files = sfiles[subj]
    #files = sfiles[subj]
    time = datetime.now()
    groupA, groupB, countA, countB = MOT_classifyer_sort_files(files)
    combined_groups = groupA + groupB
    f_space = feature_space(channel_names, channel_types, index_range)

    X = []
    y = []
    for file in combined_groups:
        df = pd.read_csv(file)
        vector= get_vector(df, coords, channel_names, channel_types, i_range=index_range)
        X.append(vector)


    for file in groupA:
        y.append(1)
    for file in groupB:
        y.append(0)
    write = r''
    nameX = f'{index_range}-all_channels-X'
    namey = f'{index_range}-all_channels-y'
    if save_axis:
        fileX = os.path.join(write, f'{nameX}.npy')
        np.save(fileX, X)
        filey = os.path.join(write, f'{namey}.npy')
        np.save(filey, y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    t_start = datetime.now()
    pipe = fit(X_train, y_train, SVC(class_weight='balanced', kernel='linear'))
    t_estimator = datetime.now() - t_start
    if save_class:
        with open(f'{subj}--Balanced Linear SVM all eeg and bio channels--{index_range}--center eqls FLSH.pkl', 'wb') as file:
            joblib.dump(pipe, file)

    xx = np.array(X)
    n_features_to_select = round(xx.shape[1]/2)
    print(xx.shape)
    estimated_time = (xx.shape[1] - n_features_to_select + 1) * t_estimator
    print(estimated_time)
    score = pipe.score(X_test, y_test)
    prediction = pipe.predict(X_test)
    acc = accuracy_score(prediction, y_test)
    prec = precision_score(prediction, y_test)

    eval = {'index_range': index_range, 'num_channels': len(channel_names),
            'score': score, 'linear_score': None,
            'accuracy': acc, 'precision': prec,
           # 'cv_score': cv_score, 'linear_cv_score': cv_score_lin,
            'estimated_RFE_time': estimated_time,
            'channels': channels, 'feature_space': f_space
            }
    if save_eval:
        with open(f'{subj}--{index_range}.pkl', 'wb') as file:
            joblib.dump(eval, file)

    # print(f"""
    # Eval: i_range={eval['index_range']}, num_channels={len(channel_names)}
    #       score={eval['score']}, lin_score={eval['linear_score']}
    #       cv_score={eval['cv_score']}, lin_cv_score={eval['linear_cv_score']}
    #       permutation duration = {datetime.now() - time}
    # """)

    arr[str(index_range)] = eval
exit()
if len(index_ranges) > 1:
    with open(f'Ms classifier performance.json', 'wb') as file:
        joblib.dump(arr, file)
