from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import numpy as np
import pandas as pd
import json, os, joblib
from APEX_EEG_Processor import EEG_Processor
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

def get_vector(df, coords, channels, i_range=(0, 1000)):
    vec_store = []
    columns = [ch for ch in df.columns if ch in channels]
    for ch in columns:
        electro_comp = np.array(df.loc[i_range[0]:i_range[1], ch], dtype=np.float64)
        spatial_comp = np.array(coords[ch], dtype=np.float64)
        combined_comp = np.concatenate((spatial_comp, electro_comp))
        vec_store.append(combined_comp)

    output_vector = vec_store[0]
    for vec in vec_store[1:]:
        output_vector = np.concatenate((output_vector, vec))
    """
    This method works fine for offline classification, 
    but need to find a faster way for live decoding
    """
    #print(output_vector.shape)
    return output_vector

read = r"C:\Users\Felix\Dropbox\My PC (LAPTOP-J41MAND4)\Users\Felix\Documents\Philosophy+\Cognitive science\Aptima Coding\data\evoked csv ful"
AP = EEG_Processor(read_dir=read, csv_files='list'  )
files = AP.csv_files

xyz = r'Electrode xyz.json'
with open(xyz, 'r') as file:
    coords = json.load(file)

channel_select = r"C:\Users\Felix\Dropbox\My PC (LAPTOP-J41MAND4)\Users\Felix\Documents\Philosophy+\Cognitive science\Aptima Coding\Classifyers\helpers"
# with open(channel_select, 'r') as file:
#     channels = json.load(file)
channels = [f'E{x}' for x in range(1,129)]

groupA, groupB, countA, countB = MOT_classifyer_sort_files(files)
if countA != countB:
    raise ValueError (f'Number of files found for groupA ({countA}) is not the same as for groupB ({countB})')

combined_groups = groupA+groupB
X = []
y = []
index_range = (0, 3000)
for file in combined_groups:
    df = pd.read_csv(file)
    vector = get_vector(df, coords, channels, i_range=index_range)
    X.append(vector)

for file in groupA:
    y.append(1)
for file in groupB:
    y.append(0)
write = r''
fileX = os.path.join(write, 'MOT_X -1.5 - 1.5 ms.npy')
np.save(fileX, X)
filey = os.path.join(write, 'MOT_y -1.5 - 1.5 ms.npy')
np.save(filey, y)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
pipe = make_pipeline(StandardScaler(), SVC())
pipe.fit(X_train, y_train)
score = pipe.score(X_test, y_test)
with open('SVM all channels -2.5 to 0.5 from G.pkl', 'wb') as file:
    joblib.dump(pipe, file)

print(f"""
Eval: i_range={index_range}, num_channels={len(channels)}
      score={score}, channels={channels}    
""")


