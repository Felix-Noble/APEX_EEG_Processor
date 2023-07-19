from APEX_EEG_Processor import EEG_Processor
import numpy as np, pandas as pd, matplotlib.pyplot as plt
import json
import numpy as np


def t_anova_permutation_test(eeg_data, num_permutations=1000):
    num_conditions = len(eeg_data)
    num_timepoints = eeg_data.iloc[0].shape[0]

    observed_diss = np.zeros((num_conditions, num_timepoints - 1))
    empirical_diss = np.zeros((num_conditions, num_permutations, num_timepoints - 1))

    for cond_idx, condition in enumerate(eeg_data):
        num_subjects = condition.shape[1]
        observed_avg_erp = condition.mean(axis=1)

        for perm_idx in range(num_permutations):
            permuted_data = np.zeros_like(condition)

            for subj_idx in range(num_subjects):
                perm_subjects = np.random.permutation(num_subjects)
                permuted_data[:, subj_idx] = condition[:, perm_subjects[subj_idx]]

            perm_avg_erp = permuted_data.mean(axis=1)

            for timepoint_idx in range(num_timepoints - 1):
                perm_diss_value = AP.global_diss(perm_avg_erp[timepoint_idx], perm_avg_erp[timepoint_idx + 1])
                empirical_diss[cond_idx, perm_idx, timepoint_idx] = perm_diss_value

        for timepoint_idx in range(num_timepoints - 1):
            observed_diss[cond_idx, timepoint_idx] = AP.global_diss(observed_avg_erp[timepoint_idx],
                                                                 observed_avg_erp[timepoint_idx + 1])

    p_values = np.zeros((num_conditions, num_timepoints - 1))

    for cond_idx in range(num_conditions):
        for timepoint_idx in range(num_timepoints - 1):
            observed_diss_value = observed_diss[cond_idx, timepoint_idx]
            p_values[cond_idx, timepoint_idx] = (np.sum(
                empirical_diss[cond_idx, :, timepoint_idx] >= observed_diss_value) + 1) / (num_permutations + 1)

    return p_values

def organise_files(filenames):
    files_dict = {}
    for filename in filenames:
        f = AP.get_main_filename(filename)
        f = f.split('-')
        # The condition name starts from the second part of the filename
        condition = "-".join(f[1:]) # Skip subject ID
        if condition not in files_dict:
            # If this is the first time we've seen this condition, initialize the list
            files_dict[condition] = []
        # Add the full filename to the list of files for this condition if it's not already there
        if filename not in files_dict[condition]:
            files_dict[condition].append(filename)
    return files_dict


read = r"C:\Users\Felix\Dropbox\My PC (LAPTOP-J41MAND4)\Users\Felix\Documents\Philosophy+\Cognitive science\Aptima Coding\data\epochs as csv"
read_fif = r"C:\Users\Felix\Dropbox\My PC (LAPTOP-J41MAND4)\Users\Felix\Documents\Philosophy+\Cognitive science\Aptima Coding\data\cropped_files"
write = r"C:\Users\Felix\Dropbox\My PC (LAPTOP-J41MAND4)\Users\Felix\Documents\Philosophy+\Cognitive science\Aptima Coding\data\write"
AP = EEG_Processor(read, write, mne_log=False)

files = AP.subj_csv_files

files_dict = organise_files(files)

ch_names = []
for x in range(1, 129):
    ch_names.append(f'E{x}')

anal = {}
for condition in files_dict.keys():
    for file in files_dict[condition]:
        subject = AP.get_main_filename(file)
        df = pd.read_csv(file)
        gathered_data = []
        for i, timepoint in enumerate(df.loc[:, ch_names].values):
            try:
                time_point_data = timepoint
                time_point_pls1 = df.loc[:, ch_names].values[i+1]
            except IndexError:
                break
            data_point = AP.global_diss(time_point_data, time_point_pls1)
            gathered_data.append(data_point)
        #print('data', gathered_data, 'data')

        if condition not in anal.keys():
            anal[condition] = [gathered_data]
        else:
            anal[condition].append(gathered_data)
        #print(anal)

with open('analysis.json', 'w') as file:
    json.dump(anal, file)
    
exit()
keys = ['CR-lvl grr14-AVG', 'CR-lvl lss13-AVG', 'CR-trial-first_half-AVG', 'CR-trial-second_half-AVG', 'CRCT-AVG', 'MISS-AVG', 'MS-lvl grr14-AVG', 'MS-lvl lss13-AVG', 'MS-trial-first_half-AVG', 'MS-trial-second_half-AVG']
for key in keys:
    values = []
    print(len(anal[key]), 'length')
    for item in anal[key]:
        #sd = np.std(item)
        values.append(item)
    AP.plot_data(title=key, xlabel="time", ylabel=key, legend=[f'subj{x}' for x in range(1, len(values))], yvar=[val for val in values],
                 error_bars=False, t_range=(-2.5, 4.5))

data = np.array(gfps)
data = data.flatten()
start_time = -2.5
end_time = 4.5

# Create the time vector
time = np.linspace(start_time, end_time, len(data))
std_dev = np.std(data, axis=0)

