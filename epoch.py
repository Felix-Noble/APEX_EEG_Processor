"""
Main loop
"""
import mne, pandas as pd
from APEX_EEG_Processor import EEG_Processor
import matplotlib.pyplot as plt

# eeg_channels = list(range(0, 128))
CRMS_errors = []
DIN_errors = []
if __name__ == '__main__':
    print('Starting main loop...')
    read = r"C:\Users\Felix\Dropbox\My PC (LAPTOP-J41MAND4)\Users\Felix\Documents\Philosophy+\Cognitive science\Aptima Coding\data\files"
    read_fif = r"C:\Users\Felix\Dropbox\My PC (LAPTOP-J41MAND4)\Users\Felix\Documents\Philosophy+\Cognitive science\Aptima Coding\data\cropped_files"
    write = r"C:\Users\Felix\Dropbox\My PC (LAPTOP-J41MAND4)\Users\Felix\Documents\Philosophy+\Cognitive science\Aptima Coding\data\write"
    AP = EEG_Processor(read, write, mne_log=False)

    for file in AP.files:
        events, event_id, info = AP.read_raw_info(file)
        event_id_name = AP.cat_events(event_id)
        new_events, new_event_id = AP.sort_events_MOT(events, event_id_name)
        new_event_id_name = AP.cat_events(new_event_id)
        # for i, event in enumerate (events):
        #     print(event_id_name[event[2]])
        # for event in new_events:
        #     print(new_event_id_name[event[2]])

        print(f'DIN ERRORS: {AP.DIN_errors}\n')

        CRMS_errors.append((file, AP.CRMS_errors))
        DIN_errors.append((file, AP.DIN_errors))

        subject_event_id = AP.get_subject_event_id(new_events, new_event_id)
        subject_event_id_name = AP.cat_events(subject_event_id)

        raw = AP.read_raw_file(file, preload=True)

        channels = raw.ch_names
        eeg_channels = channels[0:128]

        # raw_eeg = raw.pick_channels(eeg_channels, ordered=False)
        raw_eeg = mne.add_reference_channels(raw, ref_channels="average")  # sets average reference
        filtered_raw = AP.filter(raw_eeg, 0.1, 50)

        max_trial = AP.get_max_trial(subject_event_id)
        half_point = round((max_trial / 2) - 0.1)
        #print(subject_event_id)

        conditions = []
        # Conditions for correct/miss
        conditions.append(('CRCT', AP.group_events(subject_event_id, corr_miss='CR')))
        conditions.append(('MISS', AP.group_events(subject_event_id, corr_miss='MS')))
        # Conditions above and below lvl 13 correct/miss
        conditions.append(('CR-lvl lss13', AP.group_events(subject_event_id, corr_miss='CR', lvl=(0, 13))))
        conditions.append(('CR-lvl grr14', AP.group_events(subject_event_id, corr_miss='CR', lvl=(14, 40))))
        conditions.append(('MS-lvl lss13', AP.group_events(subject_event_id, corr_miss='MS', lvl=(0, 13))))
        conditions.append(('MS-lvl grr14', AP.group_events(subject_event_id, corr_miss='MS', lvl=(14, 40))))

        conditions.append(('CR-trial-first_half', AP.group_events(subject_event_id, corr_miss='CR', trial=(0, half_point))))
        conditions.append(('CR-trial-second_half', AP.group_events(subject_event_id, corr_miss='CR', trial=(half_point + 1, max_trial))))
        conditions.append(('MS-trial-first_half', AP.group_events(subject_event_id, corr_miss='MS', trial=(0, half_point))))
        conditions.append(('MS-trial-second_half', AP.group_events(subject_event_id, corr_miss='MS', trial=(half_point + 1, max_trial))))


        AP.epoch(filtered_raw, new_events, conditions, file)
        print('epochs done')
        exit()

        # corr_avg = corr_epoch.average()
        # miss_avg = miss_epoch.average()
