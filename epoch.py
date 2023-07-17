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
        events, event_id = AP.read_raw_info(file)
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

        raw = AP.read_raw_file(file, preload=True, filter=True, reference='average')

        channels = raw.ch_names
        eeg_channels = channels[0:128]

        # raw_eeg = raw.pick_channels(eeg_channels, ordered=False)

        max_trial = AP.get_max_trial(subject_event_id)
        half_point = round((max_trial / 2) - 0.1)
        #print(subject_event_id)

        conditions = []
        event_conditions = [
            ('CRCT', 'CR', None, None),
            ('MISS', 'MS', None, None),
            ('CR-lvl lss13', 'CR', (0, 13), None),
            ('CR-lvl grr14', 'CR', (14, 40), None),
            ('MS-lvl lss13', 'MS', (0, 13), None),
            ('MS-lvl grr14', 'MS', (14, 40), None),
            ('CR-trial-first_half', 'CR', None, (0, half_point)),
            ('CR-trial-second_half', 'CR', None, (half_point + 1, max_trial)),
            ('MS-trial-first_half', 'MS', None, (0, half_point)),
            ('MS-trial-second_half', 'MS', None, (half_point + 1, max_trial))
        ]

        for condition_info in event_conditions:
            condition_name = condition_info[0]
            corr_miss = condition_info[1]
            lvl_info = condition_info[2]
            trial_info = condition_info[3]

            conditions.append((condition_name,
                               AP.group_events(subject_event_id, corr_miss=corr_miss, lvl=lvl_info, trial=trial_info,
                                               evt='G')))

        AP.epoch_to_csv(raw, new_events, conditions, file, tmin=-2.5, tmax=4.5)
        print('epochs done')
        exit()

        # corr_avg = corr_epoch.average()
        # miss_avg = miss_epoch.average()
