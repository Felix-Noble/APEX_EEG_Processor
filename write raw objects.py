import mne
import numpy as np
import mffpy
import pandas as pd
from mffpy import Reader
from APEX_EEG_Processor import EEG_Processor # Make sure your custom module is accessible
import json, pickle
save = True
# Path to your .mff file
mff_file_path = r"C:\Users\Felix\Dropbox\My PC (LAPTOP-J41MAND4)\Users\Felix\Documents\Philosophy+\Cognitive science\Aptima Coding\data\files"

AP = EEG_Processor(read_dir=mff_file_path)
# save_file_path = r"C:\Users\Felix\Dropbox\My PC (LAPTOP-J41MAND4)\Users\Felix\Documents\Philosophy+\Cognitive science\Aptima Coding\data\custom evt fif"
# with open(f'{save_file_path}\\master_event_id.json', 'w') as f:
#     json.dump(AP.master_event_id_MOT(), f)

# Load the .mff file with mffpy
for file in AP.files:
    mff_reader: Reader = Reader(filename=file)

    # Get the raw data
    raw = AP.read_raw_file(file)

    # Get the original events
    original_events = mne.find_events(raw, shortest_event=1)
    event_id = raw.event_id

    # Use your custom function to get the new events and event_id
    new_events, new_event_id = AP.sort_events_MOT(original_events, event_id)

    #  ‘ecg’, ‘bio’, ‘stim’, ‘eog’, ‘misc’, ‘seeg’, ‘dbs’, ‘ecog’, ‘mag’, ‘eeg’, ‘ref_meg’, ‘grad’, ‘emg’, ‘hbr’ ‘eyetrack’ or ‘hbo’

    types = {'ECG': 'ecg',
             'BIO': 'bio',
             'STIM': 'stim',
             'EOG': 'eog',
             'EEG': 'eeg',
             'MISC': 'misc'}

    ch_types = [str(ch['kind']) for ch in raw.info['chs']]

    new_raw_ch_types = []
    for t in ch_types:
        for key, item in types.items():
            if key in t:
                new_raw_ch_types.append(item)
                break

    channel_names = list(raw.ch_names)
    channel_types = new_raw_ch_types
    info = mne.create_info(channel_names, raw.info['sfreq'], channel_types)

    # Create the new Raw object
    new_raw = mne.io.RawArray(raw.get_data(), info)
    #new_raw = new_raw.add_channels([])
    new_raw.add_events(new_events, replace=True)

    # Now you can use new_raw in your analyses
    if save:
        # Specify the file path
        save_file_path = r"C:\Users\Felix\Dropbox\My PC (LAPTOP-J41MAND4)\Users\Felix\Documents\Philosophy+\Cognitive science\Aptima Coding\data\custom evt fif"
        mff_name = AP.get_main_filename(file)
        full_save_dir_raw = f'{save_file_path}\\{mff_name}.fif'
        # Save the Raw object
        new_raw.save(full_save_dir_raw, overwrite=True)
