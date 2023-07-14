import mne, sys, os, pickle
from datetime import datetime
import numpy as np
main_script_path = r"C:\Users\Felix\Dropbox\My PC (LAPTOP-J41MAND4)\Users\Felix\Documents\Philosophy+\Cognitive science\Aptima Coding\APEX_EEG_Processor"
sys.path.append(main_script_path)
from APEX_EEG_Processor import EEG_Processor

read  = r"C:\Users\Felix\Dropbox\My PC (LAPTOP-J41MAND4)\Users\Felix\Documents\Philosophy+\Cognitive science\Aptima Coding\data\files"
write = r"C:\Users\Felix\Dropbox\My PC (LAPTOP-J41MAND4)\Users\Felix\Documents\Philosophy+\Cognitive science\Aptima Coding\data\write"
AP = EEG_Processor(read, write, mne_log=False)
datapoints = ['EVT', 'info']
files = AP.files

for file in files:
    events, event_id, info = AP.read_raw_info(file)

    event_id_name = AP.cat_events(event_id)
    srate = info['raw.info']['sfreq']

    start_index = AP.find_start_index_MOT(events, event_id_name)
    end_sample = events[-1][0]
    crop_start = events[start_index][0]/srate
    crop_end = end_sample/srate
    raw = AP.read_raw_file(file, preload=True)
    raw_cropped = raw.crop(tmin=crop_start, tmax=crop_end)
    cropped_save_dir = r"C:\Users\Felix\Dropbox\My PC (LAPTOP-J41MAND4)\Users\Felix\Documents\Philosophy+\Cognitive science\Aptima Coding\data\cropped_files"
    whole_save_dir = os.path.join(cropped_save_dir, f'{AP.get_main_filename(file)}_cropped-raw.fif')
    raw_cropped.save(f'{whole_save_dir}', overwrite=True)
    print(f'Raw cropped saved to {whole_save_dir}\n')
    print('\n')




