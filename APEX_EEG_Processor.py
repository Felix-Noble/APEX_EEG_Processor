import mne, os, json, pickle
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime


class EEG_Processor:
    def __init__(self, read_dir=None, write_dir=None, temporal_res=('Tmu', 1000000),
                 run=False, error_halt=False, load_file_list=False,
                 evt_ext='.evt', paradigm_ext='.PDG', error_ext='.txt',
                 designators=['Subj'], mne_log='WARNING'):
        print('Initialising APEX_EEG_Processor...')
        mne.set_log_level(mne_log)
        self.error_messages = []
        self.raw_files_loaded = set()
        self.error_halt = error_halt
        self.load_file_list = load_file_list
        self.write_dir = write_dir
        self.read_dir = read_dir
        self.designators = designators
        self.evt_ext = evt_ext
        self.paradigm_ext = paradigm_ext
        self.error_ext = error_ext
        self.temporal_res_header = temporal_res[0]
        self.temporal_res_freq = np.float64(temporal_res[1])
        """
                BESA filetype line formats:
                """
        self.BESA_evt_header = "{}          \tCode\tTriNo\tComnt\n".format(self.temporal_res_header)
        self.BESA_evt_line = "{0:<10}\t{1}\t{2}\t{3} {4} {5:<30}\n"  # format of lines in the .evt file.
        self.epochs_format = '{}.0 {}.0 {}.0 {}.0\t{}.0 {}.0 {}.0\t{}.0\t{}.0\n'

        self.supported_filetypes = {
            '.fif': mne.io.read_raw_fif,
            '.edf': mne.io.read_raw_edf,
            '.bdf': mne.io.read_raw_bdf,
            '.mff': mne.io.read_raw_egi,
            '.vhdr': mne.io.read_raw_brainvision
            # Add more file extensions and corresponding read functions as needed
        }

        if self.load_file_list and isinstance(self.load_file_list, list) and len(self.load_file_list) > 0:
            self.files = self.load_file_list
        elif self.load_file_list:
            with open(r'{}\file_list.json'.format(read_dir), 'r') as file:
                self.files = json.load(file)
                print(self.files)
        elif self.read_dir is not None:
            self.files = []
            self.find_subject_files()
        else:
            pass

    def find_subject_files(self, priority='_cropped'):
        try:
            error_files = list(open('errors', 'r'))
        except FileNotFoundError:
            error_files = []
        subject_files = []
        subject_identifications = []
        for filetype in self.supported_filetypes.keys():
            if self.read_dir.endswith(filetype):  # CAN SET READ FOLDER TO SINGLE ITEM
                subject_files = [self.read_dir]
        for item in os.listdir(self.read_dir):
            base_name = os.path.basename(item)
            if base_name in self.raw_files_loaded or f"{base_name}{priority}" in self.raw_files_loaded:
                print(f"File {base_name} or its cropped version has already been loaded")
                continue

            for ext in self.supported_filetypes.keys():
                if item.endswith(ext):
                    subject_files.append(self.read_dir + '\\' + item)
                    subject_identifications.append(item)
                    break
                try:
                    for designator in self.designators:
                        if designator in item:
                            dir = self.read_dir
                            dir = dir + "\\" + item
                            for file in os.listdir(dir):
                                if ext in file and file not in error_files:
                                    subject_files.append(dir + '\\' + file)
                                    subject_identifications.append(file)
                except NotADirectoryError:
                    pass

        subject_files = sorted(subject_files)
        subject_identifications = sorted(subject_identifications)
        self.files = subject_files
        print("File/s found: {}".format(subject_files))

        return subject_files

    def read_raw_file(self, file, preload=False, priority='cropped'):
        # Initialize raw as None
        raw = None
        # Extract the base name of the file
        # base_name = os.path.basename(file)
        #
        # # Check if this file or its cropped version is already loaded
        # if base_name in self.raw_files_loaded or f"{base_name}_{priority}" in self.raw_files_loaded:
        #     print(f"File {base_name} or its cropped version has already been loaded")
        #     return None

        print('Reading raw file from {}'.format(self.get_main_filename(file)))

        # Get the file extension
        ext = file[file.rfind('.'):]

        # Check if the file extension is supported
        try:
            if ext in self.supported_filetypes:
                if ext == '.vhdr':
                    vhdr_path = Path(file)
                    eeg_path = vhdr_path.with_suffix('.eeg')
                    vmrk_path = vhdr_path.with_suffix('.vmrk')
                    if not eeg_path.is_file() or not vmrk_path.is_file():
                        print(
                            f"Missing necessary files for BrainVision format in \033[91m{vhdr_path.parent}\033[0m")
                        return None
                # Call the appropriate read function based on the file extension
                raw = self.supported_filetypes[ext](file, preload=preload)
                # Add the file name to the loaded files set
                self.raw_files_loaded.add(self.get_main_filename(file))

            else:
                print("File format not supported for \033[91m{}\033[0m".format(file))
        except FileNotFoundError:
            print("Missing data in \033[91m{}\033[0m".format(file))
            return None
        return raw

    def read_raw_info(self, file):  # TODO add in a messenger function
        print('Reading info from {}'.format(self.get_main_filename(file)))
        raw_loaded = False
        main_filename = self.get_main_filename(file)
        evt_filename = '{}-EVT.npy'.format(os.path.join(self.read_dir, main_filename))
        info_filename = '{}-info.pckl'.format(os.path.join(self.read_dir, main_filename))
        if os.path.exists(evt_filename):
            events = np.load(evt_filename)
        else:
            raw = self.read_raw_file(file)
            raw_loaded = True
            events = mne.find_events(raw, shortest_event=1)
            np.save(os.path.join(self.write_dir, evt_filename), events)

        if os.path.exists(info_filename):
            with open(info_filename, 'rb') as f:
                info = pickle.load(f)

        else:
            if raw_loaded is not True:
                raw = self.read_raw_file(file)

            info = {'raw.info': {key: value for key, value in raw.info.items()},
                    'n_times': raw.n_times,
                    'event_id': {key: int(value) for key, value in raw.event_id.items()}
                    # 'events': list(mne.find_events(raw))
                    }
            with open(info_filename, 'wb') as f:
                pickle.dump(info, f)

        event_id = info['event_id']
        return events, event_id, info

    def filter(self, data, low, high, power_freq=60):
        # Apply a notch filter to remove line noise
        notch_filtered_raw = data.copy().notch_filter(freqs=[power_freq])

        # Apply a band-pass filter to keep frequencies of interest
        filtered_raw = notch_filtered_raw.copy().filter(l_freq=low, h_freq=high)

        return filtered_raw

    def find_events(self, raw, shortest_event=1, stim_channel='STI 014'):
        events, event_id = None, None
        try:
            if shortest_event is not None:
                events = mne.find_events(raw, shortest_event=shortest_event)
            else:
                events = mne.find_events(raw)
            event_id = raw.event_id
            return events, event_id
        except ValueError:

            try:
                events = mne.find_events(raw, stim_channel=stim_channel)
                event_id = raw.event_id
            except ValueError:
                events, event_id = mne.events_from_annotations(raw)

        return events, event_id

    def read_events_vmrk(self, file):
        main_filename = self.get_main_filename(file)
        evt_file = r'{}\{}{}'.format(self.read_dir, main_filename, '.vmrk')
        with open(evt_file, 'r') as file:
            evt_file = file.readlines()

        events = []
        evt_names = []
        event_id = {}
        for line in evt_file:
            if line.startswith('Mk'):
                event = (line.split(','))
                evt_type = event[0].split("=")[1]
                evt_name = '{}/{}'.format(evt_type, event[1])
                evt_names.append(evt_name)
                events.append(event)
                if evt_name not in event_id.keys():
                    event_id[evt_name] = len(event_id) + 1

        mne_form_events = []
        for i, event in enumerate(events):
            evt_sample = int(event[2])
            evt_name = evt_names[i]
            evt_change = 0
            evt_id = event_id[evt_name]

            mne_form_events.append(np.array([evt_sample, evt_change, evt_id], dtype=np.int64))

        if len(mne_form_events) > 0:
            events = mne_form_events

        return events, event_id

    def crop_events(self, events, event_id, index_event='3.00'):
        """
        :param events: events
        :param event_id: event_id dict
        :param index_event: event name to look for when cropping events
        :return cropped events: the events given to the function excluding the first appearance of the index_event, and all events prior
        """
        if index_event in event_id:
            for line in events:
                if line[2] == event_id[index_event]:
                    sample_start = line[0]
        else:
            sample_start = 0

        for i, event in enumerate(events):
            if event[0] > sample_start:
                cropped_events = events[i:]
                break
            else:
                cropped_events = events

        if len(cropped_events) > 0:
            return cropped_events
        else:
            print("\033[31mNoWARNING: cropped events list empty\033[0m")  # TODO : turn into real error message
            return []

    def cat_events(self, event_id):
        event_id_name = {}
        for key, value in event_id.items():
            if value not in event_id_name:
                event_id_name[value] = key
        return event_id_name

    def group_events(self, event_id, evt=None, lvl=None, trial=None, corr_miss=None):
        """
        Function to group events based on specific criteria: event type, level, trial, and type of result.

        Args:
            event_id (dict): a dictionary where keys represent different event ids (as strings),
            the structure is 'eventname_level_trial_result'.

            evt (str, optional): the event name to match. If none provided, it does not filter by event.

            lvl (int, tuple, list, optional): a specific level or a range of levels to match.
            If it's a tuple or list, it represents a range OR multiple ranges.
            eg., (1, 4) or [1, 4, 7, 10] will match levels 1 through 3, and 7 through 9.

            trial (int, tuple, list, optional): a specific trial or a range of trials to match. Similar to lvl.

            corr_miss (str, optional): two characters to match the type of result. If none provided, it does not filter by result.

        Returns:
            group (list): a list of all events from the input dictionary that meet the criteria.
            If an event does not meet the criteria, it is excluded.
        """
        group = []

        for ID in event_id.keys():
            exclude_lvl, exclude_trial = False, False
            # Splitting the event ID to get event name, level, trial and corr/miss result
            evt_split = ID.split('_')
            e_evt = evt_split[0]
            e_lvl = int(evt_split[1])
            e_trial = int(evt_split[2])
            e_corr_miss = evt_split[3]

            # Exclude event if event name or result do not match
            if evt is not None:
                if e_evt != evt:
                    continue
            if corr_miss is not None:
                if e_corr_miss[0:2] != corr_miss[0:2]:
                    continue

            # Process level data
            if type(lvl) in [tuple, list]:
                # If lvl is a tuple or list, we assume it represents a range or multiple ranges
                exclude_lvl = True
                # Event is set to be excluded due to lvl by default
                pairs = round((len(lvl) / 2) - 0.1)
                for x in range(0, pairs):
                    i = x * 2
                    if lvl[i] <= e_lvl < lvl[i + 1]:
                        # If current level is within range, we don't exclude
                        exclude_lvl = False
            elif type(lvl) == int and e_lvl != lvl:
                # If level does not match the specified one, exclude event
                continue

            # Process trial data
            if type(trial) in [tuple, list]:
                # If trial is a tuple or list, we assume it represents a range or multiple ranges
                exclude_trial = True
                pairs = round((len(trial) / 2) - 0.1)
                for x in range(0, pairs):
                    i = x * 2
                    if trial[i] <= e_trial < trial[i + 1]:
                        # If current trial is within range, we don't exclude
                        exclude_trial = False
            elif type(trial) == int and e_trial != trial:
                # If trial does not match the specified one, exclude event
                continue

            # If event is not excluded by any criteria, add it to the group
            if exclude_trial is not True and exclude_lvl is not True:
                if ID in event_id:
                    group.append(event_id[ID])

        return group

    def epoch(self, data, events, event_groups, file, tmin=-0.2, tmax=1, save=True):
        """
        should recognise either a tuple/list of lists or a dictionary, giving either numerical names or names specified in the dict
        """
        for group in event_groups:
            if len(group[1]) == 0:
                continue
            epoch = mne.Epochs(data, events, group[1], tmin, tmax, proj=True,
                                    picks=('eeg', 'eog'), baseline=(tmin, tmin + 0.1), preload=True)
            csv_path = r"C:\Users\Felix\Dropbox\My PC (LAPTOP-J41MAND4)\Users\Felix\Documents\Philosophy+\Cognitive science\Aptima Coding\data\epochs as csv"
            epoch_df = epoch.to_data_frame()
            filename = f'{self.get_main_filename(file)}-{group[0]}.csv'
            epoch_df.to_csv(f'{csv_path}\{filename}', index=False)
            print("""---
                    {}   
                    Epoch written to \033[93m{}\033[0m in \033[92m{}\033[0m
                                    """.format(datetime.now(), filename, csv_path))
    def get_max_trial(self, subject_event_id):
        max_trial = 0
        subject_event_id = dict(subject_event_id)
        for event in subject_event_id.keys():
            name_split = event.split('_')
            trial = int(name_split[2])
            if trial>max_trial:
                max_trial = trial

        return max_trial
    def get_max_lvl(self, subject_event_id):
        max_lvl = 0
        subject_event_id = dict(subject_event_id)
        for event in subject_event_id.keys():
            name_split = event.split('_')
            lvl = int(name_split[1])
            if lvl > max_lvl:
                max_lvl = lvl

        return max_lvl
    def format_event(self, event, event_id_name, BESA_evt_code=1, event_comment='Trigger', comment_delim='-'):
        # Formats the events given into a single line of an evt file.
        # trigger_num is the trigger identifier BESA will use,
        # it needs to be unique for BESA to recognise it as distinct from other events.
        # event_comment is the description of the event
        # BESA_evt_code signifies what evt type it is to BESA. 1 indicated a trigger

        evt_sample = event[0]  # the sample/time that the event occurred at
        trigger_no = event[2]  # the numeric code for the event that mne reads in
        evt_name = event_id_name[trigger_no]  # the name of the event
        formatted_evt = self.BESA_evt_line.format(evt_sample, BESA_evt_code, trigger_no, event_comment, comment_delim,
                                                  evt_name)

        return formatted_evt

    def get_main_filename(self, path):
        try:
            main_filename = path.rsplit('\\', 1)[-1]
            main_filename = Path(main_filename).stem
        except IndexError:
            main_filename = 'error in code - file origin: {}'.format(dir)

        return main_filename

    def extract_identity(self, filename):
        split = [char for char in filename]
        numbers = []
        for char in split:
            try:
                num = int(char)
                numbers.append(num)
            except ValueError:
                pass
        return numbers

    def write_events(self, formatted_events, main_filename, overwrite=False):
        filename_format = '{}{}'
        filename = filename_format.format(main_filename, self.evt_ext)
        save_dir = r'{}\{}'.format(self.write_dir, filename)

        if save_dir in os.listdir(self.write_dir) and overwrite is not True:
            filename = filename_format.format(main_filename, '-NEW' + self.evt_ext)

            save_dir = r'{}\{}'.format(self.write_dir, filename)

        with open(save_dir, 'w') as file:
            file.write(self.BESA_evt_header)
            for evt in formatted_events:
                file.write(evt)
            file.close()
        # output = "\n".join(formatted_events)
        # with open(save_file, 'w') as file:
        #     file.write(output)
        #     file.close()
        print("""---
        {}   
        Events written to \033[93m{}\033[0m in \033[92m{}\033[0m
                """.format(datetime.now(), filename, save_dir))

    def write(self, output_as_list, filename):
        output = "\n".join(output_as_list)
        save_file = self.write_dir + '\\' + filename
        with open(save_file, 'w') as file:
            file.write(output)
            file.close()
        print("""   \033[91mERRORS written to {} in {}\033[0m
            ---
                        """.format(self.get_main_filename(save_file), self.write_dir))

    def write_paradigm(self, event_id, main_filename='Master paradigm', write_names=True,
                       write_filter_for_artefact=True,
                       epoch_avg=(-100, 500), epoch_baseline=(-100, 0), epoch_artifact=(-100, 500),
                       epoch_stim_artifact=(0, 0), epoch_stim_delay=0):

        filename_format = '{}{}'
        filename = filename_format.format(main_filename, self.paradigm_ext)
        save_dir = r'{}\{}'.format(self.write_dir, filename)
        with open(save_dir, 'w') as file:
            file.write('')
            file.close()
        with open(save_dir, 'a') as file:
            file.write('[Attributes]\n')
            file.write('code\tname\n')
            file.write('\n')
            file.write('[Values]\n')
            for key, value in event_id.items():
                file.write('{}\t{}\n'.format(value, key))
            file.write('\n')
            # conditions = []
            file.write('[Names]\n')
            if write_names:
                names = ''
                for i, id in enumerate(event_id.keys()):
                    names += '{}\t\n'.format(id)
                file.write(names)
                # conditions.append(id)
            file.write('\n')
            file.write('[Epochs]\n')
            for x in event_id.keys():
                file.write(self.epochs_format.format(epoch_avg[0], epoch_avg[1], epoch_baseline[0],
                                                     epoch_baseline[1], epoch_artifact[0], epoch_artifact[1],
                                                     epoch_stim_artifact[0], epoch_stim_artifact[1], epoch_stim_delay))
            file.write('\n')
            file.write('[Thresholds]\n')
            file.write('\n')
            file.write('[Averaging]\n')
            file.write('\n')
            file.write('[Filter]\n')
            if write_filter_for_artefact:
                file.write("0.100000\t0\t1\t1\n")
                file.write("50.000000\t1\t0\t1\n")
                file.write("60.000000\t2.000000\tTRUE\n")
                file.write("80.000000\t5.000000\tFALSE\n)")
                file.write("FALSE\tFALSE\n)")
                file.write("FALSE\tFALSE\tFALSE\n)")
                file.write('TRUE\t\tTRUE\n')
            file.write('\n')
            file.write('[TimeFrequency]\n')
            file.write('\n')
            file.write('[CovarianceEpochs]\n')
            file.write('\n')
            file.write('[Selections]\n')
            for id in event_id.keys():
                file.write('CURRENT.name IS "{}"\n'.format(id))
                file.write('\n')
            file.write('\n')
        print("""---
        {}   
        Paradigm written to \033[93m{}\033[0m in \033[92m{}\033[0m
                        """.format(datetime.now(), filename, save_dir))

    def get_subject_event_id(self, events, event_id):
        subject_event_id = {}
        event_id_name = self.cat_events(event_id)

        for event in events:
            if event_id_name[event[2]] not in subject_event_id.keys():
                subject_event_id[event_id_name[event[2]]] = event[2]

        return subject_event_id

    def BESA_evt_export(self, events, event_id, file_dir, write=True, write_names=True):
        """
        exports evt and PDG files for sorted events to be read by BESA
        :param events: list of events to export
        :param event_id: event_id in mne format
        :param file_dir: full location of eeg file
        :return: writes events to a BESA compatible .evt file and a subsequent .PDG file with coded triggers and conditions
        """
        main_filename = self.get_main_filename(file_dir)
        # evt_filename = '{}{}'.format(main_filename, self.evt_ext)
        # paradigm_filename = '{}{}'.format(main_filename, self.paradigm_ext)
        print(event_id)
        print(events)
        subject_event_id = self.get_subject_event_id(events, event_id)
        event_id_cropped = subject_event_id

        # print(len(event_id), len(event_id_cropped))
        formatted_events = []
        for event in events:
            formatted_events.append(self.format_event(event, event_id_name))
        if write:
            self.write_events(formatted_events, main_filename)
            self.write_paradigm(event_id_cropped, main_filename, write_names=write_names)
            mssg = 'BESA evt and paradigm export complete'
            return mssg, None
        else:
            return formatted_events, event_id

    def error(self, error_mssg):
        self.error_messages.append(error_mssg)
        print(error_mssg)
        if self.error_halt:
            input("\033[91mERROR DETECTED!\033[0m halting...            : ")

    def sort_events_MOT_BESA(self, events, event_id_name, sfreq=1000):
        self.current_sfreq = sfreq
        trial_block_format = '{}-Tr{}>{}{}'
        corr_desig, miss_desig = '+CR', 'MS'
        lvl_lss_format, lvl_great_format = '{}-lvl<={}{}', '{}-lvl>={}{}'
        lvl_cutoff = 13
        trial_block_size = 20
        max_expected_trials = 100
        max_expected_lvl = 40

        trial = 0
        new_events = []
        new_event_id = {}
        index_dict = {"FX": None, "FLSH": None, "MVE0": None, "MVE1": None}
        # index_dict = {"FLSH": None, "MVE0": None}
        evt_id_stagger = 100
        for c, evt_name in enumerate(index_dict.keys()):
            num_of_blocks = round(max_expected_trials / trial_block_size)
            for x in range(1, num_of_blocks + 1):
                name = trial_block_format.format(evt_name, (x - 1) * trial_block_size, (x * trial_block_size),
                                                 corr_desig)
                new_event_id[name] = ((c + 1) * evt_id_stagger + x - 1)

            for y in range(x, x + num_of_blocks):
                name = trial_block_format.format(evt_name, (y - x) * trial_block_size, (y - x + 1) * trial_block_size,
                                                 miss_desig)
                new_event_id[name] = ((c + 1) * evt_id_stagger + y + (50 - x))

        lvl_block_stagger = 700
        for c, evt_name in enumerate(index_dict.keys()):
            evt_id = lvl_block_stagger + c
            new_event_id[lvl_lss_format.format(evt_name, lvl_cutoff, corr_desig)], new_event_id[
                lvl_lss_format.format(evt_name, lvl_cutoff, miss_desig)] = evt_id, evt_id + 50
        for d, evt_name in enumerate(index_dict.keys()):
            evt_id = lvl_block_stagger + c + d + 1
            new_event_id[lvl_great_format.format(evt_name, lvl_cutoff + 1, corr_desig)], new_event_id[
                lvl_great_format.format(evt_name, lvl_cutoff + 1, miss_desig)] = evt_id, evt_id + 50

        for i, event in enumerate(events):
            index_dict.update((k, None) for k in index_dict.keys())
            if event_id_name[event[2]].startswith("FX"):
                trial += 1
                fixation_index = i
                lvl_code = event_id_name[event[2]][2:4]
                lvl = lvl_code.replace('F', '')
                lvl = lvl.replace('X', '')
                if len(lvl) == 1:
                    lvl = '0' + lvl
                for j, check in enumerate(events[fixation_index + 1:]):
                    if event_id_name[check[2]].startswith('CRC'):
                        # miss_data_name = 'CR00'
                        corr_or_miss = corr_desig
                        break
                    elif event_id_name[check[2]].startswith('MS'):
                        # miss_data_name = event_id_name[check[2]]
                        corr_or_miss = miss_desig
                        break

            for key in index_dict.keys():
                if event_id_name[event[2]].startswith(key):
                    index_dict[key] = i

            # found_values = [k for k, v in index_dict.items() if v is not None]
            for evt_name, index in index_dict.items():
                if index is not None:
                    if event_id_name[events[index + 1][2]].startswith('DIN'):
                        block_flt = trial / trial_block_size
                        block = round(block_flt - 0.50001)

                        new_evt_name_trial = trial_block_format.format(evt_name, block * trial_block_size,
                                                                       (block + 1) * trial_block_size, corr_or_miss)
                        sample = np.float64(event[0])
                        orders_of_mag = self.temporal_res_freq / self.current_sfreq
                        new_evt_sample = np.int64(sample * orders_of_mag)
                        new_evt_change_val = events[index + 1][1]
                        new_evt_trial = np.array([new_evt_sample, new_evt_change_val, new_event_id[new_evt_name_trial]],
                                                 dtype=np.int64)
                        new_events.append(new_evt_trial)

                        new_evt_lvl = new_evt_trial.copy()
                        if int(lvl) <= lvl_cutoff:
                            new_evt_name_lvl = lvl_lss_format.format(evt_name, lvl_cutoff, corr_or_miss)
                            new_evt_lvl[2] = new_event_id[new_evt_name_lvl]
                            new_events.append(new_evt_lvl)
                        elif int(lvl) > lvl_cutoff:
                            new_evt_name_lvl = lvl_great_format.format(evt_name, lvl_cutoff + 1, corr_or_miss)
                            new_evt_lvl[2] = new_event_id[new_evt_name_lvl]
                            new_events.append(new_evt_lvl)

        return new_events, new_event_id

    def master_event_id_MOT(self):
        new_event_id = {}  # creates new event_id dict and assigns the 'correct, no misses' event to id:30

        # contains all prefixes to event codes, X=Fixation, F=Flash,
        # G=Move_0 (GO), S=Move_1 (STOP), I=Space_bar (INPUT)
        self.event_letter_codes = ["X", "F", 'G', 'S', "I"]
        event_letter_codes = self.event_letter_codes
        rst_state_evts_old = [('eyec', 'eyeo')]
        rst_state_evts = [('CLS0', 'CLS1'), ('CLS2', 'CLS3'), ('OPN0', 'OPN1'), ('OPN2', 'OPN3')]

        ms_vals = []
        # creates all event names/codes for each possible 'miss' value (NM, N missed out of  M targets) from 1/1 to 9/9
        for x in range(2, 10):
            for y in range(0, x):
                miss_val = 'NM'
                miss_val = miss_val.replace('N', str(y))
                miss_val = miss_val.replace('M', str(x))
                new_name = 'MS' + miss_val
                ms_vals.append(new_name)

        # creates all event names/codes for each prefix in 'event_letter_codes',
        # from lvl 01 to 50, coding each for Correct (C) of Miss (M)
        for i, letter in enumerate(event_letter_codes):
            for x in range(1, 99):
                trial = str(f'0{x}')
                trial = trial[-2:]
                for y in range(1, 99):
                    lvl = str(f'0{y}')
                    lvl = lvl[-2:]
                    evt_name = f'{letter}_{lvl}_{trial}_CR00'
                    new_event_id[evt_name] = len(new_event_id) + 1
                    for ms_val in ms_vals:
                        evt_name = f'{letter}_{lvl}_{trial}_{ms_val}'
                        new_event_id[evt_name] = len(new_event_id) + 1

        # for x in range(0, 10):
        #     new_event_id["ERR" + str(x)] = len(new_event_id) + 1

        return new_event_id

    def sort_events_MOT(self, events, event_id_name, sfreq=1000):
        """
        every 20 trials, separate bin
        lvl 13 below lvl 14 and above
        :info: Extracts information from existing events, creates new events list and event id's for BESA to read

        event name = 'ABCD' (4 letter code outputted by MOT task and read by BESA)
        event code = integer value, read by besa as 'trigger_no' in .evt file and by mne as the event_id

        :param events: '[[sample_1, change_value, event_id] ... [sample_n, change_value, event_id]]' numpy array
        containing each event, as read by mne
        :param events_id_code: reversed event_id dictionary returned by mne,
        allows for specification of events by their names, instead of their id's
        :return new_events: numpy array copying format of mne events. No events are copied
        :return new_event_id: dictionary of event_id's for all expected values of the new events.
        WARNING: not all events will be in this dict, create specific id dict for each event set for use in mne.
        """
        self.din_times = []
        self.current_sfreq = sfreq
        corr_miss = False
        CRMS_errors, DIN_errors = 0, 0
        trial_count = 0
        new_events = []
        self.master_event_id = self.master_event_id_MOT()
        event_letter_codes = self.event_letter_codes

        new_event_id = self.master_event_id
        new_event_id_name = self.cat_events(new_event_id)

        # scrubs events list for information, collecting the location of fixation, flash, move1, move0,
        # and space_bar, writes new events that encode these events at their DIN1 signal (actual screen refresh time)
        # alongside lvl, corr/miss and if missed, n out of m targets

        index_dict = {"FX": None, "FLSH": None, "MVE0": None, "MVE1": None, 'SPCE': None}

        for i, event in enumerate(events):
            index_dict.update((k, None) for k in index_dict.keys())
            evt_id = event[2]
            evt_name = event_id_name[evt_id]
            for key in index_dict.keys():
                if evt_name.startswith(key):
                    index_dict[key] = i

            fx_index = index_dict['FX']
            if fx_index is not None:
                corr_miss = False
                trial_count += 1
                lvl = evt_name.replace('F', '')
                lvl = lvl.replace('X', '')
                lvl = f'0{lvl}'[-2:]
                for j, c_m_evt in enumerate(events[fx_index:fx_index + 100]):
                    curr_samp = c_m_evt[0]
                    fx_samp = events[fx_index][0]
                    t_diff = (curr_samp - fx_samp) / sfreq
                    if t_diff <= 17:
                        evt_id = c_m_evt[2]
                        c_m_name = event_id_name[evt_id]
                        if c_m_name.startswith('CRC'):
                            corr_miss = 'CR00'
                            break
                        elif c_m_name.startswith('MS'):
                            corr_miss = c_m_name
                            break
                    else:
                        break
                if not corr_miss:
                    CRMS_errors += 1

            index = 0
            if corr_miss is not False:
                for key, value in index_dict.items():
                    if value is not None:
                        for j, din_evt in enumerate(events[value:value + 100]):
                            curr_samp = din_evt[0]
                            evt_samp = events[value][0]
                            t_diff = (curr_samp - evt_samp) / sfreq
                            if t_diff <= 0.3:
                                evt_id = din_evt[2]
                                evt_name = event_id_name[evt_id]
                            else:
                                DIN_errors += 1
                                break
                            if evt_name.startswith('DIN'):
                                self.din_times.append(t_diff)
                                din_sample = np.float64(din_evt[0])
                                din_change = din_evt[1]

                                trial = f'0{trial_count}'
                                trial = trial[-2:]
                                new_evt_name = f'{event_letter_codes[index]}_{lvl}_{trial}_{corr_miss}'
                                new_evt_id = new_event_id[new_evt_name]
                                orders_of_mag = 1000 / self.current_sfreq
                                new_evt_sample = np.int64(din_sample * orders_of_mag)
                                new_evt = np.array([new_evt_sample, din_change, new_evt_id], dtype=np.int64)
                                new_events.append(new_evt)
                                break

                    index += 1

        self.CRMS_errors, self.DIN_errors = CRMS_errors, DIN_errors
        try:
            new_events = np.vstack(new_events)
        except ValueError:
            mssg = 'WARNING: ValueError caught, need at least one array to concatenate'
            self.error(mssg)
        return new_events, new_event_id

    def sort_events_Sleep(self, events, events_id_name, file_dir, target_form='Stimulus/S  {}', current_sfreq=10000):
        default_return = [], {}
        self.current_sfreq = current_sfreq
        main_filename = self.get_main_filename(file_dir)
        identi1 = self.extract_identity(main_filename)
        # raw = self.read_raw_file(file_dir)
        # if raw is not None:
        #     self.current_sfreq = raw.info['sfreq']
        # else:
        #     return default_return

        new_event_id = {'CRCT': 1, "MISS": 0, 'None': 3}
        # new_event_id_name = self.cat_events(new_event_id)

        if identi1[-1] == 3 and identi1[-2] == 1:
            block = '1-3'
        else:
            block = identi1[-1]

        target_evts = []
        target_nums = [[1], [2, 7], [3], [4], [5], [6]]

        if isinstance(block, str):
            for x in range(0, 3):
                for num in target_nums[x]:
                    target_evts.append(target_form.format(num))
        else:
            for num in target_nums[block - 1]:
                target_evts.append(target_form.format(num))

        events_filtered = [event for event in events if events_id_name[event[2]] in target_evts]

        csv_file_list = [file for file in os.listdir(self.read_dir) if file.endswith('.csv')]
        for csv_file in csv_file_list:
            identi2 = self.extract_identity(csv_file)
            if len(identi2) < 1:
                continue
            if identi1 == identi2:
                path = os.path.join(self.read_dir, csv_file)
                df = pd.read_csv(path, delimiter=',', encoding='utf-8')
                accuracy = df['accuracy']
                break
            elif block == '1-3' and identi2[-1] == 3 and identi2[-2] == 1:
                accuracy = []
                csv_format = 'subj{}_block{}_Scored.csv'
                subj_id = ''
                for integer in identi1[0:3]:  # only works for subj id's that are 3 digits
                    subj_id += str(integer)

                block1_file = csv_format.format(subj_id, 1)
                block3_file = csv_format.format(subj_id, 3)
                path1 = os.path.join(self.read_dir, block1_file)
                path3 = os.path.join(self.read_dir, block3_file)
                try:
                    block1_acc = pd.read_csv(path1)['accuracy']
                except FileNotFoundError:
                    error_mssg = 'No csv file for subj{} block1 in {}'.format(subj_id, main_filename)
                    self.error(error_mssg)
                    return [], {}
                try:
                    block3_acc = pd.read_csv(path3)['accuracy']
                except FileNotFoundError:
                    error_mssg = 'No csv file for subj{} block3 in {}'.format(subj_id, main_filename)
                    self.error(error_mssg)
                    return [], {}

                len1 = len(block1_acc)
                len3 = len(block3_acc)
                len2 = len(events_filtered) - len1 - len3

                for score in block1_acc:
                    accuracy.append(score)

                for x in range(0, len2):
                    accuracy.append(new_event_id['None'])

                for score in block3_acc:
                    accuracy.append(score)
                break
            else:
                accuracy = None

        if accuracy is not None:
            new_events = []

            for i, event in enumerate(events_filtered):
                sample = np.float64(event[0])
                orders_of_mag = self.temporal_res_freq / self.current_sfreq
                new_evt_sample = np.int64(sample * orders_of_mag)
                new_evt_changevalue = event[1]
                if np.isnan(accuracy[i]):
                    new_evt_id = 3
                    mssg = 'data on line {} in {} = {}, incompatible dtype'.format(i, csv_file, accuracy[i])
                    self.error(mssg)
                else:
                    new_evt_id = accuracy[i]

                new_event = [new_evt_sample, new_evt_changevalue, new_evt_id]

                new_events.append(new_event)
        else:
            mssg = 'No Scored_csv file found for {}'.format(main_filename)
            self.error(mssg)
            return [], {}

        return new_events, new_event_id

    def find_start_index_MOT(self, events_raw, rev_dict):
        previous_events = []
        level_three_found = False
        din_max_found = False
        din_max_present = False

        for index, event in enumerate(events_raw):
            event_code = event[2]
            event_name = rev_dict[(event_code)]
            if event_name == '3.00':
                din_max_present = True
                break

        for index, event in enumerate(events_raw):
            # store the netstation id and name of the event
            event_code = event[2]
            event_name = rev_dict[(event_code)]

            if event_name == '3.00':
                din_max_found = True

            if din_max_found == False and din_max_present == True:
                continue
            # update the counter to find the start of the real trials
            j = 0

            # once we find the first instance of level 3, we know the real trials have started
            # and we will reverse iterate through the list of events using j until we find
            # the moment when the subject moved from the practice trials to the real trials.
            # This is denoted by either 'BEGN' or 'STRT'
            if event_name == 'FX3X':
                level_three_found = True
                cur_tag = 'FX3X'
                while cur_tag != 'BEGN' and cur_tag != 'STRT':
                    # current tag in the reverse iterative process
                    cur_tag = rev_dict[(events_raw[index - j][2])]
                    previous_events.append([index - j, cur_tag])
                    j += 1
            if level_three_found == True:
                break

        # extract the index in the the events_raw file of the BEGN/STRT tag
        # and return this as the index we begin with
        start_index = previous_events[-1][0]
        return start_index
