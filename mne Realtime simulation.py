import pandas as pd
import numpy as np
from mne_realtime import RtClient, MockRtClient, RtEpochs
import mne, socket, json, joblib, pickle
from datetime import datetime
from APEX_EEG_Processor import EEG_Processor
read = r"C:\Users\Felix\Dropbox\My PC (LAPTOP-J41MAND4)\Users\Felix\Documents\Philosophy+\Cognitive science\Aptima Coding\data\custom evt fif"
AP = EEG_Processor(read_dir=read, mne_log=False)
eeg_files = AP.files
auto_run = True
events_lookup = ['F']
epoch_interval = (-1.5, 1.5)
"""
X - 1.5 ... F - 1.0 ... G - 4.5 ... S - 0.0
"""
mne.set_log_level(False)

if __name__ == '__main__':
    class_loc = r"C:\Users\Felix\Dropbox\My PC (LAPTOP-J41MAND4)\Users\Felix\Documents\Philosophy+\Cognitive science\Aptima Coding\Classifyers\SVM all channels -2.5 to 0.5 from G.pkl"
    with open(class_loc, 'rb') as file:
        classifier = joblib.load(file)
    coords_loc = r"C:\Users\Felix\Dropbox\My PC (LAPTOP-J41MAND4)\Users\Felix\Documents\Philosophy+\Cognitive science\Aptima Coding\Classifyers\helpers\Electrode xyz.json"
    with open(coords_loc, 'r') as file:
        coords = json.load(file)
    channel_select = r"C:\Users\Felix\Dropbox\My PC (LAPTOP-J41MAND4)\Users\Felix\Documents\Philosophy+\Cognitive science\Aptima Coding\Classifyers\helpers\channels.json"
    with open(channel_select, 'r') as file:
        channels = json.load(file)

    # training_set_list_loc = r""
    # with open(training_set_list_loc, 'r') as file:
    #     pass
    master_event_id_loc = r"C:\Users\Felix\Dropbox\My PC (LAPTOP-J41MAND4)\Users\Felix\Documents\Philosophy+\Cognitive science\Aptima Coding\data\custom evt fif"
    with open(f'{master_event_id_loc}\\master_event_id.json') as file:
        master_event_id = json.load(file)

    stats = {'files': [],
             'event_num': [],
             'times': [],
             'predictions': [],
             'labels': [],
             'accuracy': []}

    for file in eeg_files:
        stats['files'].append(file)
        stats['event_num'].append('new_file')
        stats['times'].append('new_file')
        stats['predictions'].append('new_file')
        stats['labels'].append('new_file')
        stats['accuracy'].append('new_file')
        raw = AP.read_raw_file(file)

        events = mne.find_events(raw, shortest_event=1)
        new_events, new_events_id = AP.sort_events_MOT(events, master_event_id)
        subject = AP.get_main_filename(file)
        abs_path = file.replace('.fif', '')
        event_id_name = AP.cat_events(master_event_id)
        epoch_order = []

        for i, event in enumerate(events):
            event_id = event[2]
            event_name = event_id_name[event_id]
            split = event_name.split('_')
            if split[0] in events_lookup:
                event_sample = event[0]
                event_time = event_sample / raw.info['sfreq']
                c_m = split[3]
                label = None
                if 'CR' in c_m:
                    label = 1
                elif 'MS' in c_m:
                    label = 0

                epoch_order.append((event_id, event_time, label))
        stats['event_num'].append(len(epoch_order))

        if not auto_run:
            input(f'''
            {len(epoch_order)} events found in subject file: {subject}
            event_lookup = {events_lookup} \n
            Press any key to start simulation...\n''')

        rt_client = MockRtClient(raw)
        tmin = epoch_interval[0]
        tmax = epoch_interval[1]
        t_end = raw.n_times / raw.info['sfreq']

        picks = ['eeg']
        eeg_channels = [f'E{x}' for x in range(1, 129)] + ['VREF']
        rt_epochs = RtEpochs(rt_client, [x[0] for x in epoch_order], tmin, tmax, baseline=(tmin, tmin+0.1),picks=picks, isi_max=4.0)
        rt_epochs.start()
        print(f'Starting MOT simulation with {subject}: \n')
        print('sending data to client...')
        rt_client.send_data(rt_epochs, picks, tmin=0, tmax=200, buffer_size=100)
        print('data sent. \n')
        # cycle through the events that have been found and sent to the client

        for ev_num, ev in enumerate(rt_epochs.iter_evoked()):

            # take a timestamp at the start of processing
            proc_str = datetime.now()
            label = epoch_order[ev_num][2]

            epoch = ev.data.T
            # turn the epoch into a vector categorising the features
            df = pd.DataFrame(epoch, columns=eeg_channels)
            vec_store = []
            columns = [ch for ch in df.columns if ch in channels]
            for ch in columns:
                electro_comp = np.array(df.loc[:, ch], dtype=np.float64)
                spatial_comp = np.array(coords[ch], dtype=np.float64)
                combined_comp = np.concatenate((spatial_comp, electro_comp))
                vec_store.append(combined_comp)

            feature_vector = vec_store[0]
            for vec in vec_store[1:]:
                feature_vector = np.concatenate((feature_vector, vec))
            # 384512 is the expected vector shape
            #print(feature_vector.shape, 384512)
            # place the input vector into a 2D array for prediction
            input_vector = [feature_vector]
            # get a prediction from the classifier
            try:
                prediction = classifier.predict(input_vector)
            except ValueError:
                prediction = None
                print('failed to make suitible vector!')
            # take a timestamp at the end of processing
            proc_fin = datetime.now()

            # input statistics from run into stats arr
            stats['times'].append((proc_str, proc_fin))
            stats['predictions'].append(prediction)
            stats['labels'].append(label)
            if prediction == label:
                accuracy = 1
            elif prediction is None:
                accuracy = None
            elif prediction != label:
                accuracy = 0
            stats['accuracy'].append(accuracy)
stats_f = r"C:\Users\Felix\Dropbox\My PC (LAPTOP-J41MAND4)\Users\Felix\Documents\Philosophy+\Cognitive science\Aptima Coding\Classifyers\stats"
with open(f'{stats_f}\\{str(datetime.now()).replace(":", ".")}-stats.pkl', 'wb') as file:
    pickle.dump(stats, file)




