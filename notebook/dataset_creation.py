import ast, glob, os, tonic
import pandas as pd
from tqdm import tqdm
import numpy as np

def csv_load(path, name):
    '''load events in .csv files with the same format as in tonic. 
    '''
    df = pd.read_csv(os.path.join(path, name))
    events = None
    index = 0
    print(f'loading -> {name} ...')
    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        text = row['.events']
        if not pd.isna(text) and text[-1]==']':
            text = text.replace('[', '[{').replace(']', '}]').replace('\n', ', ').replace(' ts: ,', '').replace(", x", "}, {x").replace("x", "'x'").replace(" y", " 'y'").replace(" secs", " 'secs'").replace(" nsecs", " 'nsecs'").replace(" polarity", " 'polarity'")
            mydict = ast.literal_eval(text)
            x = pd.DataFrame.from_dict(mydict)['x'].values
            y = pd.DataFrame.from_dict(mydict)['y'].values
            # timestamps are converted in microseconds
            t = pd.DataFrame.from_dict(mydict)['secs'].values*1e6+pd.DataFrame.from_dict(mydict)['nsecs'].values*1e-3
            p = pd.DataFrame.from_dict(mydict)['polarity'].values
            if events is not None:
                events = np.vstack((events, np.array([x,y,t,p]).T))
            else:
                events = np.array([x,y,t,p]).T
        else: print(f'file corrupted at row number {index}')
    return events

def save_as_patches(events, path, label, name_num, patch_size = None, sensor_size=None, max_duration=None, min_num_events=1000, train_test_ratio=.75, ordering = 'xytp'):
    '''split a stream of events as input ('events') into patches defined spatially by 'patch_size' and temporally by 'max_duration'. 'events' is also splitted into training and testing samples with a ratio defined by 'train_test_ratio' and then stored as .npy files. 
    'path', 'label' and 'name_num' allow to store properly the splitted samples. 
    'ordering' gives the indices of the event stream vectors.
    '''
    print('splitting ...')
    if not os.path.exists(path+f'/patch_{patch_size}_duration_{max_duration}/train/{label}'):
        os.makedirs(path+f'/patch_{patch_size}_duration_{max_duration}/train/{label}')
    if not os.path.exists(path+f'/patch_{patch_size}_duration_{max_duration}/test/{label}'):
        os.makedirs(path+f'/patch_{patch_size}_duration_{max_duration}/test/{label}')
    x_index, y_index, t_index, p_index = ordering.find('x'), ordering.find('y'), ordering.find('t'), ordering.find('p')
    events[:,t_index] -= np.min(events[:,t_index]) #makes time alignment
    if sensor_size:
        width, height = sensor_size[0], sensor_size[1]
    else:
        width, height = int(max(events[:,x_index])), int(max(events[:,y_index]))
    if patch_size:
        patch_width, patch_height = patch_size
    else:
        patch_width, patch_height = sensor_size
    if max_duration:
        time_limit = max_duration*1e3 #to enter max_duration in ms
    else:
        time_limit = events[-1,t_index]
    num_patches = int(width//patch_width*height//patch_height*events[-1, t_index]//time_limit)
    pbar = tqdm(total=num_patches)
    # divide the pixel grid into patches
    indice = 0
    set_name = f'/patch_{patch_size}_duration_{max_duration}/train/{label}/'
    indice_test = int(train_test_ratio*num_patches)
    for x in range(width//patch_width):
        for y in range(height//patch_height):
            events_patch = events[
                           (events[:,x_index]>=x*patch_width)&(events[:,x_index]<(x+1)*patch_width)&
                           (events[:,y_index]>=y*patch_height)&(events[:,y_index]<(y+1)*patch_height)]
            events_patch[:,x_index] -= x*patch_width
            events_patch[:,y_index] -= y*patch_height
            for t in range(int(events[-1, t_index]//time_limit-1)):
                events_patch_timesplit = events_patch[(events_patch[:,t_index]>=t*time_limit)&(events_patch[:,t_index]<(t+1)*time_limit)]
                events_patch_timesplit -= events_patch_timesplit[-1,t_index]
                indice+=1
                if indice>indice_test:
                    set_name=f'/patch_{patch_size}_duration_{max_duration}/test/{label}/'
                if events_patch_timesplit.shape[0]>min_num_events:
                    np.save(path+set_name+f'{patch_size}_{max_duration}_{name_num}_{indice}', events_patch_timesplit)
                pbar.update(1)
    pbar.close()
                    
def build_aprovis_dataset(path, labelz, patch_size = None, sensor_size=None, max_duration=None, min_num_events=1000, train_test_ratio=.75, ordering = 'xytp'):
    '''list all files in 'path', load the events with 'csv_load' and split the events into different patches to store the patches as a dataset with 'save_as_patches'. Labels of the dataset have to be given in .csv files names and are then selected according to 'labelz'. 
    '''
    print('Building dataset')
    if not os.path.exists(path+f'patch_{patch_size}_duration_{max_duration}'):
        os.chdir(path)
        for lab_num, label in enumerate(labelz):
            list_csv = glob.glob(f'*{label}*.csv')
            for name_num, name in enumerate(list_csv):
                events = csv_load(path, name)
                save_as_patches(events, path, label, name_num, patch_size = patch_size, sensor_size=sensor_size, max_duration=max_duration, min_num_events=min_num_events, train_test_ratio=train_test_ratio, ordering = ordering)
    else: print(f'this dataset was already created, check at : \n {path}')
            
class Synthetic_Dataset(tonic.dataset.Dataset):
    '''creates a dataset from .csv files in a specific 'path'. Adapted for Synthetic dataset given by Sotiris
    '''
    classes = ["sea", "gro"]
    int_classes = dict(zip(classes, range(2)))
    sensor_size = [128, 128, 2]
    dtype = np.dtype([("x", int), ("y", int), ("t", int), ("p", int)])
    ordering = dtype.names

    def __init__(self, save_to, train=True, patch_size=None, max_duration=None, transform=tonic.transforms.NumpyAsType(int), target_transform=None):
        super(Synthetic_Dataset, self).__init__(
            save_to, transform=transform, target_transform=target_transform
        )

        if train:
            self.folder_name = f'patch_{patch_size}_duration_{max_duration}/train/'
        else:
            self.folder_name = f'patch_{patch_size}_duration_{max_duration}/test/'
            
        self.location_on_system = save_to

        file_path = os.path.join(self.location_on_system, self.folder_name)

        if not os.path.exists(file_path):
            build_aprovis_dataset(self.location_on_system, self.classes, patch_size=patch_size, sensor_size=self.sensor_size, max_duration=max_duration)
            
        self.sensor_size[0], self.sensor_size[1] = patch_size[0], patch_size[1]
        
        for path, dirs, files in os.walk(file_path):
            files.sort()
            for file in files:
                if file.endswith("npy"):
                    self.data.append(np.load(os.path.join(path, file)))
                    self.targets.append(self.int_classes[path[-3:]])

    def __getitem__(self, index):
        """
        Returns:
            a tuple of (events, target) where target is the index of the target class.
        """
        events, target = self.data[index], self.targets[index]
        #this line is used in tonic package, keep and see if needed:
        events = np.lib.recfunctions.unstructured_to_structured(events, self.dtype)

        if self.transform is not None:
            events = self.transform(events)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return events, target

    def __len__(self):
        return len(self.data)

    def _check_exists(self):
        return self._is_file_present() and self._folder_contains_at_least_n_files_of_type(
            20, ".npy"
        )
    
def get_isi(events, ordering = 'xytp', verbose = False):
    x_index, y_index, t_index, p_index = ordering.find('x'), ordering.find('y'), ordering.find('t'), ordering.find('p')
    mean_isi = None
    isipol = np.zeros([2])
    for polarity in [0,1]:
        events_pol = events[(events[:, p_index]==polarity)]
        N_events = events_pol.shape[0]-1
        for i in range(events_pol.shape[0]-1):
            isi = events_pol[i+1,t_index]-events_pol[i,t_index]
            if isi>0:
                mean_isi = (N_events-1)/N_events*mean_isi+1/N_events*isi if mean_isi else isi
        isipol[polarity]=mean_isi
    if verbose:
        print(f'Mean ISI for ON events: {np.round(isipol[1].mean()*1e-3,1)} in ms \n')
        print(f'Mean ISI for OFF events: {np.round(isipol[0].mean()*1e-3,1)} in ms \n')
    return isipol

def get_info(path,
            patch_size = (16,16),
            max_duration = None,
            labelz = ['sea','gro'],
            sensor_size = (128, 128, 2),
            min_num_events = 1000,
            train_test_ratio = .75,
            ordering = 'xytp'):
    x_index, y_index, t_index, p_index = ordering.find('x'), ordering.find('y'), ordering.find('t'), ordering.find('p')
    # get the averaged ISI for the different labels
    nb_ground, nb_sea, isi_neg_sea, isi_neg_ground, isi_pos_sea, isi_pos_ground, dura_sea, dura_ground = [0]*8
    print('Number of samples:\n') 
    for train in [False,True]:
        dataset = Synthetic_Dataset(save_to=path, train=train, patch_size=patch_size, max_duration=max_duration, transform=tonic.transforms.NumpyAsType(int))
        if train: print(f'training set: {len(dataset)}')
        else: print(f'testing set: {len(dataset)}')
        for i in range(len(dataset)):
            events, label = dataset[i]
            isi_neg, isi_pos = get_isi(events)
            if label:
                nb_ground+=1
                isi_neg_ground += isi_neg
                isi_pos_ground += isi_pos
                dura_ground += events[-1,t_index]
            else:
                nb_sea+=1
                isi_neg_sea += isi_neg
                isi_pos_sea += isi_pos
                dura_sea += events[-1,t_index]
    print(40*'-')
    print('Inter-Spike-Interval:\n')
    print(f'Mean ISI for ON events and SEA label: {np.round(isi_pos_sea/nb_sea*1e-3,1)} in ms')
    print(f'Mean ISI for OFF events and SEA label: {np.round(isi_neg_sea/nb_sea*1e-3,1)} in ms \n')
    print(f'Mean ISI for ON events and GROUND label: {np.round(isi_pos_ground/nb_ground*1e-3,1)} in ms')
    print(f'Mean ISI for OFF events and GROUND label: {np.round(isi_neg_ground/nb_ground*1e-3,1)} in ms')
    print(40*'-')
    print('Event stream duration:\n')
    print(f'Mean duration for SEA label: {np.round(dura_sea/nb_sea*1e-6,1)} in s')
    print(f'Mean duration for GROUND label: {np.round(dura_ground/nb_ground*1e-6,1)} in s')