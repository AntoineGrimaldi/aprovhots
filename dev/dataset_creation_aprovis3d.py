import ast, glob, os, struct, tonic
import pandas as pd
from tqdm import tqdm
import numpy as np


def loadaerdat(datafile, debug=1, camera='DVS128'):
    """
    Adapted from https://github.com/SensorsINI/processAEDAT/blob/master/jAER_utils/loadaerdat.py
    load AER data file and parse these properties of AE events:
    - timestamps (in us), 
    - x,y-position [0..127]
    - polarity (0/1)
    @param datafile - path to the file to read
    @param debug - 0 = silent, 1 (default) = print summary, >=2 = print all debug
    @param camera='DVS128' or 'DAVIS240'
    @return (ts, xpos, ypos, pol) 4-tuple of lists containing data of all events;
    """
    # constants
    EVT_DVS = 0  # DVS event type
    aeLen = 8  # 1 AE event takes 8 bytes
    readMode = '>II'  # struct.unpack(), 2x ulong, 4B+4B
    td = 0.000001  # timestep is 1us   
    if(camera == 'DVS128'):
        xmask = 0x00fe
        xshift = 1
        ymask = 0x7f00
        yshift = 8
        pmask = 0x1
        pshift = 0
    elif(camera == 'DAVIS240'):  # values take from scripts/matlab/getDVS*.m
        xmask = 0x003ff000
        xshift = 12
        ymask = 0x7fc00000
        yshift = 22
        pmask = 0x800
        pshift = 11
        eventtypeshift = 31
    else:
        raise ValueError("Unsupported camera: %s" % (camera))

    aerdatafh = open(datafile, 'rb')
    k = 0  # line number
    p = 0  # pointer, position on bytes
    statinfo = os.stat(datafile)
    length = statinfo.st_size 
    if debug > 0:
        print ("file size", length)
    
    # header
    lt = aerdatafh.readline()
    while lt and lt[0] == "#":
        p += len(lt)
        k += 1
        lt = aerdatafh.readline() 
        if debug >= 2:
            print (str(lt))
        continue
    
    # variables to parse
    timestamps = []
    xaddr = []
    yaddr = []
    pol = []
    
    # read data-part of file
    aerdatafh.seek(p)
    s = aerdatafh.read(aeLen)
    p += aeLen
    
    while p < length:
        addr, ts = struct.unpack(readMode, s)
        # parse event type
        if(camera == 'DAVIS240'):
            eventtype = (addr >> eventtypeshift)
        else:  # DVS128
            eventtype = EVT_DVS
        
        # parse event's data
        if(eventtype == EVT_DVS):  # this is a DVS event
            x_addr = (addr & xmask) >> xshift
            y_addr = (addr & ymask) >> yshift
            a_pol = (addr & pmask) >> pshift


            if debug >= 3: 
                print("ts->", ts)  # ok
                print("x-> ", x_addr)
                print("y-> ", y_addr)
                print("pol->", a_pol)

            timestamps.append(ts)
            xaddr.append(x_addr)
            yaddr.append(y_addr)
            pol.append(a_pol)
                  
        aerdatafh.seek(p)
        s = aerdatafh.read(aeLen)
        p += aeLen        

    if debug > 0:
        try:
            print ("read %i (~ %.2fM) AE events, duration= %.2fs" % (len(timestamps), len(timestamps) / float(10 ** 6), (timestamps[-1] - timestamps[0]) * td))
            n = 5
            print ("showing first %i:" % (n))
            print ("timestamps: %s \nX-addr: %s\nY-addr: %s\npolarity: %s" % (timestamps[0:n], xaddr[0:n], yaddr[0:n], pol[0:n]))
        except:
            print ("failed to print statistics")

    events = np.concatenate((
        [[e] for e in xaddr],
        [[e] for e in yaddr],
        [[e] for e in pol],
        [[e] for e in timestamps]
    ), axis=1).astype('float64')

    return events

def save_as_patches(events, path, label, name_num, patch_size = None, sensor_size=None, max_duration=None, min_num_events=1000, train_test_ratio=.75, ordering = 'xytp'):
    '''split a stream of events as input ('events') into patches defined spatially by 'patch_size' and temporally by 'max_duration'. 'events' is also splitted into training and testing samples with a ratio defined by 'train_test_ratio' and then stored as .npy files. 
    'path', 'label' and 'name_num' allow to store properly the splitted samples. 
    'ordering' gives the indices of the event stream vectors.
    '''
    # print('splitting ...')
    if not os.path.exists(path+f'/patch_{patch_size}_duration_{max_duration}/train/{label}'):
        os.makedirs(path+f'/patch_{patch_size}_duration_{max_duration}/train/{label}')
    if not os.path.exists(path+f'/patch_{patch_size}_duration_{max_duration}/test/{label}'):
        os.makedirs(path+f'/patch_{patch_size}_duration_{max_duration}/test/{label}')
    x_index, y_index, t_index, _ = ordering.find('x'), ordering.find('y'), ordering.find('t'), ordering.find('p')
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
    num_patches = width//patch_width*height//patch_height*int(events[-1, t_index]//time_limit)
    pbar = tqdm(total=num_patches)
    # divide the pixel grid into patches
    indice = 0
    not_saved = 0
    set_name = f'/patch_{patch_size}_duration_{max_duration}/train/{label}/'
    indice_test = int(train_test_ratio*num_patches)
    for x in range(width//patch_width):
        for y in range(height//patch_height):
            events_patch = events[
                           (events[:,x_index]>=x*patch_width)&(events[:,x_index]<(x+1)*patch_width)&
                           (events[:,y_index]>=y*patch_height)&(events[:,y_index]<(y+1)*patch_height)]
            events_patch[:,x_index] -= x*patch_width
            events_patch[:,y_index] -= y*patch_height
            for t in range(int(events[-1, t_index]//time_limit)):
                events_patch_timesplit = events_patch[(events_patch[:,t_index]>=t*time_limit)&(events_patch[:,t_index]<(t+1)*time_limit)]
                indice+=1
                if events_patch_timesplit.shape[0]>min_num_events:
                    # print("a",events_patch_timesplit)
                    # print("HERE",events_patch_timesplit[-1,t_index])
                    # events_patch_timesplit -= events_patch_timesplit[-1,t_index]
                    # print("b",events_patch_timesplit)
                    if indice>indice_test:
                        set_name=f'/patch_{patch_size}_duration_{max_duration}/test/{label}/'
                    # print("c",events_patch_timesplit)
                    np.save(path+set_name+f'{patch_size}_{max_duration}_{name_num}_{indice}', events_patch_timesplit)
                    # print()
                else: 
                    not_saved += 1
                pbar.update(1)
    pbar.close()

def build_aprovis_dataset(path, labelz, data_type, patch_size = None, sensor_size=None, max_duration=None, min_num_events=1000, train_test_ratio=.75, ordering = 'xytp'):
    '''list all files in 'path', load the events and split the events into different patches to store the patches as a dataset with 'save_as_patches'. Labels of the dataset have to be given in .csv files names and are then selected according to 'labelz'. 
    '''
    print('Building dataset - '+data_type+' data')
    if not os.path.exists(path+f'patch_{patch_size}_duration_{max_duration}'):
        os.chdir(path)

        if data_type == 'experimental':
            load_data = lambda x: loadaerdat(datafile=x, debug=0, camera='DVS128')
            extension = '.aedat'
        elif data_type == 'synthetic':
            load_data = lambda x: np.load(x)
            extension = '.npy'

        for _, label in enumerate(labelz):
            print(label)
            list_files = glob.glob(f'./*{label}*/*{extension}')
            for name_num, name in enumerate(list_files):
                events = load_data(path+name)
                save_as_patches(events, path, label, name_num, patch_size = patch_size, sensor_size=sensor_size, max_duration=max_duration, min_num_events=min_num_events, train_test_ratio=train_test_ratio, ordering = ordering)
    else: print(f'this dataset was already created, check at : \n {path}')
            
class aprovis3dDataset(tonic.dataset.Dataset):
    '''creates a dataset from .npy or .aedat files in a specific 'path'. Adapted for synthetic events obtained from RGB frames given by UCA and for experimental events obtained by NTUA
    '''
    classes = ["sea", "gro", "mix"]
    int_classes = dict(zip(classes, range(len(classes))))
    dtype = np.dtype([("x", int), ("y", int), ("t", int), ("p", int)])
    ordering = dtype.names

    def __init__(self, save_to, data_type, train=True, patch_size=None, max_duration=None, min_num_events=1000, transform=tonic.transforms.NumpyAsType(int), target_transform=None, sensor_size=[128,128,2]):
        super(aprovis3dDataset, self).__init__(
            save_to, transform=transform, target_transform=target_transform
        )

        self.sensor_size = sensor_size
        self.data_type = data_type
        assert data_type in ['synthetic', 'experimental']

        if train:
            self.folder_name = f'patch_{patch_size}_duration_{max_duration}/train/'
        else:
            self.folder_name = f'patch_{patch_size}_duration_{max_duration}/test/'
            
        self.location_on_system = save_to

        file_path = os.path.join(self.location_on_system, self.folder_name)

        if not os.path.exists(file_path):
            build_aprovis_dataset(self.location_on_system, self.classes, self.data_type, patch_size=patch_size, sensor_size=self.sensor_size, max_duration=max_duration, min_num_events=min_num_events)
            
        self.sensor_size[0], self.sensor_size[1] = patch_size[0], patch_size[1]
        
        for path, _, files in os.walk(file_path):
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
        dataset = aprovis3dDataset(save_to=path, train=train, patch_size=patch_size, max_duration=max_duration, transform=tonic.transforms.NumpyAsType(int))
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