import glob, ast, os
import pandas as pd
from tqdm import tqdm
import numpy as np

index = 'xytp'
x_index, y_index, t_index, p_index = 0, 1, 2, 3
labelz = ['sea', 'ground', 'mixed']

#    #get label name -> for later
#for i, lab in enumerate(labelz):
#    if name.find(lab) != -1: break
#label = i
#print(name, labelz[i])


def csv2npy(path):
    os.chdir(path)
    list_csv = glob.glob('*.csv')
    list_npy = glob.glob('*.npy')
    print(f'list of all .csv files : \n {list_csv} \n')

    for number, name in enumerate(list_csv):
        if name[:-4]+'.npy' not in list_npy:
            print(f'loading: {name}')
            df = pd.read_csv(path+name)
            events = None
            index = 0
            for index, row in tqdm(df.iterrows(), total=df.shape[0]):
                text = row['.events']
                text = text.replace('[', '[{').replace(']', '}]').replace('\n', ', ').replace(' ts: ,', '').replace(", x", "}, {x").replace("x", "'x'").replace(" y", " 'y'").replace(" secs", " 'secs'").replace(" nsecs", " 'nsecs'").replace(" polarity", " 'polarity'")
                mydict = ast.literal_eval(text)
                x = pd.DataFrame.from_dict(mydict)['x'].values
                y = pd.DataFrame.from_dict(mydict)['y'].values
                t = pd.DataFrame.from_dict(mydict)['secs'].values*1e6+pd.DataFrame.from_dict(mydict)['nsecs'].values*1e-3
                p = pd.DataFrame.from_dict(mydict)['polarity'].values
                if events is not None:
                    events = np.vstack((events, np.array([x,y,t,p]).T))
                else:
                    events = np.array([x,y,t,p]).T
            np.save(path+name[:-4], events)
        else: print(f'{name} was already loaded saved as .npy file')

def split2patches(path, patch_size, min_events=100):
    os.chdir(path)
    list_npy = glob.glob('*.npy')
    print(f'list of all files : \n {list_npy} \n')
    for name in list_npy:
        if name.find('patches') == -1 and name[:-4]+f'_patches_{patch_size}.npy' not in list_npy:
            events = np.load(path+name)
            events_stacked = np.array([])
            width, height = int(max(events[:,x_index])), int(max(events[:,y_index]))
            pbar = tqdm(total=width//patch_size*height//patch_size)
            # divide the pixel grid into patches
            for x in range(width//patch_size):
                for y in range(height//patch_size):
                    events_patch = events[
                                   (events[:,x_index]>=x*patch_size)&(events[:,x_index]<(x+1)*patch_size)&
                                   (events[:,y_index]>=y*patch_size)&(events[:,y_index]<(y+1)*patch_size)]
                    events_patch[:,x_index] -= x*patch_size
                    events_patch[:,y_index] -= y*patch_size
                    if len(events_patch)<min_events:
                        pass
                    else:
                        events_patch[:,t_index] -= np.min(events_patch[:,t_index]) #makes time alignment
                        events_stacked = np.vstack([events_stacked, events_patch]) if events_stacked.size else events_patch
                    pbar.update(1)
            pbar.close()
            np.save(path+name[:-4]+f'_patches_{patch_size}', events_stacked)
        else: print(f'{name} was already divided into patches')
        
def get_labels_indices(path, labelz, patch_size):
    os.chdir(path)
    events_stacked = np.array([])
    indices_stacked = np.array([])
    label_stacked = np.array([])
    for i, lab in enumerate(labelz):
        list_npy = glob.glob(f'*{lab}*patches_{patch_size}.npy')
        for name in list_npy:
            events = np.load(path+name)
            indices = np.argwhere(events[:,t_index]==0)
            label = np.ones([indices.shape[0],1])*i
            events_stacked = np.vstack([events_stacked, events]) if events_stacked.size else events
            indices_stacked = np.vstack([indices_stacked, indices]) if indices_stacked.size else indices
            label_stacked = np.vstack([label_stacked, label]) if label_stacked.size else label
    return events_stacked, indices_stacked, label_stacked
        
        
        
        
        
        
        
        
        
import os
import torch
import tonic
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from HOTS.Tools import LRtorch, classification_numbevents

def load_synthetic_data_patches(path, rec_number, patch_size, time_limit, min_events):
    # this function loads .csv files from the 'path' that is defined. A patch_size can be 
    # given to divide the pixel grid into patches of 'patch_size'*'patch_size'. 'time_limit' 
    # sets a maximum recording time (not ready) and if a sample has less that 'min-events'
    # events it is not kept
    f_name = f'../Data/aprovisynt_{patch_size}_{rec_number}_{time_limit}_{min_events}.pkl'
    list = os.listdir(path)
    
    x_index, y_index, t_index, p_index = 0, 1, 2, 3
    labelz = [0,0,1,2,2,2,1] # 0 for 'sea' and 1 for 'ground' (labels for synthetic data)
    label_name = ['sea', 'ground', 'mixed']
    
    if os.path.isfile(f_name):
        with open(f_name, 'rb') as file:
            events_stacked, label_stacked, indices = pickle.load(file)
            label = np.unique(label_stacked)
            num_label = len(label)
            FR = np.zeros(num_label)
            for i, ind in enumerate(indices[:-1]):
                nbev = len(events_stacked[ind:indices[i+1]])
                duration = (events_stacked[indices[i+1]-1,t_index]-events_stacked[ind,t_index])*1e-6
                fr = nbev/duration
                FR[label_stacked[i]] += fr
                #print(f'label: {label_stacked[i]} \n total number of events: {nbev} \n recording time: {duration} sec \n firing rate: {nbev/duration} Hz \n')
            for l in label:
                FR[l] /= np.sum(label_stacked==l)
                print(f'Mean firing rate for label {label_name[l]}: {FR[l]} Hz')
    else:
        events_stacked = np.array([])
        label_stacked = []
        indices = [0]
        for num in rec_number:
            fname_csv = f'{path}/{list[num]}'
            df = pd.read_csv(fname_csv)
            events = np.zeros([df.shape[0],4]).astype(int)
            events[:,[x_index, y_index, p_index]] = df.values[:,[x_index,y_index,4]]
            t_sec = df.values[:,2]
            t_nsec = df.values[:,3]
            initial_time = t_sec[0]*1e6+t_nsec[0]*1e-3
            events[:,t_index] = t_sec[:]*1e6+t_nsec[:]*1e-3-initial_time
            # converts time into microsecs
            label = labelz[num]
            print(f'file name: {list[num]} \n total number of events: {len(events)} \n recording time: {events[-1,t_index]*1e-6} sec \n firing rate: {len(events)/(events[-1,t_index]*1e-6)} Hz \n')
            width, height = max(events[:,x_index]), max(events[:,y_index])
            pbar = tqdm(total=width//patch_size*height//patch_size)
            # divide the pixel grid into patches
            for x in range(width//patch_size):
                for y in range(height//patch_size):
                    events_patch = events[
                                   (events[:,x_index]>=x*patch_size)&(events[:,x_index]<(x+1)*patch_size)&
                                   (events[:,y_index]>=y*patch_size)&(events[:,y_index]<(y+1)*patch_size)]
                    events_patch[:,x_index] -= x*patch_size
                    events_patch[:,y_index] -= y*patch_size
    #                if time_limit:
    #                    time = 0
    #                    events_patch[:,t_index] = events_patch[(events_patch[:,t_index]<time)]
                    if len(events_patch)<min_events:
                        pass
                    else:
                        events_patch[:,t_index] -= np.min(events_patch[:,t_index])
                        indices.append(indices[-1]+events_patch.shape[0])
                        label_stacked.append(label)
                        events_stacked = np.vstack([events_stacked, events_patch]) if events_stacked.size else events_patch
                    pbar.update(1)
            # print(np.max(events_patch[:,x_index]), np.max(events_patch[:,y_index]), np.max(events_patch[:,t_index]))
            pbar.close()
        with open(f_name, 'wb') as file:
            pickle.dump([events_stacked, label_stacked, indices], file, pickle.HIGHEST_PROTOCOL)
    return events_stacked, label_stacked, indices, 



def fit_MLR(events_train, label_train, indices_train, tau_cla, patch_R):
    num_workers = 0 # ajouter num_workers si besoin!
    learning_rate = 0.005
    beta1, beta2 = 0.9, 0.999
    betas = (beta1, beta2)
    num_epochs = 2 ** 5 + 1
    sample_space = 1
    jitonic = [None, None]
    subset_size = None
    verbose=True
    dataset = 'aprovisynt'
    nb_pola = 2
    N = patch_R*patch_R*nb_pola
    
    nb_digit = len(indices_train)

    transform = tonic.transforms.Compose([tonic.transforms.ToTimesurface(surface_dimensions=False, tau=tau_cla, decay="exp", merge_polarities=False)])
    train_dataset = AERDataset(tensors=(events_train, label_train), indices=indices_train, name = 'aprovisynt', transform=transform)
    loader = DataLoader(train_dataset, shuffle=True)

    torch.set_default_tensor_type("torch.DoubleTensor")
    criterion = torch.nn.BCELoss(reduction="mean")
    amsgrad = True #or False gives similar results
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if verbose:
        print(f'device -> {device} - num workers -> {num_workers}')

    n_classes = len(train_dataset.classes)
    logistic_model = LRtorch(N, n_classes)
    logistic_model = logistic_model.to(device)
    logistic_model.train()
    optimizer = torch.optim.Adam(
        logistic_model.parameters(), lr=learning_rate, betas=betas, amsgrad=amsgrad
    )
    pbar = tqdm(total=int(num_epochs))
    for epoch in range(int(num_epochs)):
        losses = []
        for X, label in loader:
            X, label = X[0].to(device) ,label[0].to(device)
            X = X.reshape(X.shape[0], N)
            outputs = logistic_model(X)

            n_events = X.shape[0]
            labels = label*torch.ones(n_events).type(torch.LongTensor).to(device)
            labels = torch.nn.functional.one_hot(labels, num_classes=n_classes).type(torch.DoubleTensor).to(device)

            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        print(f'loss for epoch number {epoch}: {loss}')
        pbar.update(1)

    pbar.close()
    with open(f'../Records/model/torch_model_{tau_cla}_{patch_R}.pkl', 'wb') as file:
        pickle.dump([logistic_model, losses], file, pickle.HIGHEST_PROTOCOL)
    return logistic_model, losses