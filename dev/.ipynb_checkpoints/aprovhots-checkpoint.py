import glob, ast, os, tonic, torch
import pandas as pd
from tqdm import tqdm
import numpy as np
from torch.utils.data import Dataset, DataLoader

index = 'xytp'
x_index, y_index, t_index, p_index = 0, 1, 2, 3
label_list = ['sea', 'ground', 'mixed']

#    #get label name -> for later
#for i, lab in enumerate(labelz):
#    if name.find(lab) != -1: break
#label = i
#print(name, labelz[i])

def csv2npy(path):
    os.chdir(path)
    list_csv = glob.glob('*.csv')
    list_npy = glob.glob('*.npy')
    list_npy_nopatches = []
    print(f'list of all .csv files : \n {list_csv} \n')

    for number, name in enumerate(list_csv):
        if name.find('patches') == -1:
            list_npy_nopatches.append(name[:-4]+'.npy')
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
    return list_npy_nopatches

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
        if patch_size:
            list_npy = glob.glob(f'*{lab}*patches_{patch_size}.npy')
        else:
            list_npy = glob.glob(f'*{lab}*.npy')
        print(f'using these files: \n {list_npy}')
        for name in list_npy:
            events = np.load(path+name)
            indices = np.vstack((np.array([events_stacked.shape[0]]),np.argwhere((events[1:,t_index]==0)&(np.diff(events[:,t_index])<0))+events_stacked.shape[0]+1))
            label = np.ones([indices.shape[0],1])*label_list.index(lab)
            events_stacked = np.vstack([events_stacked, events]) if events_stacked.size else events
            indices_stacked = np.vstack([indices_stacked, indices]) if indices_stacked.size else indices
            label_stacked = np.vstack([label_stacked, label]) if label_stacked.size else label
    return events_stacked, indices_stacked.astype(int), label_stacked.astype(int)
        
def get_info(path, list_files):
    D = []
    for name in list_files:
        events = np.load(path+name)
        duration = (events[-1,2]-events[0,2])*1e-6
        nb_events = events.shape[0]
        nb_OFF = (events[:,3]==0).sum()
        nb_ON = (events[:,3]==1).sum()
        density = nb_events/duration
        D.append(density*1e-3)
        print(f'file name: {name}')
        print(f'recording duration: {np.round(duration)} s \n events density: {np.round(density,3)} ev/sec \n number of ON/OFF events: {np.round(nb_ON/nb_OFF,3)}\n')
    return D
        
def get_isi(events):
    mean_isi = None
    isipol = np.zeros([2])
    t_index = 2
    for polarity in [0,1]:
        events_pol = events[(events[:, p_index]==polarity)]
        N_events = events_pol.shape[0]-1
        for i in range(events_pol.shape[0]-1):
            isi = events_pol[i+1,t_index]-events_pol[i,t_index]
            if isi>0:
                mean_isi = (N_events-1)/N_events*mean_isi+1/N_events*isi if mean_isi else isi
        isipol[polarity]=mean_isi
    print(f'Mean ISI for ON events: {np.round(isipol[1].mean()*1e-3,1)} in ms \n')
    print(f'Mean ISI for OFF events: {np.round(isipol[0].mean()*1e-3,1)} in ms \n')
    return isipol

def fit_MLR(events_train, label_train, indices_train, tau_cla, patch_R, date):
    num_workers = 0 # ajouter num_workers si besoin!
    learning_rate = 0.005
    beta1, beta2 = 0.9, 0.999
    betas = (beta1, beta2)
    num_epochs = 1#2 ** 5 + 1
    sample_space = 1
    jitonic = [None, None]
    subset_size = None
    verbose=True
    dataset = 'aprovisynt'
    nb_pola = 2
    N = patch_R*patch_R*nb_pola
    sensor_size = (patch_R, patch_R,nb_pola)
    
    nb_digit = len(indices_train)

    transform = tonic.transforms.Compose([tonic.transforms.ToTimesurface(sensor_size=sensor_size, tau=tau_cla, decay="exp")])
    train_dataset = APROVIS_Dataset(tensors=(events_train, label_train), indices=indices_train, name = 'aprovisynt', transform=transform)
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
    with open(f'../Records/model/{date}_torch_model_{tau_cla}_{patch_R}.pkl', 'wb') as file:
        pickle.dump([logistic_model, losses], file, pickle.HIGHEST_PROTOCOL)
    return logistic_model, losses

class APROVIS_Dataset(Dataset):
    """makes a dataset allowing aer_to_vect() transform from tonic
    """
    #sensor_size = (128, 128,2)
    dtype = np.dtype([("x", int), ("y", int), ("t", int), ("p", int)])
    ordering = dtype.names
    classes = ["sea", "ground"]
    
    def __init__(self, tensors, indices, name, transform=None, nb_pola=2):
        self.X_train, self.y_train = tensors
        self.transform = transform
        self.digind = indices[:,0]
        assert len(self.digind) == len(self.y_train)

    def __getitem__(self, index):
        events = make_structured_array(self.X_train[self.digind[index]:self.digind[index+1], x_index], self.X_train[self.digind[index]:self.digind[index+1], y_index], self.X_train[self.digind[index]:self.digind[index+1], t_index], self.X_train[self.digind[index]:self.digind[index+1], p_index], dtype=self.dtype)
        if self.transform:
            events = self.transform(events)
        target = self.y_train[index]
        return events.astype(float), target

    def __len__(self):
        return len(self.digind)-1
    
def make_structured_array(x, y, t, p, dtype):
    """
    Make a structured array given lists of x, y, t, p
    Args:
        x: List of x values
        y: List of y values
        t: List of times
        p: List of polarities boolean
    Returns:
        xytp: numpy structured array
    """
    return np.fromiter(zip(x, y, t, p), dtype=dtype)

    
class LRtorch(torch.nn.Module):
    #torch.nn.Module -> Base class for all neural network modules
    def __init__(self, N, n_classes, bias=True):
        super(LRtorch, self).__init__()
        self.linear = torch.nn.Linear(N, n_classes, bias=bias)
        self.nl = torch.nn.Softmax(dim=1)

    def forward(self, factors):
        return self.nl(self.linear(factors))