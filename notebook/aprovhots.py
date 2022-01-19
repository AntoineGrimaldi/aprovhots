import torch, tonic, pickle, os
from dataset_creation import Synthetic_Dataset
from HOTS.tools import HOTS_Dataset, get_loader
from tqdm import tqdm
import numpy as np


def score_classif_events(likelihood, true_target, thres=None, verbose=True):
    
    max_len = 0
    for likeli in likelihood:
        if max_len<likeli.shape[0]:
            max_len=likeli.shape[0]

    matscor = np.zeros([len(true_target),max_len])
    matscor[:] = np.nan
    sample = 0
    lastac = 0
    nb_test = len(true_target)

    for likelihood_, true_target_ in zip(likelihood, true_target):
        pred_target = np.zeros(len(likelihood_))
        pred_target[:] = np.nan
        if not thres:
            pred_target = np.argmax(likelihood_, axis = 1)
        else:
            for i in range(len(likelihood_)):
                if np.max(likelihood_[i])>thres:
                    pred_target[i] = np.argmax(likelihood_[i])
        for event in range(len(pred_target)):
            if np.isnan(pred_target[event])==False:
                matscor[sample,event] = pred_target[event]==true_target_
        if pred_target[-1]==true_target_:
            lastac+=1
        sample+=1

    meanac = np.nanmean(matscor)
    onlinac = np.nanmean(matscor, axis=0)
    lastac/=nb_test
    truepos = len(np.where(matscor==1)[0])
    falsepos = len(np.where(matscor==0)[0])

    if verbose:
        print(f'Mean accuracy: {np.round(meanac,3)*100}%')
        plt.plot(onlinac);
        plt.xlabel('number of events');
        plt.ylabel('online accuracy');
        plt.title('LR classification results evolution as a function of the number of events');
    
    return meanac, onlinac, lastac, truepos, falsepos

def score_classif_time(likelihood, true_target, timestamps, timestep, thres=None, verbose=True):
    
    max_dur = 0
    for time in timestamps:
        if max_dur<time[-1]:
            max_dur=time[-1]
            
    time_axis = np.arange(0,max_dur,timestep)

    matscor = np.zeros([len(true_target),len(time_axis)])
    matscor[:] = np.nan
    sample = 0
    lastac = 0
    nb_test = len(true_target)
    
    for likelihood_, true_target_, timestamps_ in zip(likelihood, true_target, timestamps):
        pred_timestep = np.zeros(len(time_axis))
        pred_timestep[:] = np.nan
        for step in range(1,len(pred_timestep)):
            indices = np.where((timestamps_.numpy()<=time_axis[step])&(timestamps_.numpy()>time_axis[step-1]))[0]
            mean_likelihood = np.mean(likelihood_[indices,:],axis=0)
            if np.isnan(mean_likelihood).sum()>0:
                if not np.isnan(np.array(pred_timestep[step-1])):
                    pred_timestep[step] = pred_timestep[step-1]
                    #pred_timestep[step] = np.nan
            else:
                if not thres:
                    pred_timestep[step] = np.nanargmax(mean_likelihood)
                elif np.max(likelihood_[indices,np.nanargmax(mean_likelihood)])>thres:
                    pred_timestep[step] = np.nanargmax(mean_likelihood)
                elif not np.isnan(np.array(pred_timestep[step-1])):
                    pred_timestep[step] = pred_timestep[step-1]
                    #pred_timestep[step] = np.nan
            if not np.isnan(pred_timestep[step]):
                matscor[sample,step] = pred_timestep[step]==true_target_
        if pred_timestep[-1]==true_target_:
            lastac+=1
        sample+=1
        
    meanac = np.nanmean(matscor)
    onlinac = np.nanmean(matscor, axis=0)
    lastac/=nb_test
    truepos = len(np.where(matscor==1)[0])
    falsepos = len(np.where(matscor==0)[0])
        
    if verbose:
        print(f'Mean accuracy: {np.round(meanac,3)*100}%')
        plt.plot(time_axis*1e-3,onlinac);
        plt.xlabel('time (in ms)');
        plt.ylabel('online accuracy');
        plt.title('LR classification results evolution as a function of time');
    
    return meanac, onlinac, lastac, truepos, falsepos


## OTHER 
# classif avec histogram

def fit_histo(network, 
              num_workers=0,
              verbose=True):
    
    path_to_dataset = f'../Records/output/train/{network.get_fname()}_None/'
    if not os.path.exists(path_to_dataset):
        print('process samples with the HOTS network first')
        return
    
    timesurface_size = (network.TS[0].camsize[0], network.TS[0].camsize[1], network.L[-1].kernel.shape[1])
    dataset = HOTS_Dataset(path_to_dataset, timesurface_size, transform=tonic.transforms.NumpyAsType(int))
    loader = get_loader(dataset, num_workers = num_workers)
    if verbose: print(f'Number of training samples: {len(loader)}')
    model_name = f'../Records/models/{network.get_fname()}_{len(loader)}_histo.pkl' 

    if os.path.isfile(model_name):
        print('load existing histograms')
        with open(model_name, 'rb') as file:
            histo, labelz = pickle.load(file)
    else:
        p_index = dataset.ordering.index('p')
        #n_classes = len(dataset.classes)
        n_polarity = timesurface_size[2]
        histo = np.zeros([len(loader),n_polarity])
        labelz = []
        pbar = tqdm(total=len(loader))
        sample_number = 0
        for events, label in loader:
            events, label = events.squeeze(0), label.squeeze(0) # just one digit = one batch
            labelz.append(label)
            value, frequency = np.unique(events[:,p_index], return_counts=True)
            histo[sample_number,value] = frequency
            sample_number+=1
            pbar.update(1)
        pbar.close()
        with open(model_name, 'wb') as file:
            pickle.dump([histo, labelz], file, pickle.HIGHEST_PROTOCOL)

    return histo, labelz

def predict_histo(network,
                  histo_train,
                  labelz_train,
                  num_workers=0,
                  measure='knn',
                  k = 6,
                  n_jobs = 16,
                  verbose=True):
    path_to_dataset = f'../Records/output/test/{network.get_fname()}_None/'
    if not os.path.exists(path_to_dataset):
        print('process samples with the HOTS network first')
        return
    timesurface_size = (network.TS[0].camsize[0], network.TS[0].camsize[1], network.L[-1].kernel.shape[1])
    dataset = HOTS_Dataset(path_to_dataset, timesurface_size, transform=tonic.transforms.NumpyAsType(int))
    loader = get_loader(dataset, num_workers = num_workers)
    if verbose: print(f'Number of testing samples: {len(loader)}')
    
    p_index = dataset.ordering.index('p')
    n_polarity = timesurface_size[2]
    histo_test = np.zeros([len(loader),n_polarity])
    labelz_true = []
    pbar = tqdm(total=len(loader))
    sample_number = 0
    for events, label in loader:
        events, label = events.squeeze(0), label.squeeze(0) # just one digit = one batch
        labelz_true.append(label)
        value, frequency = np.unique(events[:,p_index], return_counts=True)
        histo_test[sample_number,value] = frequency
        sample_number += 1
        pbar.update(1)
    pbar.close()
    
    histo_train = (histo_train.T/np.sum(histo_train, axis=1)).T
    histo_test = (histo_test.T/np.sum(histo_test, axis=1)).T
    
    if measure == 'knn':
        knn = KNeighborsClassifier(n_neighbors=k, weights='uniform', metric = 'euclidean', n_jobs = n_jobs)
    elif measure == 'euclidian':
        knn = KNeighborsClassifier(n_neighbors=1, weights='uniform', metric = 'euclidean', n_jobs = n_jobs)
    knn.fit(histo_train,labelz_train)
    labelz_hat = knn.predict(histo_test)
    accuracy = np.mean(labelz_hat==labelz_true)
    #elif measure == 'KL':
    #elif measure == 'EMD':
    #https://mathoverflow.net/questions/103115/distance-metric-between-two-sample-distributions-histograms
    
    return accuracy



def get_properties(events, target, ind_sample, values, ordering = 'xytp', distinguish_polarities = False):
    t_index, p_index = ordering.index('t'), ordering.index('p')
    if distinguish_polarities: 
        for polarity in [0,1]:
            events_pol = events[(events[:, p_index]==polarity)]
            isi = np.diff(events_pol[:, t_index])
            if 'mean_isi' in values.keys():
                values['mean_isi'][polarity, ind_sample, target] = (isi[isi>0]).mean()
            if 'median_isi' in values.keys():
                values['median_isi'][polarity, ind_sample, target] = np.median((isi[isi>0]))
            if 'null_isi' in values.keys():
                values['null_isi'][polarity, ind_sample, target] = (isi==0).mean()
            if 'nb_events' in values.keys():
                values['nb_events'][polarity, ind_sample, target] = events_pol.shape[0]
    else:
        events_pol = events
        isi = np.diff(events_pol[:, t_index])
        if 'mean_isi' in values.keys():
            values['mean_isi'][0, ind_sample, target] = (isi[isi>0]).mean()
        if 'median_isi' in values.keys():
            values['median_isi'][0, ind_sample, target] = np.median((isi[isi>0]))
        if 'null_isi' in values.keys():
            values['null_isi'][0, ind_sample, target] = (isi==0).mean()
        if 'nb_events' in values.keys():
            values['nb_events'][0, ind_sample, target] = events_pol.shape[0]
    if 'time' in values.keys():
        values['time'][0, ind_sample, target] = events[-1,t_index]-events[0,t_index]
        
    return values

def get_dataset_info(trainset, testset, properties = ['mean_isi', 'null_isi', 'nb_events'], distinguish_labels = False, distinguish_polarities = False):
    
    print(f'number of samples in the trainset: {len(trainset)}')
    print(f'number of samples in the testset: {len(testset)}')
    print(40*'-')
    
    #x_index, y_index, t_index, p_index = trainset.ordering.index("x"), trainset.ordering.index("y"), trainset.ordering.index("t"), trainset.ordering.index("p")
    nb_class = len(trainset.classes)
    nb_sample = len(trainset)+len(testset)
    nb_pola = 2
    
    values = {}
    for name in properties:
        values.update({name:np.zeros([nb_pola, nb_sample, nb_class])})

    ind_sample = 0
    
    loader = get_loader(trainset, shuffle=False)
    for events, target in loader:
        events = events.squeeze().numpy()
        values = get_properties(events, target, ind_sample, values, ordering = trainset.ordering, distinguish_polarities = distinguish_polarities)
        ind_sample += 1
                
    loader = get_loader(testset, shuffle=False)
    for events, target in loader:
        events = events.squeeze().numpy()
        values = get_properties(events, target, ind_sample, values, ordering = trainset.ordering, distinguish_polarities = distinguish_polarities)
        ind_sample += 1
        
    width_fig = 30
    fig, axs = plt.subplots(1,len(values.keys()), figsize=(width_fig,width_fig//len(values.keys())))
    for i, value in enumerate(values.keys()):
        if distinguish_polarities:
            x = []
            for p in range(nb_pola):
                x.append(values[value][p,:,:].sum(axis=1).ravel())
            ttl = value
        elif distinguish_labels:
            x = []
            for c in range(nb_class):
                x.append(values[value][0,np.nonzero(values[value][0,:,c]),c].ravel())
            ttl = value
        else:
            x = []
            x.append(values[value][0,:,:].sum(axis=1).ravel())
            ttl = value

        for k in range(len(x)):
            n, bins, patches = axs[i].hist(x=x[k], bins='auto',
                                    alpha=.5, rwidth=0.85)
            
        axs[i].grid(axis='y', alpha=0.75)
        axs[i].set_xlabel('Value')
        axs[i].set_ylabel('Frequency')
        axs[i].set_title(f'Histogram for the {ttl}')
        maxfreq = n.max()
        axs[i].set_ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)
        #axs[i].set_xscale("log")
        #axs[i].set_yscale("log")
        
    return values

def get_loader(dataset, kfold = None, kfold_ind = 0, num_workers = 0, shuffle=True, seed=42):
    # creates a loader for the samples of the dataset. If kfold is not None, 
    # then the dataset is splitted into different folds with equal repartition of the classes.
    if kfold:
        subset_indices = []
        subset_size = len(dataset)//kfold
        for i in range(len(dataset.classes)):
            all_ind = np.where(np.array(dataset.targets)==i)[0]
            subset_indices += all_ind[kfold_ind*subset_size//len(dataset.classes):
                            min((kfold_ind+1)*subset_size//len(dataset.classes), len(dataset)-1)].tolist()
        g_cpu = torch.Generator()
        g_cpu.manual_seed(seed)
        subsampler = torch.utils.data.SubsetRandomSampler(subset_indices, g_cpu)
        loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, sampler=subsampler, num_workers = num_workers)
    else:
        loader = torch.utils.data.DataLoader(dataset, shuffle=shuffle, num_workers = num_workers)
    return loader


#___________________________________________________________________

class LRtorch(torch.nn.Module):
    #torch.nn.Module -> Base class for all neural network modules
    def __init__(self, N, n_classes, bias=True):
        super(LRtorch, self).__init__()
        self.linear = torch.nn.Linear(N, n_classes, bias=bias)
        self.nl = torch.nn.Softmax(dim=1)
    def forward(self, factors):
        return self.nl(self.linear(factors))

def fit_MLR(path, 
            date, 
            tau_cla, #enter tau_cla in ms
            network = None,
            patch_size = None,
            max_duration = None,
            kfold = None,
            kfold_ind = 0,
        #parameters of the model learning
            num_workers = 0, # ajouter num_workers si besoin!
            learning_rate = 0.005,
            betas = (0.9, 0.999),
            num_epochs = 2 ** 5 + 1,
            seed = 42,
            verbose=True):
    
    if network:
        model_name = f'../Records/models/{network.get_fname()}_{int(tau_cla)}_{patch_size}_{kfold}_LR.pkl'
    else:
        model_name = f'../Records/models/{date}_RAW_{int(tau_cla)}_{patch_size}_{kfold}_LR.pkl'
    
    if os.path.isfile(model_name):
        print('load existing model')
        with open(model_name, 'rb') as file:
            logistic_model, losses = pickle.load(file)
    else:
        tau_cla *= 1e3
        sensor_size = (128,128,2)
        if patch_size: sensor_size = (patch_size[0], patch_size[1], 2)
        if network: sensor_size = (network.TS[0].camsize[0], network.TS[0].camsize[1], network.L[-1].kernel.shape[1])
        N = sensor_size[0]*sensor_size[1]*sensor_size[2]

        transform = tonic.transforms.Compose([tonic.transforms.ToTimesurface(sensor_size=sensor_size, tau=tau_cla, decay="exp")])
        if network:
            path_to_dataset = f'../Records/output/train/{network.get_fname()}_None/';
            dataset = HOTS_Dataset(path_to_dataset, timesurface_size, transform = transform)
        else:
            dataset = Synthetic_Dataset(save_to = path, train = True, patch_size = patch_size, max_duration = max_duration, transform = transform)
        loader = get_loader(dataset, kfold = kfold, kfold_ind = kfold_ind, num_workers = num_workers, seed = seed)
        if verbose: print(f'Number of training samples: {len(loader)}')

        torch.set_default_tensor_type("torch.DoubleTensor")
        criterion = torch.nn.BCELoss(reduction="mean")
        amsgrad = True #or False gives similar results
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if verbose:
            print(f'device -> {device} - num workers -> {num_workers}')

        n_classes = len(dataset.classes)
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
            if verbose:
                print(f'loss for epoch number {epoch}: {loss}')
            pbar.update(1)

        pbar.close()
        with open(model_name, 'wb') as file:
            pickle.dump([logistic_model, losses], file, pickle.HIGHEST_PROTOCOL)
            
    return logistic_model, losses

def predict_data(path,
                 date, 
                 tau_cla, #enter tau_cla in ms
                 network = None,
                 patch_size = None,
                 max_duration = None,
                 kfold = None,
                 kfold_ind = 0,
                 num_workers = 0,
                 seed=42,
                 verbose=False,
        ):
    
    if network:
        model_name = f'../Records/models/{network.get_fname()}_{int(tau_cla)}_{patch_size}_{kfold}_LR.pkl'
    else:
        model_name = f'../Records/models/{date}_RAW_{int(tau_cla)}_{patch_size}_{kfold}_LR.pkl'
    
    with open(model_name, 'rb') as file:
        model, loss = pickle.load(file)
    
    tau_cla *= 1e3
    sensor_size = (128,128,2)
    if patch_size: sensor_size = (patch_size[0], patch_size[1], 2)
    if network: sensor_size = (network.TS[0].camsize[0], network.TS[0].camsize[1], network.L[-1].kernel.shape[1])
    N = sensor_size[0]*sensor_size[1]*sensor_size[2]

    transform = tonic.transforms.Compose([tonic.transforms.ToTimesurface(sensor_size=sensor_size, tau=tau_cla, decay="exp")])
    if network:
        path_to_dataset = f'../Records/output/test/{network.get_fname()}_None/';
        dataset = HOTS_Dataset(path_to_dataset, timesurface_size, transform = transform)
    else:
        dataset = Synthetic_Dataset(save_to = path, train = False, patch_size = patch_size, max_duration = max_duration, transform = transform)
    loader = get_loader(dataset, kfold = kfold, kfold_ind = kfold_ind, num_workers = num_workers, seed = seed)
    if verbose: print(f'Number of training samples: {len(loader)}')
    
    with torch.no_grad():
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if verbose:
            print(f'device -> {device} - num workers -> {num_workers}')

        logistic_model = model.to(device)

        pbar = tqdm(total=len(loader))
        likelihood, true_target = [], []

        for X, label in loader:
            X, label = X[0].to(device) ,label[0].to(device)
            X = X.reshape(X.shape[0], N)
            n_events = X.shape[0]
            outputs = logistic_model(X)
            likelihood.append(outputs.cpu().numpy())
            true_target.append(label.cpu().numpy())
            pbar.update(1)
        pbar.close()

    return likelihood, true_target