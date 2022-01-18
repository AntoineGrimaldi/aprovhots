import torch, tonic, pickle, os
from dataset_creation import Synthetic_Dataset
from HOTS.tools import HOTS_Dataset, get_loader
from tqdm import tqdm
import numpy as np

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
        path_to_dataset = f'../Records/output/train/{network.get_fname()}_None/';
        dataset = HOTS_Dataset(path_to_dataset, timesurface_size, transform = transform)
    else:
        dataset = Synthetic_Dataset(save_to = path, train = True, patch_size = patch_size, max_duration = max_duration, transform = transform)
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