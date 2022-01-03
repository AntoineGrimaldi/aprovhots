import torch, tonic, pickle
from dataset_creation import Synthetic_Dataset
from tqdm import tqdm
import numpy as np

def get_loader(dataset, kfold = None, kfold_ind = 0, num_workers = 0, seed=42):
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
        loader = torch.utils.data.DataLoader(dataset, shuffle=True, num_workers = num_workers)
    return loader

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
            patch_size = None,
            kfold = None,
            kfold_ind = 0,
        #parameters of the model learning
            num_workers = 0, # ajouter num_workers si besoin!
            learning_rate = 0.005,
            betas = (0.9, 0.999),
            num_epochs = 2 ** 5 + 1,
            seed = 42,
            verbose=True):
    
    nb_pola = 2
    tau_cla *= 1e3
    sensor_size = (128,128,nb_pola)
    if patch_size: sensor_size = (patch_size[0], patch_size[1], nb_pola)
    N = sensor_size[0]*sensor_size[1]*nb_pola

    transform = tonic.transforms.Compose([tonic.transforms.ToTimesurface(sensor_size=sensor_size, tau=tau_cla, decay="exp")])
    dataset = Synthetic_Dataset(save_to=path, train=True, patch_size=patch_size, transform=transform)
    loader = get_loader(dataset, kfold = kfold, kfold_ind = kfold_ind, num_workers = num_workers, seed=seed)
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
        print(f'loss for epoch number {epoch}: {loss}')
        pbar.update(1)

    pbar.close()
    with open(f'../Records/model/{date}_torch_model_{tau_cla}_{patch_size}.pkl', 'wb') as file:
        pickle.dump([logistic_model, losses], file, pickle.HIGHEST_PROTOCOL)
    return logistic_model, losses

def predict_data(date, 
                 tau_cla, #enter tau_cla in ms
                 patch_size = None,
                 num_workers = 0,
                 verbose=False,
        ):
    nb_pola = 2
    tau_cla *= 1e3
    sensor_size = (128,128,nb_pola)
    if patch_size: sensor_size = (patch_size[0], patch_size[1], nb_pola)
    N = sensor_size[0]*sensor_size[1]*nb_pola
    
    with open(f'../Records/model/{date}_torch_model_{tau_cla}_{patch_size}.pkl', 'rb') as file:
        model = pickle.load(file)

    transform = tonic.transforms.Compose([tonic.transforms.ToTimesurface(sensor_size=sensor_size, tau=tau_cla, decay="exp")])
    dataset = Synthetic_Dataset(save_to=path, train=False, patch_size=patch_size, transform=transform)
    loader = get_loader(dataset, kfold = kfold, kfold_ind = kfold_ind, num_workers = num_workers, seed=seed)
    if verbose: print(f'Number of training samples: {len(loader)}')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if verbose:
        print(f'device -> {device} - num workers -> {num_workers}')

    logistic_model = model.to(device)

    pbar = tqdm(total=nb_digit)
    likelihood, true_target, timing = [], [], []

    for X, label in loader:
        X, label = X[0].to(device) ,label[0].to(device)
        X = X.reshape(X.shape[0], N)
        n_events = X.shape[0]
        outputs = logistic_model(X)
        likelihood.append(outputs.cpu().numpy())
        true_target.append(label.cpu().numpy())
        timing.append(label.cpu().numpy())
        pbar.update(1)
    pbar.close()

    return likelihood, true_target