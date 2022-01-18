from aprovhots import fit_MLR
from dataset_creation import Synthetic_Dataset
from HOTS.tools import get_loader
from HOTS.network import network

path = '/home/INT/grimaldi.a/Documents/projets/WP3/2021-12-06_simulator_data/'
patch_size = (32,32)
max_duration = 1e3
date = '2022-01-03'
# we set tau_cla emprirically based on the ISI for SEA samples (see Make_dataset notebook)
tau_cla = 200*patch_size[0]*patch_size[1]*2

trainset = Synthetic_Dataset(save_to=path, train=True, patch_size=patch_size, max_duration=max_duration)

kfold = 30
loader = get_loader(trainset, kfold=kfold)

timestr = f'2022-01-17_synthetic_{patch_size}_{max_duration}'
name = 'homhots'
learn = True

for h1 in [.1,.2,.5,1,2]:
    for h2 in [.1,.2,.5,1,2]:
        for prototau in [.01,.1,1,10]:
            homeo = (h1,h2)
            tau = (prototau*2,prototau*4,prototau*8)

            hots = network(name=name, tau=tau, homeo=homeo, timestr=timestr, camsize = patch_size)
            hots.running(loader, trainset.ordering, trainset.classes, learn=learn)
#MLR_model, losses = fit_MLR(path, date, tau_cla, patch_size=patch_size, kfold=10)