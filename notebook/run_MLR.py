from aprovhots import fit_MLR
from dataset_creation import Synthetic_Dataset

path = '/home/INT/grimaldi.a/Documents/projets/WP3/2021-12-06_simulator_data/'
patch_size = (16,16)
max_duration = 1e3
date = '2022-01-03'
# we set tau_cla emprirically based on the ISI for SEA samples (see Make_dataset notebook)
tau_cla = 200*patch_size[0]*patch_size[1]*2


dataset = Synthetic_Dataset(save_to=path, train=False, patch_size=patch_size, max_duration=max_duration)


#MLR_model, losses = fit_MLR(path, date, tau_cla, patch_size=patch_size, kfold=10)