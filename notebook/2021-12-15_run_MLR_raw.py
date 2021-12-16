from aprovhots import get_labels_indices, fit_MLR
data_path = '/home/INT/grimaldi.a/Documents/projets/WP3/2021-12-06_simulator_data/'
#mean_isi = (isi_OFF+isi_ON)/2
patch_size = 5
mean_isi = 190*1e3
tau_cla = mean_isi*patch_size**2*1e-3 # tau has to be given in ms 
labelz = ['sea', 'ground']

split2patches(data_path, patch_size)

events_train, indices_train, label_train = get_labels_indices(data_path, labelz, patch_size)

model, loss = fit_MLR(events_train, label_train, indices_train, tau_cla, patch_size)