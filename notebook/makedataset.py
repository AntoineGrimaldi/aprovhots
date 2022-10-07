from dataset_creation import build_aprovis_dataset, Synthetic_Dataset
from aprovhots import get_dataset_info

# path where you'll go to find your .csv files to make the dataset with
path = '/home/INT/grimaldi.a/Documents/projets/WP3/2021-12-06_simulator_data/'
# gives a patch_size to divide spatially the event streams
patch_size = (64,64) 
# gives a max duration for the samples of the dataset to divide temporally the event streams
max_duration = 1e3
# labels given to the different classes of the dataset
labelz = ['sea','gro']
# original sensor_size of the DVS (width,height,polarity)
sensor_size = (128, 128, 2)
# discard samples with less than min_num_events events
min_num_events = 1000
# split the recordings into train and test sets with train_test_ratio ratio
train_test_ratio = .75
# gives the indexing of the event stream
ordering = 'xytp'

trainset = Synthetic_Dataset(save_to=path, train=True, patch_size=patch_size, max_duration=max_duration)
testset = Synthetic_Dataset(save_to=path, train=False, patch_size=patch_size, max_duration=max_duration)

print('-'*100)
print('-'*100)
print('DONE WITH FIRST ONE')
print('-'*100)
print('-'*100)

patch_size = (16,16) 

path = '/home/INT/grimaldi.a/Documents/projets/WP3/2022-02-05_simulator_data/'

trainset = Synthetic_Dataset(save_to=path, train=True, patch_size=patch_size, max_duration=max_duration)
testset = Synthetic_Dataset(save_to=path, train=False, patch_size=patch_size, max_duration=max_duration)
