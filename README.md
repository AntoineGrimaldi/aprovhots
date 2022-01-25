# aprovhots
Testing HOTS algorithm for coastline segmentation

## TODO:
- apply temporal jitter on the events from the simulator to avoid synchronous spiking (wait for Sotiris to check how the events are created)
- find good parameters (tau) for the segmentation
- apply the HOTS network to the segmentation
- make a visualization of the classification per event
- problems with the dataset:
    - a 'burst' of ON events alternating with a 'burst' of OFF events
    - balance the number of events per class