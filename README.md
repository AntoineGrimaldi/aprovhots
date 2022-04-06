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

## Limits of the SpiNN-3
The SpiNN-3 board is composed of 4 chips, each composed of 18 cores. Since each core can simulate up to 256 spiking neurons, the SpiNN-3 can simulate around 18,000 neurons at most. 
Another limit applies to the number of input synapses connecting as input onto the neuron: according to [van Albada et al. (2018)](https://www.frontiersin.org/articles/10.3389/fnins.2018.00291/full#:~:text=The%20design%20specifications%20of%20SpiNNaker,otherwise%20the%20synchronization%20of%20the), the design specifications of SpiNNaker assume a connectivity of 1000 incoming synapses per neuron. More than this is not supported by the board.

It is to be noted that "In normal (default) use of the code, unless you change various parameters in your config file, there are actually only 63 cores available under normal circumstances: each chip has 18 cores, but 1 on each chip is used for the OS, 1 on each chip is used for further monitoring, and then 1 on the single ethernet chip for faster data I/O. If you really need 68 then the extra monitoring and faster data I/O can be turned off if necessary." [link](https://groups.google.com/u/0/g/spinnakerusers/c/_ROWtb8m-Wg/m/Fx5usieNBAAJ)

SpiNN-3 also allows the user to make use of spiking neurons' delays while implementing the network, however this delay cannot be greater than "144 time steps (i.e., 1 - 144ms when using 1ms time steps, or 0.1 - 14.4ms when using 0.1ms time steps). Delays of more than 16 time steps require an additional “delay population” to be added; this is done automatically by the software when such delays are detected." [link](spinnakermanchester.github.io/spynnaker/6.0.0/SPyNNakerModelsAndLimitations.html)
