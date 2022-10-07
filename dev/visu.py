import matplotlib.pyplot as plt 
from matplotlib.animation import FuncAnimation, PillowWriter

def visualization(events, ordering, sensor_size):
    #options
    frame_interval = 1e3
    figure_interval = 200
    
    x_index, y_index = ordering.index('x'), ordering.index('y')
    t_index, p_index = ordering.index('t'), ordering.index('p')
    # initialise the figure
    timestamps=0
    fig_events = plt.figure()
    ax = plt.axes(xlim=(0,sensor_size[0]), ylim=(0,sensor_size[1]))
    scatter_neg_events = ax.scatter([],[], marker="s", animated=True, color="springgreen", label="SEA")
    scatter_pos_events = ax.scatter([],[], marker="s", animated=True, color="dodgerblue", label="GROUND")

    # define the animation
    def animate(i):
        events_in_frame = events[(events[:,t_index] >= i*frame_interval) & (events[:,t_index] < (i+1)*frame_interval)]
        scatter_pos_events.set_offsets(events_in_frame[events_in_frame[:,p_index] == 1][: , [x_index,y_index]])
        scatter_neg_events.set_offsets(events_in_frame[events_in_frame[:,p_index] == 0][: , [x_index,y_index]])
        
        #print(i, i*frame_interval)
        #if events_in_frame.shape[0]>0:
        #    print(events_in_frame.shape)

        return scatter_pos_events, scatter_neg_events,

    animation = FuncAnimation(fig_events, animate, blit=True, interval=figure_interval, save_count=1000)
    #plt.title("Events from "+args.events[0]+" over time")
    plt.xlabel("Width (in pixels)")
    plt.ylabel("Height (in pixels)")
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=2)
    plt.draw()

    # save the video
    name = "animated.gif" 
    writergif = PillowWriter(fps=frame_interval*1000) 
    animation.save(name, writer=writergif)
    plt.show()