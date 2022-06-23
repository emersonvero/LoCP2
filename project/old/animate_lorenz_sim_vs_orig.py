import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

r_sim = np.load("./data/lorenz_r_sim.npy")
r = np.load("./data/lorenz_r.npy")
t = np.load("./data/lorenz_t.npy")

N=100
frames = np.arange(0,len(t),N)


# Print iterations progress
def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Print New Line on Complete
    if iteration == total: 
        print()



def update_traj(num,trajs,lines,points):
    # traj in trajs are (n_samples,3) shaped
    printProgressBar(num,len(t))
    for traj, line, point in zip(trajs,lines,points):
        line.set_data(traj[:num,:2].T)
        line.set_3d_properties(traj[:num,2])
        point.set_data(traj[num,:2].T)
        point.set_3d_properties(traj[num,2])
    return lines,points


# Attaching 3D axis to the figure
fig = plt.figure()
ax = fig.add_subplot(projection="3d")

# Create lines initially without data
colors = ['C2','C1']
trajs = [r,r_sim]
lines = [ax.plot([], [], [],marker="",ls='--',color=color,alpha=0.5)[0] for color in colors]
points = [ax.plot([],[],[],marker="o",ls="",color=color)[0] for color in colors]

# Setting the axes properties
ax.set(xlim3d=(-20, 20), xlabel='X')
ax.set(ylim3d=(-25, 28), ylabel='Y')
ax.set(zlim3d=(0, 50), zlabel='Z')

# Creating the Animation object
ani = animation.FuncAnimation(
    fig, update_traj, frames, fargs=(trajs, lines , points), interval=10)

ani.save("./img/lorenz_sim_vs_orig.gif")
