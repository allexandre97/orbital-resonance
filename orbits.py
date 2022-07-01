import math
import numpy as np
import matplotlib.pyplot as plt

class Planet:
    def __init__(self, Name, Radius, Period):        
        self.Name      = Name
        self.Radius    = Radius
        self.Perimeter = 2*np.pi*Radius
        self.Period    = Period
        self.AngVel    = 2*np.pi/(Period + 1e-9)
    
    def CalcTraj(self, NPoints, NOrbits):
        
        dTheta = NOrbits*2*np.pi/NPoints
        
        Theta = np.linspace(0, NPoints, NPoints)*dTheta
        
        traj = np.zeros((NPoints, 2))
        
        traj[:,0] = self.Radius*np.cos(Theta)
        traj[:,1] = self.Radius*np.sin(Theta)
        
        self.Trajectory = traj
        self.NOrbits    = NOrbits


#Periods = [1, 225/88, 365/88, 687/88, 4332/88, 10759/88, 30689/88, 60195/88]

#Radius  = [1, 0.73/0.39, 1/0.39, 1.53/0.39, 5.21/0.39, 9.58/0.39, 19.19/0.39, 30.07/0.39]

Periods = [1, 2, 4, 7, 11, 16]

Radius  = [1, 1, 1, 1, 1, 1, 1, 1]

Trajs   = []
Bounds = []

for p1, r1 in zip(Periods, Radius):
    trj = []
    bnd = []
    for p2, r2 in zip(Periods, Radius):

        Planet1 = Planet('One', r1, p1)
        Planet2 = Planet('Two', r2, p2)

        lcm = 3*math.lcm(int(p1), int(p2))

        N1  = lcm/p1
        N2  = lcm/p2

        Planet1.CalcTraj(3*60*10, N1)
        Planet2.CalcTraj(3*60*10, N2)

        if p1 != p2:

            Planet1 = Planet('One', r1, p1)
            Planet2 = Planet('Two', r2, p2)

            lcm = 3*math.lcm(int(p1), int(p2))

            N1  = lcm/p1
            N2  = lcm/p2

            Planet1.CalcTraj(3*60*10, N1)
            Planet2.CalcTraj(3*60*10, N2)
            Trajectory = Planet2.Trajectory - Planet1.Trajectory

        else:

            Planet1 = Planet('One', Radius[0], Periods[0])
            Planet2 = Planet('Two', r2, p2)

            lcm = 3*math.lcm(int(Periods[0]), int(p2))

            N1 = lcm/Periods[0]
            N2 = lcm/p2

            Planet1.CalcTraj(3*60*10, N1)
            Planet2.CalcTraj(3*60*10, N2)

            Trajectory = Planet1.Trajectory

        minX, maxX = np.min(Trajectory[:,0]), np.max(Trajectory[:,0])
        minY, maxY = np.min(Trajectory[:,1]), np.max(Trajectory[:,1])

        trj.append(Trajectory)
        bnd.append([[minX, maxX],[minY,maxY]])

    Trajs.append(trj)
    Bounds.append(bnd)

Trajs  = np.array(Trajs)
Bounds = np.array(Bounds)

def Animate(Frame, Trajectory, Bounds, Trail):
    
    if Frame < Trail:
        beg = 0
        end = Frame+1 
    else:
        beg = Frame - Trail
        end = Frame
    
    aux = np.linspace(0, 1, (end-beg))
    
    maxColor = 1
    A_color = maxColor/(np.e - 1)
    B_color = -A_color
    
    maxSize = 50
    A_size  = maxSize/(np.e - 1)
    B_size  = -A_size

    fig, ax = plt.subplots(Trajectory.shape[0], Trajectory.shape[1], figsize = (12,12))

    for row in range(Trajectory.shape[0]):
        for col in range(Trajectory.shape[1]):

            segment    = Trajectory[row,col,beg:end]
            minX, maxX = Bounds[row,col,0,0], Bounds[row,col,0,1]
            minY, maxY = Bounds[row,col,1,0], Bounds[row,col,1,1]

            if minX == maxX:
                minX, maxX = -1, 1
            if minY == maxY:
                minY, maxY = -1, 1
            ax[row,col].scatter(Trajectory[row,col,0:end,0], Trajectory[row,col,0:end,1],
                    s = 2,
                    color = 'gray',
                    edgecolors = None,
                    alpha = 0.2)
            
            ax[row,col].scatter(segment[:,0], segment[:,1],
                    color = plt.cm.gnuplot(A_color*np.exp(aux) + B_color),
                    s = A_size*np.exp(aux) + B_size,
                    alpha = aux)
    
            ax[row,col].set_xlim(minX*(1+0.05), maxX*(1+0.05))
            ax[row,col].set_ylim(minY*(1+0.05), maxY*(1+0.05))

            ax[row,col].axis('off')

    fig.tight_layout()
    fig.savefig('frame_%i.png' % (Frame+1), dpi = 150, transparent = True)
    
    plt.close()


if __name__=='__main__':

    import os
    import multiprocessing as mp
    from multiprocessing import set_start_method
    from functools import partial
    
    os.system('rm *png')

    set_start_method("spawn")
    
    run = partial(Animate,
                  Trajectory = Trajs,
                  Bounds = Bounds,
                  Trail = 20)
    
    Frames = np.arange(1800)
    
    pool = mp.Pool(os.cpu_count()-1)
    
    plt.ioff()
    
    pool.map(run, Frames)

    plt.ion()

    os.system('ffmpeg -r 60 -f image2 -s 1920x1080 -i frame_%0d.png -vcodec libx264 -crf 25  -pix_fmt yuv420p test.mp4 -y')
