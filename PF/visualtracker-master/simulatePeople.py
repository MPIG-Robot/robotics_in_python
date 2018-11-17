'''
Person implementation to simulate movement in a room

In every time step t, move in a random direction a small amount
UNITS: METERS
'''
# from matplotlib import pyplot as plt
# import mpl_toolkits.mplot3d.axes3d as p3
# from matplotlib import animation

import math
import random as rand
import numpy as np

MAX_TIME = 24 #seconds
T_STEP = 1.0/25.0
THRES = 0.1

class Person(object):
    def __init__(self, start, angle):
        self.curPos = np.array(start) #[X,Y,Z]
        self.angle = angle

    def randomMove(self):
        d = rand.random()*0.05
        phi = rand.random()
        if phi < THRES:
            if phi >= THRES/2.0:
                self.angle += math.fmod(phi*2*math.pi, 2*math.pi)
            else:
                self.angle -= math.fmod(phi*2*math.pi, 2*math.pi)

        newPos = np.array([0.0,0.0,self.curPos[2]])
        newPos[0] = self.curPos[0] + d*math.cos(self.angle)
        newPos[1] = self.curPos[1] + d*math.sin(self.angle)
        return newPos

def simulatePeopleTest():
    p1TrajX = []
    p1TrajY = []
    p1TrajZ = []
    p2TrajX = []
    p2TrajY = []
    p2TrajZ = []

    person1 = Person([0,0,1.7], 0)
    person2 = Person([5,5,1.7], math.pi)
    
    fig = plt.figure()
    ax = p3.Axes3D(fig)

    p1TrajX.append(person1.curPos[0])
    p1TrajY.append(person1.curPos[1])
    p1TrajZ.append(person1.curPos[2])

    p2TrajX.append(person2.curPos[0])
    p2TrajY.append(person2.curPos[1])
    p2TrajZ.append(person2.curPos[2])

    curTime = 0
    while curTime < MAX_TIME:
        person1.randomMove()
        person2.randomMove()

        p1TrajX.append(person1.curPos[0])
        p1TrajY.append(person1.curPos[1])
        p1TrajZ.append(person1.curPos[2])

        p2TrajX.append(person2.curPos[0])
        p2TrajY.append(person2.curPos[1])
        p2TrajZ.append(person2.curPos[2])

        curTime += T_STEP

    ax.scatter(p1TrajX,p1TrajY,p1TrajZ,depthshade=False)
    ax.scatter(p2TrajX,p2TrajY,p2TrajZ,c="red",depthshade=False,)
    plt.show()

# # create the first plot
# point, = ax.plot([x[0]], [y[0]], [z[0]], 'o')
# line, = ax.plot(x, y, z, label='parametric curve')
# ax.legend()
# ax.set_xlim([-1.5, 1.5])
# ax.set_ylim([-1.5, 1.5])
# ax.set_zlim([-1.5, 1.5])

# # second option - move the point position at every frame
# def update_point(n, x, y, z, point):
#     point.set_data(np.array([x[n], y[n]]))
#     point.set_3d_properties(z[n], 'z')
#     return point

# ani=animation.FuncAnimation(fig, update_point, 99, fargs=(x, y, z, point))

#simulatePeopleTest()

