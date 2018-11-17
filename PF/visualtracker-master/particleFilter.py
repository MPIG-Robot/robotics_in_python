import numpy as np
import math
import random as rand
import copy

import simulatePeople as sim

from matplotlib import pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3

Np = 1 #Number of particles
B = 200
Tv = 1/25.0
BETA = 10.0
v_x = 1 #m/s
a_x = math.exp(-BETA*Tv)
b_x = v_x * math.sqrt(1 - math.pow(a_x,2))
MAX_TIME = 24 #seconds
SIGMA = 1
m = 4 #Number of people
print 'a_x', a_x
print 'b_x', b_x

class Particle(object):
	def __init__(self, pos, velocity):
		self.pos = pos
		self.velocity = velocity

'''
obs_i = Z_i,k
pred_i = X_i,k
'''
def dynamicProb(obs_i, pred_i):
	return math.exp(-math.pow(np.linalg.norm(obs_i - pred_i), 2)/(2.0*math.pow(SIGMA,2)))

def dist(a,b):
	return math.sqrt(math.pow(b[0]-a[0],2) + math.pow(b[1]-a[1],2) + math.pow(b[2]-a[2],2))

def updateParticle(p,Z_t):
	prev_vel = p.velocity
	vel = prev_vel

	#Get vel closest to prev_vel
	for i in range(m):
		# vel[i][0] = a_x*prev_vel[i][0] + b_x*rand.random()
		# vel[i][1] = a_x*prev_vel[i][1] + b_x*rand.random()
		bestJ = 0
		closestDist = dist(Z_t[bestJ],p.pos[i])
		for j in range(m):
			d = dist(Z_t[j],p.pos[i])
			if d < closestDist:
				bestJ = j
				closestDist = d
		
		vel[i][0] = (Z_t[bestJ][0] - p.pos[i][0])/Tv + b_x*rand.random()
		vel[i][1] = (Z_t[bestJ][1] - p.pos[i][1])/Tv + b_x*rand.random()
		# vel[i][0] = (Z_t[i][0] - p.pos[i][0])/Tv + b_x*rand.random()
		# vel[i][1] = (Z_t[i][1] - p.pos[i][1])/Tv + b_x*rand.random()
	pos = p.pos + Tv*vel

	p.pos = pos
	p.vel = vel

def getColor(i):
	color = None
	if i == 0:
			color = 'blue'
	elif i == 1:
		color = 'red'
	elif i == 2:
		color = 'orange'
	else:
		color = 'green'
	return color


def runPF():
	#Initialize people
	peopleTraj = []
	people = []
	rand.seed()

	for i in range(m):
		x = rand.random()*10
		theta = rand.random()*2*math.pi
		people.append(sim.Person([x,x,1.7], theta))
		peopleTraj.append([np.array([x,x,1.7])])

	#Initialize particle filter
	Z_0 = [p.curPos for p in people]
	for i in range(m):
		newPos = people[i].randomMove()
		people[i].curPos = newPos
		peopleTraj[i].append(newPos)
	Z_1 = [p.curPos for p in people]

	#For each particle
	particles = []
	weights = []
	for n in range(Np):
		r = rand.randint(0,m-1)
		chosen = []
		positions = []
		velocities = []

		for i in range(m):
			r = rand.randint(0,m-1)
			while r in chosen:
				r = rand.randint(0,m-1)
			chosen.append(r)

			positions.append(Z_1[r])
			v = (Z_1[r] - Z_0[i])/Tv
			velocities.append(v)

		particles.append(Particle(np.array(positions), np.array(velocities)))
		weights.append(1/float(Np)) #Equal weights

	###########################################################################################################

	t = Tv*2
	predictedPos = [[] for i in range(m)]
	while(t < MAX_TIME):
		Z_t = []
		#New observations
		for i in range(m):
			newPos = people[i].randomMove()
			people[i].curPos = newPos
			peopleTraj[i].append(newPos)
			Z_t.append(newPos)

		rand.shuffle(Z_t) #Shuffle observations, we don't know order in real life

		#Re-sample
		new_particles = np.random.choice(particles, B + Np,True, weights) #TODO: Fix sampling distribution later
		new_particles = new_particles[B:] #Remove first B samples

		positions = [n.pos for n in new_particles]

		#Advance samples
		for p in new_particles:
			updateParticle(p,Z_t)

		#Update weights
		new_weights = []
		sumWeights = 0
		for n in range(Np):
			w = dynamicProb(Z_t, new_particles[n].pos)
			new_weights.append(w)
			sumWeights += w
		if sumWeights == 0:
			print "Weird"
			new_weights = [1/float(Np) for w in new_weights]
		else:
			new_weights = [w/sumWeights for w in new_weights]

		#TODO: MCMC sampling
		#Ramdomly select speaker i
		i = rand.randint(0,m-1)
		#Sample new state X_star
		q_weights = []
		for n in range(Np):
			q_weights.append(0)
		#Compute acceptance ratio
		#Draw randomly to accept/reject

		#Output position/velocity for this time
		for i in range(m):
			#pos = np.array([0.0,0.0,0.0])
			bestPos = new_particles[0].pos[i]
			bestWeight = new_weights[0]
			for n in range(Np):
				if new_weights[n] > bestWeight:
					bestPos = new_particles[n].pos[i]
				#pos += new_weights[n]*new_particles[n].pos[i] / sumWeights
			predictedPos[i].append(bestPos)
			#print bestWeight

		particles = new_particles
		weights = new_weights
		t += Tv

	#Plot results
	fig = plt.figure()
	ax = p3.Axes3D(fig)
	for i in range(m):
		color = getColor(i)
		x = [p[0] for p in predictedPos[i][1:]]
		y = [p[1] for p in predictedPos[i][1:]]
		z = [p[2] for p in predictedPos[i][1:]]
		ax.scatter(x,y,z,c=color,depthshade=False)


	fig = plt.figure()
	ax = p3.Axes3D(fig)
	for i in range(m):
		color = getColor(i)
		x = [p[0] for p in peopleTraj[i]]
		y = [p[1] for p in peopleTraj[i]]
		z = [p[2] for p in peopleTraj[i]]
		#print peopleTraj[i]

		ax.scatter(x,y,z,c=color,depthshade=False)

	plt.show()

runPF()
