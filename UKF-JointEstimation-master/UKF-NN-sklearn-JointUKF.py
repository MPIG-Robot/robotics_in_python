###########################################################
# Copyright @ 2018
# Donghai He (gsutilml@gmail.com)
# Free to use as long as keep this information at top.
###########################################################
# Joint UKF
import copy
import numpy as np
from filterpy import kalman, common
from sklearn.neural_network import MLPRegressor
from matplotlib import pyplot as pl
import observations
import ArgUKF

# Parameters
n_fx_input    = 1 # process fx input
n_fx_output   = 1 # process output
n_hx_input    = 1 # measurement hx input
n_hx_output   = 1 # measurement hx output
n_nn_hidden   = [4] # number of neurons for each layer
#nn_activation = 'tanh'
nn_activation = 'identity'
#nn_activation = 'logistic'
# number of NN fx model states in ukf
n_hidden_weights = [n_fx_input * n_nn_hidden[0]] + [n_nn_hidden[k-1]*n_nn_hidden[k] for k in range(1,len(n_nn_hidden))]
n_output_weights   = n_nn_hidden[len(n_nn_hidden)-1] * n_fx_output
n_total_weights    = sum(n_hidden_weights) + n_output_weights
n_total_intercepts = sum(n_nn_hidden) + n_fx_output
n_total_nn_parameters = n_total_weights + n_total_intercepts
n_joint_ukf_process_states    = n_fx_input + n_total_nn_parameters
n_ukf_process_noise= n_fx_input
n_joint_ukf_process_noise  = n_ukf_process_noise + n_total_nn_parameters
Xdim = n_joint_ukf_process_states
Vdim = n_joint_ukf_process_noise
Ndim = 1 # measurement 
Ldim = Xdim+Vdim+Ndim

# create NN model 
model = MLPRegressor(hidden_layer_sizes=tuple(n_nn_hidden),  activation=nn_activation, solver='lbfgs', alpha=0.001, batch_size='auto', learning_rate='constant', \
    learning_rate_init=0.00001, power_t=0.5, max_iter=1, shuffle=True, random_state=9, tol=1000, verbose=False, warm_start=False, momentum=0.9, \
    nesterovs_momentum=True, early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e08)

# fake fit to initialze coef
model.fit(np.random.normal(size=(1,n_fx_input)), np.random.normal(size=(1,n_fx_output)))

def array2coef_intercepts(w):
    left = 0

    coef = []
    weight_sizes = [n_fx_input] + n_nn_hidden + [n_fx_output]
    for i in range(len(n_nn_hidden)+1):
        weight_size = weight_sizes[i]*weight_sizes[i+1]
        coef.append(w[left:(left + weight_size)].reshape(weight_sizes[i], weight_sizes[i+1],))
        left += weight_size

    intercepts = []
    weight_sizes = n_nn_hidden + [n_fx_output]
    for i in range(len(n_nn_hidden)+1):
        intercepts.append(w[left:(left + weight_sizes[i])].reshape(1, weight_sizes[i]))
        left += weight_sizes[i]
    
    return coef, intercepts

#   process model with NN
def fx(nn_x, dt):
    x = copy.deepcopy(nn_x)
    update_nn_weights(nn_x) ####  
    x_pred = model.predict(np.array(x[:n_fx_input]).reshape(1,n_fx_input))

    offset = n_total_nn_parameters + n_ukf_process_noise
    for i in range(n_fx_input):
        x[i] = x_pred[i] + nn_x[i + offset]
    # process noise
    for i in range(n_fx_input, n_joint_ukf_process_states):
        x[i] = x[i] + nn_x[i + offset]  
    return x

#   measurement model with NN
def hx(nn_x):
    return np.array([nn_x[0]])

def update_nn_weights(nn_x):
    w = copy.deepcopy(nn_x[n_fx_input:])
    coef, intercepts = array2coef_intercepts(w)
    model.coefs_ = coef
    model.intercepts_ = intercepts

def get_nn_weights(nn_model):
    return (copy.deepcopy(nn_model.coefs_), copy.deepcopy(nn_model.intercepts_))

def set_nn_weights(nn_model, coefs, intercepts):
    nn_model.coefs_ = coefs
    nn_model.intercepts_ = intercepts

#    System Model
#%   This filter assumes the following standard state-space model:
#%
#%     x(k) = ffun[x(k-1),v(k-1),U1(k-1)]
#%     y(k) = hfun[x(k),n(k),U2(k)]

dt = 0.1
# create sigma points to use in the filter. This is standard for Gaussian processes
points = kalman.MerweScaledSigmaPoints(Ldim, alpha=.001, beta=2., kappa=0)

kf = ArgUKF.UnscentedKalmanFilter(dim_x=Xdim, dim_z=n_hx_output, dt=dt, fx=fx, hx=hx, points=points)

kf.Pn = np.diag([6.1412]) # 1 standard
for i in range (n_fx_input):
    kf.Pv[i][i]= 0.01 # initial uncertainty of process noise state
for i in range (n_fx_input,Vdim):
    kf.Pv[i][i]= 0.0001 # initial uncertainty of process noise state

# Train
#obs = observations.load('observations-sin.txt')
obs = observations.load('observations-multi-model.txt')
num_use_obs = len(obs)

# EMA for comparison
ema_alpha = 0.5
ema = [0]*len(obs)
for i in range(len(obs)):
    ema [i] = ema_alpha*ema[i-1] + (1-ema_alpha)*obs[i] if i>0 else obs[i]

rd = 0
kf.x = np.array(np.zeros(kf.x.shape)) # initial state
for rd in range(10):
    zs = obs[:num_use_obs] #[[i/10.0+np.random.randn()*z_std] for i in range(100)] # measurements
    x_priors=[]
    x_posts=[]
    for z in zs:
        kf.predict()
        x_prior = copy.deepcopy(kf.x_prior)
        x_priors.append(x_prior)
        kf.update(z)
        x_post = copy.deepcopy(kf.x)
        x_posts.append(x_post)
        # Update process noise source (which can avoid oscillation)
        # method: anneal
        idx = [i for i in range(n_fx_input, kf.Px.shape[0])] # index of parameter block
        for i in idx:
            kf.Pv[i][i]= max(0.995 * kf.Pv[i][i], 0.0000001)
        print('Round=',rd, ', x_prior=',x_prior,', x_post=',x_post, ', log-likelihood', kf.log_likelihood)

    # draw chart
    pl.figure(rd)
    pl.scatter(range(len(zs)), [z for z in zs], c='b')
    pl.plot(range(len(x_priors)),[x[0] for x in x_priors],'y', \
     range(len(x_posts)),[x[0] for x in x_posts],'r', range(len(x_posts)), ema[:len(x_posts)], 'g')
    pl.title('Round='+str(rd))
    pl.legend(['x_prior','x_post', 'ema', 'Observation'])

    # update NN weights
    update_nn_weights(kf.x)
    # reset states, keep NN weights
    kf.x[:n_fx_input].fill(0.0)
    kf.x_prior[:n_fx_input].fill(0.0)
    # copy process covariance
    sx = np.eye(n_fx_input,dtype=float)
    idx = [i for i in range(n_fx_input, kf.Px.shape[0])]
    sv = kf.Px[np.ix_(idx,idx)]
    kf.Px = np.block([[sx, np.zeros((n_fx_input, len(idx)))],[np.zeros((len(idx), n_fx_input)),sv]])

# Validate
pl.figure(rd+1)
X=np.arange(0,100,0.1)
y=[model.predict(x) for x in X]
pl.plot(X, X, 'r', X, y,'g')
pl.legend(['True','Predict'])

pl.show()
