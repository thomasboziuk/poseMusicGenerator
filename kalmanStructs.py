import sympy
from sympy.utilities.lambdify import lambdify
import numpy as np
from numpy.linalg import inv

def genNXYVelMats(NP, sigma_P = .1, sigma_P_dot = 0.05, sigma_R = 0.1, sigma_Q = 0.5, sigma_Q_dot = 0.1):
    '''
    This function will create and return the initial inputs for the class kalmanFilt.
    You simply need to pass it the number of points that will be measured.
    It will assume a constant velocity model, so the state will be size n = 2x2xNP
    and the measurements will be size m = 2xNP
    Entries will be created for X1,Y1,...XN,YN,,X1_dot,Y1_dot...XN_dot,YN_dot
    Optionally, pass varainces for:
    sigma_P, for position states initially in P
    sigma_P_dot, for velocity states initially in P
    sigma_R, for measured values used in R
    sigma_Q, for process noise for position used in Q
    sigma_Q_dot, for process noise for velocity used in Q
    This does not support different variances for different points or covariances; if required,
    you will need to manually adjust after generation.
    Returned to you will be numpy arrays (except A, a sympy matrix):
    seed state x, n x 1
    state covariance matrix P, n x n
    measurement covariance matrix R, m x m
    process noise covariance matrix Q, n x n
    state transition matrix A, which must be a sympy matrix containing dt, n x n
    state-to-measurement matrix H, m x n
    '''

    n = 4 * NP
    m = 2 * NP
    dt = sympy.symbols('dt')

    x = np.concatenate((0.5*np.ones((1,m)), 0.1*np.ones((1,m))), axis = 1)[0]
    P = np.diag(np.concatenate((sigma_P*np.ones((1,m)), sigma_P_dot*np.ones((1,m))), axis = 1)[0])
    R = sigma_R * np.eye(m)
    Q = np.diag(np.concatenate((sigma_Q*np.ones((1,m)), sigma_Q_dot*np.ones((1,m))), axis = 1)[0])
    A = sympy.eye(m).row_join(dt * sympy.eye(m)).col_join(sympy.zeros(m).row_join(sympy.eye(m)))
    H = np.eye(m,n)

    return x, P, R, Q, A, H


def genNXYVelAccelMats(NP, sigma_P = .1, sigma_P_dot = 0.05, sigma_P_dot_dot = 0.025, sigma_R = 0.1, sigma_Q = 0.5, sigma_Q_dot = 0.1, sigma_Q_dot_dot = 0.05):
    '''
    This function will create and return the initial inputs for the class kalmanFilt.
    You simply need to pass it the number of points that will be measured.
    It will assume a constant acceleration model, so the state will be size n = 2x3xNP
    (you will get back position, velocity, and acceleration)
    and the measurements will be size m = 2xNP
    Entries will be created for X1,Y1,...XN,YN,,X1_dot,Y1_dot...XN_dot,YN_dot... X1_dot_dot,Y1_dot_dot,....XN_dot_dot,XN_dot_dot
    Optionally, pass varainces for:
    sigma_P, for position states initially in P
    sigma_P_dot, for velocity states initially in P
    sigma_P_dot_dot, for velocity states initially in P
    sigma_R, for measured values used in R
    sigma_Q, for process noise for position used in Q
    sigma_Q_dot, for process noise for velocity used in Q
    sigma_Q_dot_dot, for process noise for acceleration used in Q
    This does not support different variances for different points or covariances; if required,
    you will need to manually adjust after generation.
    Returned to you will be numpy arrays (except A, a sympy matrix):
    seed state x, n x 1
    state covariance matrix P, n x n
    measurement covariance matrix R, m x m
    process noise covariance matrix Q, n x n
    state transition matrix A, which must be a sympy matrix containing dt, n x n
    state-to-measurement matrix H, m x n
    '''

    n = 6 * NP
    m = 2 * NP
    dt = sympy.symbols('dt')

    x = np.concatenate((0.5*np.ones((1,m)), 0.1*np.ones((1,m)), 0.05*np.ones((1,m))), axis = 1)[0]
    P = np.diag(np.concatenate((sigma_P*np.ones((1,m)), sigma_P_dot*np.ones((1,m)), sigma_P_dot_dot*np.ones((1,m))), axis = 1)[0])
    R = sigma_R * np.eye(m)
    Q = np.diag(np.concatenate((sigma_Q*np.ones((1,m)), sigma_Q_dot*np.ones((1,m)), sigma_Q_dot_dot*np.ones((1,m))), axis = 1)[0])
    A = sympy.eye(m).row_join(dt * sympy.eye(m)).row_join(1/2 *dt**2 * sympy.eye(m)).col_join(sympy.zeros(m).row_join(sympy.eye(m)).row_join(dt * sympy.eye(m))).col_join(sympy.zeros(m).row_join(sympy.zeros(m)).row_join(sympy.eye(m)))
    H = np.eye(m,n)

    return x, P, R, Q, A, H

class kalmanFilt:
    '''
    Kalman Filter for process with m measurement dimension and n state dimension.
    Initialize with numpy arrays except A:
    seed state x, n x 1
    state covariance matrix P, n x n
    measurement covariance matrix R, m x m
    process noise covariance matrix Q, n x n
    state transition matrix A, which must be a sympy matrix containing dt, n x n
    state-to-measurement matrix H, m x n
    '''
    def __init__(self, x, P, R, Q, A, H):
        self.dt = sympy.symbols('dt')
        self.x = x
        self.P = P
        self. R = R
        self.Q = Q
        self.A = A
        self.Afunc = lambdify(self.dt, self.A, modules = 'numpy')
        self.H = H
        
    
    def update(self, z, dt):
        '''
        update the Kalman Filter with measurement z and timestep dt
        returns new state.
        '''
        # cast state transition matrix A to numpy array based on given dt
        self.A = self.Afunc(dt)

        xp = self.A.dot(self.x)
        Pp = self.A.dot(self.P).dot(self.A.T) + self.Q
        S = self.H.dot(Pp).dot(self.H.T) + self.R
        K = self.P.dot(self.H.T).dot(inv(S))
        self.x = xp + K.dot(z - self.H.dot(xp))
        self.P = Pp - K.dot(self.H).dot(Pp)
        return self.x


    def propagate(self, dt):
        '''
        propagate the state forward by timestep dt.
        returns a new state but does not change parameters or update current state estimate.
        you will need to account for that when passing in a future dt
        '''
        # cast state transition matrix A to numpy array based on given dt

        self.A = self.Afunc(dt)
        xp = self.A.dot(self.x)
        return xp