# -*- coding: utf-8 -*-
"""
Created on Wed Jun 18 14:59:30 2025

@author: tateb
"""

# REFERENCE bo_gen_04-08-25.py IN two_tanks FOLDER FOR ORIGINAL.
# import os
# os.environ["OMP_NUM_THREADS"] = "1"  # Must come before numpy, GPy, etc.
import matplotlib.pyplot as plt
import GPy
import numpy as np
# from numpy.random import seed
# seed(12345)
from emukit.core import ParameterSpace, ContinuousParameter
from emukit.model_wrappers.gpy_model_wrappers import GPyModelWrapper
from emukit.bayesian_optimization.loops import BayesianOptimizationLoop
from emukit.core.loop import LoopState # Needed for getting current evaluation
from emukit.core.loop import FixedIterationsStoppingCondition
from emukit.core.loop import StoppingCondition
# from decimal import Decimal, ROUND_HALF_UP # Good for rounding number with more control than round()
# from IPython.display import display

# from scipy.signal import find_peaks
from scipy.stats import qmc
# import itertools
import csv
import os
# from IPython import display
from scipy.integrate import odeint
# from simple_pid import PID
# import datetime
import control as ctl
# import math
import pandas as pd
# from joblib import Parallel, delayed, parallel_backend
# import tempfile


# Not sure what this does, but need it to access log like other conditions do
import logging
_log = logging.getLogger(__name__)

class Bound_or_Cluster_Stop(StoppingCondition):
    def __init__(self, bounds: np.ndarray, boundary_margin: float = 0.05, N: int = 8, 
                 M: int = 10, top_count: int = 10, eps_x: float = 1e-2, eps_y: float = 1e-2) -> None:
        """
        :param bounds: List of (lower, upper) tuples for each parameter.
        :param boundary_margin: Fractional margin to define proximity to bounds (e.g., 0.05 = 5%)
        :param N: Number of best points required to be near boundary to trigger stop
        :param M: Number of best points required to be clustered to trigger stop
        :param top_count: Number of best points to consider
        :param eps_x: Distance threshold for x clustering
        :param eps_y: Distance threshold for y clustering
        """
        self.bounds = bounds
        self.boundary_margin = boundary_margin
        self.N = N
        self.M = M
        self.top_count = top_count
        self.eps_x = eps_x
        self.eps_y = eps_y
        assert self.top_count >= self.M and self.top_count >= self.N

    def should_stop(self, loop_state: LoopState) -> bool:
        if loop_state.iteration < max(self.N, self.M):
            return False

        X = loop_state.X
        Y = loop_state.Y
        best_indices = np.argsort(Y.flatten())[:self.top_count]
        best_X = X[best_indices]
        best_Y = Y[best_indices]

        # --- 1. Boundary Proximity Check ---
        boundary_hits = {}  # key = (dim, "lower"/"upper")

        for x in best_X[:self.top_count]:  # Only check top-N for boundary proximity
            for dim in range(x.shape[0]):
                low, high = self.bounds[dim]
                margin = self.boundary_margin * (high - low)

                if x[dim] <= low + margin:
                    boundary_hits[(dim, "lower")] = boundary_hits.get((dim, "lower"), 0) + 1

                if x[dim] >= high - margin:
                    boundary_hits[(dim, "upper")] = boundary_hits.get((dim, "upper"), 0) + 1
        
        for (dim, side), count in boundary_hits.items():
            if count >= self.N:
                print(f"Stopping due to {count} best points near {side} boundary of dimension {dim}.")
                _log.info(f"Stopping due to {count} best points near {side} boundary of dimension {dim}.")
                return True

        # --- 2. Clustering Check ---
        clustered_count = 0
        for i in range(self.top_count):
            for j in range(i + 1, self.top_count):
                dist_x = np.linalg.norm(best_X[i] - best_X[j])
                dist_y = np.abs(best_Y[i] - best_Y[j])
                if dist_x < self.eps_x and dist_y < self.eps_y:
                    clustered_count += 1

        max_cluster_pairs = self.M * (self.M - 1) // 2
        if clustered_count >= max_cluster_pairs // 2:
            print(f"Stopping due to {clustered_count} close pairs among top {self.M} points.")
            _log.info(f"Stopping due to {clustered_count} close pairs among top {self.M} points.")
            return True

        return False
    
    def get_expansion_suggestions(self, X: np.ndarray, Y: np.ndarray, secondary_threshold: float = 0.5) -> list[tuple[int, str]]:
        assert np.all((X >= 0) & (X <= 1)), "X must be normalized to [0, 1] range"
        if len(X) < self.top_count:
            return []
    
        best_indices = np.argsort(Y.flatten())[:self.top_count]
        best_X = X[best_indices]
    
        boundary_hits = {}
    
        for x in best_X:
            for dim in range(x.shape[0]):
                low, high = self.bounds[dim]
                margin = self.boundary_margin * (high - low)
                if x[dim] <= low + margin:
                    boundary_hits[(dim, "lower")] = boundary_hits.get((dim, "lower"), 0) + 1
                if x[dim] >= high - margin:
                    boundary_hits[(dim, "upper")] = boundary_hits.get((dim, "upper"), 0) + 1
                
        # First collect directly triggered expansions (meeting the N count threshold)
        triggered = [(dim, side) for (dim, side), count in boundary_hits.items() if count >= self.N]
    
        # Then check if any dimensions should be expanded proactively based on percentage
        proactively_triggered = []
        if triggered:
            for (dim, side), count in boundary_hits.items():
                if (dim, side) not in triggered:
                    fraction = count / self.top_count
                    if fraction >= secondary_threshold:
                        print(f"[get_expansion_suggestions] Proactively expanding dimension {dim} ({side}): {count} of {self.M} points ({fraction:.1%})")
                        proactively_triggered.append((dim, side))
    
        return triggered + proactively_triggered

class DynNorm:
    def __init__(self, X_initial=None, bounds=None):
        """
        Initializes normalizer with either a dataset X or explicit bounds.
        Args:
            X (np.ndarray): Data array of shape (n_samples, n_dimensions).
            bounds (list of tuples): [(low1, high1), (low2, high2), (low3, high3)]
        """
        if bounds is not None:
            self.min = np.array([b[0] for b in bounds])
            self.max = np.array([b[1] for b in bounds])
        elif X_initial is not None:
            self.min = np.min(X_initial, axis=0)
            self.max = np.max(X_initial, axis=0)
        else:
            raise ValueError("Either X or bounds must be provided.")
        
        if np.any(self.max - self.min == 0):
            raise ValueError("One or more dimensions have zero range.")

    def normalize(self, X):
        """
        Normalize input array to [0, 1] scale per dimension.
        Args:
            X (np.ndarray): Input of shape (n_samples, n_dimensions)
        Returns:
            np.ndarray: Normalized array, same shape
        """
        return (X - self.min) / (self.max - self.min)

    def denormalize(self, X_norm):
        """
        Denormalize from [0, 1] scale back to original scale.
        Args:
            X_norm (np.ndarray): Normalized input, shape (n_samples, n_dimensions)
        Returns:
            np.ndarray: Denormalized array, same shape
        """
        return self.min + X_norm * (self.max - self.min)

class YNorm:
    def __init__(self, Y_train):
        """
        Normalizes Y based on training data using standardization: (Y - mean) / std.
        Args:
            Y_train (np.ndarray): 1D or 2D array of shape (n_samples,) or (n_samples, 1)
        """
        self.mean = np.mean(Y_train)
        self.std = np.std(Y_train)
        if self.std == 0:
            raise ValueError("Standard deviation of Y is zero. Cannot normalize.")
    
    def normalize(self, Y):
        return (Y - self.mean) / self.std

    def denormalize(self, Y_norm):
        return Y_norm * self.std + self.mean

import time
import warnings
from collections import deque

def _clamp(value, limits):
    lower, upper = limits
    if value is None:
        return None
    elif upper is not None and value > upper:
        return upper
    elif lower is not None and value < lower:
        return lower
    return value

try:
    # get monotonic time to ensure that time deltas are always positive
    _current_time = time.monotonic
except AttributeError:
    # time.monotonic() not available (using python < 3.3), fallback to time.time()
    _current_time = time.time
    warnings.warn('time.monotonic() not available in python < 3.3, using time.time() as fallback')


class PID(object):
    """
    A simple PID controller. No fuss.
    """

    def __init__(self,
                 Kp=1.0, Ki=0.0, Kd=0.0,
                 setpoint=0,
                 sample_time=0.01,
                 meas_int=1,
                 output_limits=(None, None),
                 auto_mode=True,
                 proportional_on_measurement=False,
                 integral_timescale=0):
        """
        :param Kp: The value for the proportional gain Kp
        :param Ki: The value for the integral gain Ki
        :param Kd: The value for the derivative gain Kd
        :param setpoint: The initial setpoint that the PID will try to achieve
        :param sample_time: The time in seconds which the controller should wait before generating a new output value.
                            The PID works best when it is constantly called (eg. during a loop), but with a sample
                            time set so that the time difference between each update is (close to) constant. If set to
                            None, the PID will compute a new output value every time it is called.
        :param output_limits: The initial output limits to use, given as an iterable with 2 elements, for example:
                              (lower, upper). The output will never go below the lower limit or above the upper limit.
                              Either of the limits can also be set to None to have no limit in that direction. Setting
                              output limits also avoids integral windup, since the integral term will never be allowed
                              to grow outside of the limits.
        :param auto_mode: Whether the controller should be enabled (in auto mode) or not (in manual mode)
        :param proportional_on_measurement: Whether the proportional term should be calculated on the input directly
                                            rather than on the error (which is the traditional way). Using
                                            proportional-on-measurement avoids overshoot for some types of systems.
        """
        self.Kp, self.Ki, self.Kd = Kp, Ki, Kd
        self.setpoint = setpoint
        self.sample_time = sample_time

        self._min_output, self._max_output = output_limits
        self._auto_mode = auto_mode
        self.proportional_on_measurement = proportional_on_measurement
        
        self.integral_timescale = integral_timescale # Do in hours
        self.meas_int = meas_int # This will be in seconds
        
        # Declare list that will maintain a set timeframe of error terms to sum
        self.integral_terms = deque(maxlen=round(self.integral_timescale*60*60/self.meas_int))

        self.reset()

    def __call__(self, input_, dt=None):
        """
        Call the PID controller with *input_* and calculate and return a control output if sample_time seconds has
        passed since the last update. If no new output is calculated, return the previous output instead (or None if
        no value has been calculated yet).
        :param dt: If set, uses this value for timestep instead of real time. This can be used in simulations when
                   simulation time is different from real time.
        """
        if not self.auto_mode:
            return self._last_output

        now = _current_time()
        if dt is None:
            dt = now - self._last_time if now - self._last_time else 1e-16
        elif dt <= 0:
            raise ValueError("dt has nonpositive value {}. Must be positive.".format(dt))

        if self.sample_time is not None and dt < self.sample_time and self._last_output is not None:
            # only update every sample_time seconds
            return self._last_output

        # compute error terms
        error = self.setpoint - input_
        d_input = input_ - (self._last_input if self._last_input is not None else input_)

        # compute the proportional term
        if not self.proportional_on_measurement:
            # regular proportional-on-error, simply set the proportional term
            self._proportional = self.Kp * error
        else:
            # add the proportional error on measurement to error_sum
            self._proportional -= self.Kp * d_input

        # compute integral and derivative terms
        if self.integral_timescale == 0:
            self._integral += self.Ki * error * dt
            self._integral = _clamp(self._integral, self.output_limits)  # avoid integral windup
        # if interested in using a set timeframe for error integral, use this
        else:
            self.integral_terms.append(self.Ki * error * dt) # deque automatically discards oldest element when full
            self._integral = sum(self.integral_terms)
            self._integral = _clamp(self._integral, self.output_limits)  # avoid integral windup            

        self._derivative = -self.Kd * d_input / dt

        # compute final output
        output = self._proportional + self._integral + self._derivative
        output = _clamp(output, self.output_limits)

        # keep track of state
        self._last_output = output
        self._last_input = input_
        self._last_time = now

        return output

    @property
    def components(self):
        """
        The P-, I- and D-terms from the last computation as separate components as a tuple. Useful for visualizing
        what the controller is doing or when tuning hard-to-tune systems.
        """
        return self._proportional, self._integral, self._derivative

    @property
    def tunings(self):
        """The tunings used by the controller as a tuple: (Kp, Ki, Kd)"""
        return self.Kp, self.Ki, self.Kd

    @tunings.setter
    def tunings(self, tunings):
        """Setter for the PID tunings"""
        self.Kp, self.Ki, self.Kd = tunings

    @property
    def auto_mode(self):
        """Whether the controller is currently enabled (in auto mode) or not"""
        return self._auto_mode

    @auto_mode.setter
    def auto_mode(self, enabled):
        """Enable or disable the PID controller"""
        self.set_auto_mode(enabled)

    def set_auto_mode(self, enabled, last_output=None):
        """
        Enable or disable the PID controller, optionally setting the last output value.
        This is useful if some system has been manually controlled and if the PID should take over.
        In that case, pass the last output variable (the control variable) and it will be set as the starting
        I-term when the PID is set to auto mode.
        :param enabled: Whether auto mode should be enabled, True or False
        :param last_output: The last output, or the control variable, that the PID should start from
                            when going from manual mode to auto mode
        """
        if enabled and not self._auto_mode:
            # switching from manual mode to auto, reset
            self.reset()

            self._integral = (last_output if last_output is not None else 0)
            self._integral = _clamp(self._integral, self.output_limits)

        self._auto_mode = enabled

    @property
    def output_limits(self):
        """
        The current output limits as a 2-tuple: (lower, upper). See also the *output_limts* parameter in
        :meth:`PID.__init__`.
        """
        return self._min_output, self._max_output

    @output_limits.setter
    def output_limits(self, limits):
        """Setter for the output limits"""
        if limits is None:
            self._min_output, self._max_output = None, None
            return

        min_output, max_output = limits

        if None not in limits and max_output < min_output:
            raise ValueError('lower limit must be less than upper limit')

        self._min_output = min_output
        self._max_output = max_output

        self._integral = _clamp(self._integral, self.output_limits)
        self._last_output = _clamp(self._last_output, self.output_limits)

    def reset(self):
        """
        Reset the PID controller internals, setting each term to 0 as well as cleaning the integral,
        the last output and the last input (derivative calculation).
        """
        self._proportional = 0
        self._integral = 0
        self._derivative = 0

        self._last_time = _current_time()
        self._last_output = None
        self._last_input = None

# define Van de Vusse system model
# Possible to add more adjustable parameters to the input.
# def vdv_lin(x,t,u0,u1,Cai,Ti):
def vdv_lin(u0,u1,Cain_lin,Tin_lin,Ca_ss,Cb_ss,Tr_ss,Tk_ss):
    # Declare control state from u vector
    Vdot_Vr = u0      # Volumetric flowrate divided by reactor volume, 1/hr
    # Qkdot = u1      # Heat removal from reactor, kJ/hr
    
    # Declare states from fixed point
    # get_ss(Vdot_Vr, Qkdot, Cain, Tin)
    TrK_ss = Tr_ss + 273.15 # Reactor temperature, K
    # TkK_ss = Tk_ss + 273.15 # Coolant temperature, K    
    
    # Uncertainties taken from Chen, Kremling, and Allgower, 1995
    # These uncertainties match Hajaya and Shaqarin, 2019, which I am using heavily for the model
    # Note that A -> B and B -> C share the same reaction parameters, k1 and E1
    
    # Collision rates
    k10 = 1.287 * 10**12      # hr^-1, +- 0.04 before exponential
    k20 = 9.043 * 10**6      # m^3/(molA*hr), +- 0.27 before exponential
    ### 10^9 is for L/(molA*hr).
    
    # Activation energies
    E1 = -9758.3      # K
    E2 = -8560.0      # K
                    
    # Reaction rate constants
    k1 = k10*np.exp(E1/(TrK_ss))      # 
    k2 = k20*np.exp(E2/(TrK_ss))      # 
    
    # Enthalpy of reaction for all three reactions
    delH_Rab = 4.2      # kJ/molA, +- 2.36 
    delH_Rbc = -11.0      #kJ/molB, +- 1.92
    delH_Rad = -41.85      #kJ/molA, +- 1.41
    
    # Heat capacity of liquid phase in reactor
    Cp = 3.01      # kJ/(kg*K)
    
    # Density of liquid phase
    rho = 934.2      # kg/m3, +- 0.4
    
    # Heat capacity of coolant
    Cpk = 2.0      # kJ/(kg*K), +- 0.05
    
    # Mass of coolant
    mk = 5.0      # kg
    
    # Volume of reactor
    Vr = 0.01      # m3
    
    # Surface area of cooling jacket
    Ar = 0.215      # m2
    
    # Heat transfer coefficient for cooling jacket
    kw = 4032      # kJ/(h*m2*K), +- 120
    
    A = [[-Vdot_Vr - k1 - 2*k2*Ca_ss, 0, -k1*E1*Ca_ss/((TrK_ss)**2) - k2*E2*(Ca_ss**2)/((TrK_ss)**2), 0],
         [k1, -Vdot_Vr - k1, k1*E1*(Ca_ss-Cb_ss)/((TrK_ss)**2), 0],
         [-(1/(rho*Cp))*(k1*delH_Rab + 2*k2*delH_Rad*Ca_ss), -(1/(rho*Cp))*(k1*delH_Rbc), -Vdot_Vr - (1/(rho*Cp))*(k1*E1*Ca_ss*delH_Rab/((TrK_ss)**2) + k1*E1*Cb_ss*delH_Rbc/((TrK_ss)**2) + k2*E2*(Ca_ss**2)*delH_Rad/((TrK_ss)**2)) - (kw*Ar/(rho*Cp*Vr)), (kw*Ar/(rho*Cp*Vr))],
         [0, 0, (kw*Ar/(mk*Cpk)), -(kw*Ar/(mk*Cpk))]]
    
    B = [0, 0, 0, (1/(mk*Cpk))]
    
    C = [0, 1, 0, 0]
    
    D = [0]
    
    vdv_sys = ctl.StateSpace(A, B, C, D)
    # print(vdv_sys)
    
    return vdv_sys

# define Van de Vusse system model
# Possible to add more adjustable parameters to the input.
def vdv(x,t,u0,u1,Cai,Ti):
    # Declare states from x vector
    Ca = x[0]      # Concentration of A exiting the tank, mol/m3
    Cb = x[1]      # Concentration of B exiting the tank, mol/m3
    Tr = x[2]      # Reactor temperature, K
    Tk = x[3]      # Coolant temperature, K
    
    # Declare control state from u vector
    Vdot_Vr = u0      # Volumetric flowrate divided by reactor volume, 1/hr
    Qkdot = u1      # Heat removal from reactor, kJ/hr
    
    
    # Uncertainties taken from Chen, Kremling, and Allgower, 1995
    # These uncertainties match Hajaya and Shaqarin, 2019, which I am using heavily for the model
    # Note that A -> B and B -> C share the same reaction parameters, k1 and E1
    
    # Collision rates
    k10 = 1.287 * 10**12      # hr^-1, +- 0.04 before exponential
    k20 = 9.043 * 10**6      # m^3/(molA*hr), +- 0.27 before exponential
    ### 10^9 is for L/(molA*hr).
    
    # Activation energies
    E1 = -9758.3      # K
    E2 = -8560.0      # K
                    
    # Reaction rate constants
    k1 = k10*np.exp(E1/(Tr+273.15))      # 
    k2 = k20*np.exp(E2/(Tr+273.15))      # 
    
    # Enthalpy of reaction for all three reactions
    delH_Rab = 4.2      # kJ/molA, +- 2.36 
    delH_Rbc = -11.0      #kJ/molB, +- 1.92
    delH_Rad = -41.85      #kJ/molA, +- 1.41
    
    # Heat capacity of liquid phase in reactor
    Cp = 3.01      # kJ/(kg*K)
    
    # Density of liquid phase
    rho = 934.2      # kg/m3, +- 0.4
    
    # Heat capacity of coolant
    Cpk = 2.0      # kJ/(kg*K), +- 0.05
    
    # Mass of coolant
    mk = 5.0      # kg
    
    # Volume of reactor
    Vr = 0.01      # m3
    
    # Surface area of cooling jacket
    Ar = 0.215      # m2
    
    # Heat transfer coefficient for cooling jacket
    kw = 4032      # kJ/(h*m2*K), +- 120
    
    # Derivatives
    # Calculate derivative of Ca
    dCadt = (Vdot_Vr)*(Cai-Ca) - k1*Ca - k2*(Ca**2)
    
    # Calculate derivative of Cb
    dCbdt = -Vdot_Vr*Cb + k1*(Ca-Cb)
    
    # Calculate derivative of reactor temperature
    dTrdt = (Vdot_Vr)*(Ti-Tr) - (1/(rho*Cp))*((k1*Ca*delH_Rab) \
                        + k1*Cb*delH_Rbc + k2*(Ca**2)*delH_Rad) + (kw*Ar/(rho*Cp*Vr))*(Tk-Tr)
    
    # Calculate derivative of coolant temperature
    dTkdt = (1/(mk*Cpk))*(Qkdot + kw*Ar*(Tr - Tk))
    
    # Return xdot:
    xdot = np.zeros(4)
    xdot[0] = dCadt
    xdot[1] = dCbdt
    xdot[2] = dTrdt
    xdot[3] = dTkdt
    return xdot

def get_sys(Vdot_Vr_sys,Qk_sys,Cain_sys,Tin_sys):
    Ca_ss_sys, Cb_ss_sys, Tr_ss_sys, Tk_ss_sys = get_ss(Vdot_Vr_sys,Qk_sys,Cain_sys,Tin_sys)
    
    sys = vdv_lin(Vdot_Vr_sys, Qk_sys, Cain_sys, Tin_sys, Ca_ss_sys, Cb_ss_sys, Tr_ss_sys, Tk_ss_sys)

    zeros = ctl.zeros(sys)
    poles = ctl.poles(sys)

    tcz = 1/abs(zeros.real)
    tcp = 1/abs(poles.real)

    tcp.sort()

    tau1 = tcp[-1]
    tau2 = tcp[-2] + tcp[-3]/2

    gain = ctl.dcgain(sys) # * abs(poles.real) / abs(zeros.real)
    print('gain:', gain)
    
    theta = tcp[-3]/2 - sum(tcz)
    for ctr in range(len(tcp)-3):
        theta += tcp[ctr]
    # theta_check = tcp[-3]/2 + tcp[-4] - sum(tcz)
    
    taus = np.sqrt(tau1*tau2)
    if abs(0.2*taus) > abs(1.6*theta):
        tauc = 0.2*taus
    else:
        tauc = 1.6*theta
    Kci = (1/gain)*((tau1+tau2)/(theta+tauc))
    Kii = Kci / (tau1 + tau2)
    Kdi = Kci * (tau1*tau2/(tau1 + tau2))
    
    global Ca_ss_global, Cb_ss_global, Tr_ss_global, Tk_ss_global, VVr_global, Qk_global, Cain_global, Tin_global
    Ca_ss_global = Ca_ss_sys
    Cb_ss_global = Cb_ss_sys
    Tr_ss_global = Tr_ss_sys
    Tk_ss_global = Tk_ss_sys
    VVr_global = Vdot_Vr_sys
    Qk_global = Qk_sys
    Cain_global = Cain_sys
    Tin_global = Tin_sys
    
    print("Global variables upon declaration:")
    print("Cain_global:", Cain_global)
    print("Tin_global:", Tin_global)
    print("Ca_ss_global:", Ca_ss_global)
    print("Cb_ss_global:", Cb_ss_global)
    print("Tr_ss_global:", Tr_ss_global)
    print("Tk_ss_global:", Tk_ss_global)
    print("VVr_global:", VVr_global)
    print("Qk_global:", Qk_global)

    print('\nUsing tau1 and tau2, here are the suggeseted parameters from Skogestads powerpoint:')
    print('Kc:', (1/gain)*(tau1/(theta-tauc)))
    print('tauI:', (tau1 + tau2))
    print('tauD:', (tau1*tau2/(tau1 + tau2)))
    print('Ki:', Kii)
    print('Kd:', Kdi)
    
    return Kci, Kii, Kdi, Ca_ss_sys, Cb_ss_sys, Tr_ss_sys, Tk_ss_sys

def get_ss(Vdot_Vr, Qkdot, Cain, Tin):
    # Simulation time
    num_sec = 5
    num_hours = 2
    t = np.linspace(0,num_hours,int(num_hours*60*(60/num_sec)+1)) # Total of two hours of simulation time, sampling every 5 seconds

    # Steady State conditions
    Ca_ss = 0
    Cb_ss = 0
    Tr_ss = 0
    Tk_ss = 0

    x0 = np.empty(4)
    x0[0] = Ca_ss
    x0[1] = Cb_ss
    x0[2] = Tr_ss
    x0[3] = Tk_ss

    u0 = np.ones(len(t))*Vdot_Vr
    u1 = np.ones(len(t))*Qkdot
    # u1[round(1*60*60/num_sec):] = Qkdot + 500
    # u1[round(2*60*60/num_sec):] = Qkdot - 1000
    # u1[round(3*60*60/num_sec):] = Qkdot - 7000
    # sp0[round(1*60*60/num_sec):] = Cb_ss + 5
    
    print(set(u1))

    # Vectors for state variables
    Ca = np.zeros(len(t))
    Cb = np.zeros(len(t))
    Tr = np.zeros(len(t))
    Tk = np.zeros(len(t))

    # Manipulated variable conditions
    Cai = np.ones(len(t)) * Cain # mol/m3
    Ti = np.ones(len(t)) * Tin # Celsius

    # sp = np.ones(len(t)) * 1.09

    for i in range(len(t)-1):
        ts = [t[i],t[i+1]]
        y = odeint(vdv,x0,ts,args=(u0[i],u1[i],Cai[i],Ti[i]))
        Ca[i+1] = y[-1][0]
        Cb[i+1] = y[-1][1]
        Tr[i+1] = y[-1][2]
        Tk[i+1] = y[-1][3]
        x0[0] = Ca[i+1]
        x0[1] = Cb[i+1]
        x0[2] = Tr[i+1]
        x0[3] = Tk[i+1]

#         if i == 1000:
#             print('time:', t[i])
#             print('y[-1]:', y[-1])
#             print('y[-1][1]:', y[-1][1])
#             print('Cb:', Cb[i])
#             print('Cb[i+1]:', Cb[i+1])
#             print('y:', y)


    # plt.figure(figsize=(10,15))
    # plt.subplot(8,1,1)
    # plt.plot(t,Cai,'b',linewidth=3)
    # plt.ylabel('Ca initial (mol/m3)')
    # plt.legend(['Cai'],loc='best')

    # plt.subplot(8,1,2)
    # plt.plot(t,Ti,'r',linewidth=3)
    # plt.ylabel('Temperature initial (C)')
    # plt.legend(['Ti'],loc='best')

    # plt.subplot(8,1,3)
    # plt.plot(t,Ca,'b',linewidth=3)
    # plt.ylabel('Ca (mol/m3)')
    # plt.legend(['Reactor Ca'],loc='best')

    # plt.subplot(8,1,4)
    # plt.plot(t,Cb,'b',linewidth=3)
    # plt.ylabel('Cb (mol/m3)')
    # # plt.ylim([100, 1200])
    # plt.legend(['Reactor Cb'],loc='best')

    # plt.subplot(8,1,5)
    # plt.plot(t,Tr,'r',linewidth=3)
    # plt.ylabel('Reactor temperature (C))')
    # plt.legend(['Tr'],loc='best')

    # plt.subplot(8,1,6)
    # plt.plot(t,Tk,'r',linewidth=3)
    # plt.ylabel('Coolant temperature (C)')
    # plt.legend(['Tk'],loc='best')
    
    # plt.subplot(8,1,7)
    # plt.plot(t,u0,'g',linewidth=3)
    # plt.ylabel('VVr')
    # plt.legend(['VVr'],loc='best')

    # plt.subplot(8,1,8)
    # plt.plot(t,u1,'r',linewidth=3)
    # plt.ylabel('Heat removal rate')
    # plt.legend(['Qkdot'],loc='best')

    # plt.xlabel('Time (hr)')

    # plt.show()
    
    lastCa = Ca[-1]
    lastCb = Cb[-1]
    lastTr = Tr[-1]
    lastTk = Tk[-1]

#     print("last Ca:", Ca[-1])
#     print("last Cb:", Cb[-1])
#     print("last Tr:", Tr[-1])
#     print("last Tk:", Tk[-1])
    
    return(lastCa, lastCb, lastTr, lastTk)

def doublet_test_exp(Kp, Ki, Kd, Cai_dt, Ti_dt, Ca_dt, Cb_dt, Tr_dt, Tk_dt, Vdot_Vr_dt, Qkdot_dt):
    # Simulation time
    num_sec = 5
    num_hours = 4
    t = np.linspace(0,num_hours,round(num_hours*60*(60/num_sec)+1)) # Total of two hours of simulation time, sampling every 5 seconds

    # Steady State conditions
    Ca_ss = Ca_dt
    Cb_ss = Cb_dt
    Tr_ss = Tr_dt
    Tk_ss = Tk_dt
    
    print('Ca_ss:', Ca_ss)
    print('Cb_ss:', Cb_ss)
    print('Tr_ss:', Tr_ss)
    print('Tk_ss:', Tk_ss)

    Vdot_Vr = Vdot_Vr_dt # 1/hr
    Qkdot = Qkdot_dt # kJ/hr
    
    print('Vdot_Vr:', Vdot_Vr)
    print('Qkdot:', Qkdot)
    
    # Manipulated variable conditions
    Cai = np.ones(len(t)) * Cai_dt # mol/m3
    Ti = np.ones(len(t)) * Ti_dt # Celsius
    
    print('Ti[-1]:', Ti[-1])
    print('Cai[-1]:', Cai[-1])

    # Declare variables to be passed to function for evaluation at each timestep
    x0 = np.empty(4)
    x0[0] = Ca_ss
    x0[1] = Cb_ss
    x0[2] = Tr_ss
    x0[3] = Tk_ss
    
    x1 = np.empty(4)
    x1[0] = Ca_ss
    x1[1] = Cb_ss
    x1[2] = Tr_ss
    x1[3] = Tk_ss

    # Declare manipulated variables
    u0_0 = np.ones(len(t))*Vdot_Vr
    u1_0 = np.ones(len(t))*Qkdot
    
    u0_1 = np.ones(len(t))*Vdot_Vr
    u1_1 = np.ones(len(t))*Qkdot

    # Vectors for state variables
    Ca0 = np.ones(len(t)) * Ca_ss
    Cb0 = np.ones(len(t)) * Cb_ss
    Tr0 = np.ones(len(t)) * Tr_ss
    Tk0 = np.ones(len(t)) * Tk_ss
    
    Ca1 = np.ones(len(t)) * Ca_ss
    Cb1 = np.ones(len(t)) * Cb_ss
    Tr1 = np.ones(len(t)) * Tr_ss
    Tk1 = np.ones(len(t)) * Tk_ss

    # Declare setpionts for both doublet tests
    sp0 = np.ones(len(t)) * Cb_ss
    sp0[round(1*60*60/num_sec):] = Cb_ss + 5
    sp0[round(2*60*60/num_sec):] = Cb_ss - 5
    sp0[round(3*60*60/num_sec):] = Cb_ss
    
    sp1 = np.ones(len(t)) * Cb_ss
    sp1[round(1*60*60/num_sec):] = Cb_ss - 5
    sp1[round(2*60*60/num_sec):] = Cb_ss + 5
    sp1[round(3*60*60/num_sec):] = Cb_ss
    
    # Track error for testing purposes
    error0 = np.zeros(len(t))
    error1 = np.zeros(len(t))
    
    PID0_prop_err = np.zeros(len(t))
    PID0_int_err = np.zeros(len(t))
    PID0_der_err = np.zeros(len(t))
    
    PID1_prop_err = np.zeros(len(t))
    PID1_int_err = np.zeros(len(t))
    PID1_der_err = np.zeros(len(t))
    
    # Set up PID controllers
    Kpn = Kp
    Kin = Ki
    Kdn = Kd
    
    umin = -8500
    umax = 0
    u_bias = Qkdot
    # pid = PID(Kp,Ki,Kd,sample_time=None,output_limits=(umin,umax),meas_int=5,integral_timescale=1)
    pid0 = PID(Kpn,Kin,Kdn,sample_time=None,output_limits=(umin-u_bias,umax-u_bias),meas_int=5,integral_timescale=4)
    pid1 = PID(Kpn,Kin,Kdn,sample_time=None,output_limits=(umin-u_bias,umax-u_bias),meas_int=5,integral_timescale=4)

    for i in range(len(t)-1):
        # Update setpoint for each PID controller
        pid0.setpoint = sp0[i]
        pid1.setpoint = sp1[i]
        
        # Get next control action
        u1_0[i+1] = pid0(Cb0[i], t[i+1]-t[i]) + u_bias
        u1_1[i+1] = pid1(Cb1[i], t[i+1]-t[i]) + u_bias
        
        # Declare time interval
        ts = [t[i],t[i+1]]
        
        # Get next state based on control action and current state
        y0 = odeint(vdv,x0,ts,args=(u0_0[i],u1_0[i],Cai[i],Ti[i]))
        y1 = odeint(vdv,x1,ts,args=(u0_1[i],u1_1[i],Cai[i],Ti[i]))
        
        # Update vectors data storage
        Ca0[i+1] = y0[-1][0]
        Cb0[i+1] = y0[-1][1]
        Tr0[i+1] = y0[-1][2]
        Tk0[i+1] = y0[-1][3]
        x0[0] = Ca0[i+1]
        x0[1] = Cb0[i+1]
        x0[2] = Tr0[i+1]
        x0[3] = Tk0[i+1]
        
        Ca1[i+1] = y1[-1][0]
        Cb1[i+1] = y1[-1][1]
        Tr1[i+1] = y1[-1][2]
        Tk1[i+1] = y1[-1][3]
        x1[0] = Ca1[i+1]
        x1[1] = Cb1[i+1]
        x1[2] = Tr1[i+1]
        x1[3] = Tk1[i+1]
        
        # Track error
        error0[i] = sp0[i] - Cb0[i]
        error1[i] = sp1[i] - Cb1[i]
        
        PID0_prop_err[i] = pid0.components[0]
        PID0_int_err[i] = pid0.components[1]
        PID0_der_err[i] = pid0.components[2]
        PID1_prop_err[i] = pid1.components[0]
        PID1_int_err[i] = pid1.components[1]
        PID1_der_err[i] = pid1.components[2]

    plt.figure(figsize=(10,28))
    plt.subplot(14,1,1)
    plt.plot(t,Cai,'b',linewidth=3)
    plt.ylabel('Ca initial (mol/m3)')
    plt.legend(['Cai'],loc='best')

    plt.subplot(14,1,2)
    plt.plot(t,Ti,'r',linewidth=3)
    plt.ylabel('Temperature initial (C)')
    plt.legend(['Ti'],loc='best')

    plt.subplot(14,1,3)
    plt.plot(t,Ca0,'b',linewidth=3)
    plt.ylabel('Ca0 (mol/m3)')
    plt.legend(['Reactor 0 Ca'],loc='best')

    plt.subplot(14,1,4)
    plt.plot(t,Cb0,'b',linewidth=3)
    plt.plot(t,sp0,'k--',linewidth=3)
    plt.ylabel('Cb0 (mol/m3)')
    plt.legend(['Reactor 0 Cb, Setpoint'],loc='best')

    plt.subplot(14,1,5)
    plt.plot(t,Tr0,'r',linewidth=3)
    plt.ylabel('Reactor 0 temperature (C))')
    plt.legend(['Tr0'],loc='best')

    plt.subplot(14,1,6)
    plt.plot(t,Tk0,'r',linewidth=3)
    plt.ylabel('Coolant 0 temperature (C)')
    plt.legend(['Tk0'],loc='best')
    
    plt.subplot(14,1,7)
    plt.plot(t,Ca1,'b',linewidth=3)
    plt.ylabel('Ca1 (mol/m3)')
    plt.legend(['Reactor 1 Ca'],loc='best')

    plt.subplot(14,1,8)
    plt.plot(t,Cb1,'b',linewidth=3)
    plt.plot(t,sp1,'k--',linewidth=3)
    plt.ylabel('Cb1 (mol/m3)')
    plt.legend(['Reactor 1 Cb, Setpoint'],loc='best')

    plt.subplot(14,1,9)
    plt.plot(t,Tr1,'r',linewidth=3)
    plt.ylabel('Reactor 1 temperature (C))')
    plt.legend(['Tr1'],loc='best')

    plt.subplot(14,1,10)
    plt.plot(t,Tk1,'r',linewidth=3)
    plt.ylabel('Coolant 1 temperature (C)')
    plt.legend(['Tk1'],loc='best')
    
    plt.subplot(14,1,11)
    plt.plot(t,u1_0,'g',linewidth=3)
    plt.ylabel('Qkdot Reactor 0 (kJ/hr)')
    plt.legend(['Qkdot0'],loc='best')
    
    plt.subplot(14,1,12)
    plt.plot(t,u1_1,'g',linewidth=3)
    plt.ylabel('Qkdot Reactor 1 (kJ/hr)')
    plt.legend(['Qkdot1'],loc='best')
    
    plt.subplot(14,1,13)
    plt.plot(t,error0,'y',linewidth=3)
    plt.ylabel('Error reactor 0')
    plt.legend(['error0'],loc='best')
    
    plt.subplot(14,1,14)
    plt.plot(t,error1,'y',linewidth=3)
    plt.ylabel('Error reactor 1')
    plt.legend(['error1'],loc='best')
    plt.xlabel('Time (hr)')

    plt.show()
    
#     plt.figure(figsize=(10,14))
#     plt.subplot(6,1,1)
#     plt.plot(t,PID0_prop_err,'b',linewidth=3)
#     plt.ylabel('Error from Kp0')
#     plt.legend(['Kp_err_0'],loc='best')

#     plt.subplot(6,1,2)
#     plt.plot(t,PID0_int_err,'b',linewidth=3)
#     plt.ylabel('Error from Ki0')
#     plt.legend(['Ki_err_0'],loc='best')

#     plt.subplot(6,1,3)
#     plt.plot(t,PID0_der_err,'b',linewidth=3)
#     plt.ylabel('Error from Kd0')
#     plt.legend(['Kd_err_0'],loc='best')

#     plt.subplot(6,1,4)
#     plt.plot(t,PID1_prop_err,'b',linewidth=3)
#     plt.ylabel('Error from Kp1')
#     plt.legend(['Kp_err_1'],loc='best')
    
#     plt.subplot(6,1,5)
#     plt.plot(t,PID1_int_err,'b',linewidth=3)
#     plt.ylabel('Error from Ki1')
#     plt.legend(['Ki_err_1'],loc='best')
    
#     plt.subplot(6,1,6)
#     plt.plot(t,PID1_der_err,'b',linewidth=3)
#     plt.ylabel('Error from Kd1')
#     plt.legend(['Kd_err_1'],loc='best')
#     plt.xlabel('Time (hr)')
    
    print('sum error0:', sum(error0))
    print('sum error1:', sum(error1))
    print('sum abs error0:', sum(abs(error0)))
    print('sum abs error1:', sum(abs(error1)))
    print('total error from doublet_test_exp:', sum(abs(error0)) + sum(abs(error1)))
    total_error = np.array([sum(abs(error0)) + sum(abs(error1))])
    print('last line from doublet_test:', total_error)

#     print("last Ca:", Ca[-1])
#     print("last Cb:", Cb[-1])
#     print("last Tr:", Tr[-1])
#     print("last Tk:", Tk[-1])

def doublet_test(cont_par):
    # Simulation time
    num_sec = 5
    num_hours = 4
    t = np.linspace(0,num_hours,round(num_hours*60*(60/num_sec)+1)) # Total of two hours of simulation time, sampling every 5 seconds
    
    # Steady State conditions
    Ca_ss = Ca_ss_global
    Cb_ss = Cb_ss_global
    Tr_ss = Tr_ss_global
    Tk_ss = Tk_ss_global
    Vdot_Vr = VVr_global # 1/hr
    Qkdot = Qk_global # kJ/hr
    
    # Manipulated variable conditions
    Cai = np.ones(len(t)) * Cain_global # mol/m3
    Ti = np.ones(len(t)) * Tin_global # Celsius

    # Declare variables to be passed to function for evaluation at each timestep
    x0 = np.empty(4)
    x0[0] = Ca_ss
    x0[1] = Cb_ss
    x0[2] = Tr_ss
    x0[3] = Tk_ss
    
    x1 = np.empty(4)
    x1[0] = Ca_ss
    x1[1] = Cb_ss
    x1[2] = Tr_ss
    x1[3] = Tk_ss

    # Declare manipulated variables
    u0_0 = np.ones(len(t))*Vdot_Vr
    u1_0 = np.ones(len(t))*Qkdot
    
    u0_1 = np.ones(len(t))*Vdot_Vr
    u1_1 = np.ones(len(t))*Qkdot

    # Vectors for state variables
    Ca0 = np.ones(len(t)) * Ca_ss
    Cb0 = np.ones(len(t)) * Cb_ss
    Tr0 = np.ones(len(t)) * Tr_ss
    Tk0 = np.ones(len(t)) * Tk_ss
    
    Ca1 = np.ones(len(t)) * Ca_ss
    Cb1 = np.ones(len(t)) * Cb_ss
    Tr1 = np.ones(len(t)) * Tr_ss
    Tk1 = np.ones(len(t)) * Tk_ss

    # Declare setpionts for both doublet tests
    sp0 = np.ones(len(t)) * Cb_ss
    sp0[round(1*60*60/num_sec):] = Cb_ss + 5
    sp0[round(2*60*60/num_sec):] = Cb_ss - 5
    sp0[round(3*60*60/num_sec):] = Cb_ss
    
    sp1 = np.ones(len(t)) * Cb_ss
    sp1[round(1*60*60/num_sec):] = Cb_ss - 5
    sp1[round(2*60*60/num_sec):] = Cb_ss + 5
    sp1[round(3*60*60/num_sec):] = Cb_ss
    
    # Track error for testing purposes
    error0 = np.zeros(len(t))
    error1 = np.zeros(len(t))
    
#     PID0_prop_err = np.zeros(len(t))
#     PID0_int_err = np.zeros(len(t))
#     PID0_der_err = np.zeros(len(t))
    
#     PID1_prop_err = np.zeros(len(t))
#     PID1_int_err = np.zeros(len(t))
#     PID1_der_err = np.zeros(len(t))
    
    # Set up PID controllers
    Kp = cont_par[0][0]
    Ki = cont_par[0][1]
    Kd = cont_par[0][2]
    
    umin = -8500
    umax = 0
    u_bias = Qkdot
    # pid = PID(Kp,Ki,Kd,sample_time=None,output_limits=(umin,umax),meas_int=5,integral_timescale=1)
    pid0 = PID(Kp,Ki,Kd,sample_time=None,output_limits=(umin-u_bias,umax-u_bias),meas_int=5,integral_timescale=4)
    pid1 = PID(Kp,Ki,Kd,sample_time=None,output_limits=(umin-u_bias,umax-u_bias),meas_int=5,integral_timescale=4)

    for i in range(len(t)-1):
        # Update setpoint for each PID controller
        pid0.setpoint = sp0[i]
        pid1.setpoint = sp1[i]
        
        # Get next control action
        u1_0[i+1] = pid0(Cb0[i], t[i+1]-t[i]) + u_bias
        u1_1[i+1] = pid1(Cb1[i], t[i+1]-t[i]) + u_bias
        
        # Declare time interval
        ts = [t[i],t[i+1]]
        
        # Get next state based on control action and current state
        y0 = odeint(vdv,x0,ts,args=(u0_0[i],u1_0[i],Cai[i],Ti[i]))
        y1 = odeint(vdv,x1,ts,args=(u0_1[i],u1_1[i],Cai[i],Ti[i]))
        
        # Update vectors data storage
        Ca0[i+1] = y0[-1][0]
        Cb0[i+1] = y0[-1][1]
        Tr0[i+1] = y0[-1][2]
        Tk0[i+1] = y0[-1][3]
        x0[0] = Ca0[i+1]
        x0[1] = Cb0[i+1]
        x0[2] = Tr0[i+1]
        x0[3] = Tk0[i+1]
        
        Ca1[i+1] = y1[-1][0]
        Cb1[i+1] = y1[-1][1]
        Tr1[i+1] = y1[-1][2]
        Tk1[i+1] = y1[-1][3]
        x1[0] = Ca1[i+1]
        x1[1] = Cb1[i+1]
        x1[2] = Tr1[i+1]
        x1[3] = Tk1[i+1]
        
        # Track error
        error0[i] = sp0[i] - Cb0[i]
        error1[i] = sp1[i] - Cb1[i]
        
#     plt.figure(figsize=(10,28))
#     plt.subplot(14,1,1)
#     plt.plot(t,Cai,'b',linewidth=3)
#     plt.ylabel('Ca initial (mol/m3)')
#     plt.legend(['Cai'],loc='best')

#     plt.subplot(14,1,2)
#     plt.plot(t,Ti,'r',linewidth=3)
#     plt.ylabel('Temperature initial (C)')
#     plt.legend(['Ti'],loc='best')

#     plt.subplot(14,1,3)
#     plt.plot(t,Ca0,'b',linewidth=3)
#     plt.ylabel('Ca0 (mol/m3)')
#     plt.legend(['Reactor 0 Ca'],loc='best')

#     plt.subplot(14,1,4)
#     plt.plot(t,Cb0,'b',linewidth=3)
#     plt.plot(t,sp0,'k--',linewidth=3)
#     plt.ylabel('Cb0 (mol/m3)')
#     plt.legend(['Reactor 0 Cb, Setpoint'],loc='best')

#     plt.subplot(14,1,5)
#     plt.plot(t,Tr0,'r',linewidth=3)
#     plt.ylabel('Reactor 0 temperature (C))')
#     plt.legend(['Tr0'],loc='best')

#     plt.subplot(14,1,6)
#     plt.plot(t,Tk0,'r',linewidth=3)
#     plt.ylabel('Coolant 0 temperature (C)')
#     plt.legend(['Tk0'],loc='best')
    
#     plt.subplot(14,1,7)
#     plt.plot(t,Ca1,'b',linewidth=3)
#     plt.ylabel('Ca1 (mol/m3)')
#     plt.legend(['Reactor 1 Ca'],loc='best')

#     plt.subplot(14,1,8)
#     plt.plot(t,Cb1,'b',linewidth=3)
#     plt.plot(t,sp1,'k--',linewidth=3)
#     plt.ylabel('Cb1 (mol/m3)')
#     plt.legend(['Reactor 1 Cb, Setpoint'],loc='best')

#     plt.subplot(14,1,9)
#     plt.plot(t,Tr1,'r',linewidth=3)
#     plt.ylabel('Reactor 1 temperature (C))')
#     plt.legend(['Tr1'],loc='best')

#     plt.subplot(14,1,10)
#     plt.plot(t,Tk1,'r',linewidth=3)
#     plt.ylabel('Coolant 1 temperature (C)')
#     plt.legend(['Tk1'],loc='best')
    
#     plt.subplot(14,1,11)
#     plt.plot(t,u1_0,'g',linewidth=3)
#     plt.ylabel('Qkdot Reactor 0 (kJ/hr)')
#     plt.legend(['Qkdot0'],loc='best')
    
#     plt.subplot(14,1,12)
#     plt.plot(t,u1_1,'g',linewidth=3)
#     plt.ylabel('Qkdot Reactor 1 (kJ/hr)')
#     plt.legend(['Qkdot1'],loc='best')
    
#     plt.subplot(14,1,13)
#     plt.plot(t,error0,'y',linewidth=3)
#     plt.ylabel('Error reactor 0')
#     plt.legend(['error0'],loc='best')
    
#     plt.subplot(14,1,14)
#     plt.plot(t,error1,'y',linewidth=3)
#     plt.ylabel('Error reactor 1')
#     plt.legend(['error1'],loc='best')
#     plt.xlabel('Time (hr)')

#     plt.show()
        
    total_error = np.array([sum(abs(error0)) + sum(abs(error1))])
#     print('Time for measuring intervals:', time.ctime())
#     print('total error from doublet_test:', total_error)
    # print(gpy_model)
    
    # print('total error from doublet_test_exp:', total_error)
    return total_error[:, None]

# def oom(number):
#     if number == 0:
#         return 0
#     return math.floor(math.log10(abs(number)))


def get_largest_dec(number):
    num_str = str(format(abs(number), '0.3E'))
    return int(num_str[0])

def round_to_sf(x, n):
    if x == 0:
        return 0
    else:
        return float(f'{x:.{n}g}')
    
def expand_bounds(old_bounds, bounds_to_expand, expansion_fraction=0.25, contraction_fraction=0.1,
                  max_width=None, best_point=None):
    """
    Expand bounds where suggested, and contract opposing sides if only one side is being expanded.
    Optionally clip to a per-dimension maximum width, centered on best_point if given.
    
    :param old_bounds: List of [low, high] bounds for each parameter.
    :param bounds_to_expand: List of (dim, "lower"/"upper") tuples.
    :param expansion_fraction: Fraction to expand sides by (relative to current width).
    :param contraction_fraction: Fraction to contract opposite side by.
    :param max_width: Either scalar or list/array of per-dimension max widths.
    :param best_point: Optional np.ndarray of best parameter values.
    :return: List of new bounds per dimension.
    """
    new_bounds = []
    dim_count = len(old_bounds)

    # Normalize max_width input
    if max_width is None:
        max_widths = [float('inf')] * dim_count  # No clipping
    elif isinstance(max_width, (int, float)):
        max_widths = [max_width] * dim_count
    else:
        max_widths = list(max_width)

    # Organize expansion requests
    expand_sides = {dim: set() for dim in range(dim_count)}
    for dim, side in bounds_to_expand:
        expand_sides[dim].add(side)

    for i, (low, high) in enumerate(old_bounds):
        width = high - low
        new_low, new_high = low, high

        expand_lower = "lower" in expand_sides[i]
        expand_upper = "upper" in expand_sides[i]

        # Expand requested sides
        if expand_lower:
            new_low = low - expansion_fraction * width
        if expand_upper:
            new_high = high + expansion_fraction * width

        # Contract opposing side if only one side is expanded
        if expand_lower and not expand_upper:
            new_high = max(new_low + 1e-6, high - contraction_fraction * width)
        elif expand_upper and not expand_lower:
            new_low = min(new_high - 1e-6, low + contraction_fraction * width)

        # Enforce per-dimension max width
        allowed_width = max_widths[i]
        if new_high - new_low > allowed_width:
            center = best_point[i] if best_point is not None else (new_high + new_low) / 2
            new_low = center - allowed_width / 2
            new_high = center + allowed_width / 2

        new_bounds.append([new_low, new_high])

    return new_bounds

from itertools import product
def get_delta_LHS(old_bounds, new_bounds, n_samples):
    dim = len(old_bounds)

    # Create ranges: 3 options per dimension (low expansion, old range, high expansion)
    axis_ranges = []
    for i in range(dim):
        low_o, high_o = old_bounds[i]
        low_n, high_n = new_bounds[i]

        ranges = []
        if low_n < low_o:
            ranges.append((low_n, low_o))  # lower expansion
        ranges.append((max(low_n, low_o), min(high_n, high_o)))  # overlapping old region
        if high_n > high_o:
            ranges.append((high_o, high_n))  # upper expansion

        axis_ranges.append(ranges)

    # Create all combinations of slabs
    all_boxes = list(product(*axis_ranges))
    
    # Remove boxes that fall entirely within the old bounds (i.e., center box)
    delta_slabs = []
    for box in all_boxes:
        box_in_old = all(
            old_bounds[d][0] <= box[d][0] and box[d][1] <= old_bounds[d][1]
            for d in range(dim)
        )
        if not box_in_old:
            delta_slabs.append(box)

    if len(delta_slabs) == 0:
        print("[get_delta_LHS] No expanded slabs found. Returning empty array.")
        return np.empty((0, dim))

    # Compute volume of each slab
    volumes = [np.prod([hi - lo for (lo, hi) in slab]) for slab in delta_slabs]
    total_volume = sum(volumes)

    # Determine number of samples per slab
    samples_per_slab = [int(round(n_samples * (v / total_volume))) for v in volumes]

    # Adjust if rounding causes mismatch
    total_allocated = sum(samples_per_slab)
    diff = n_samples - total_allocated
    if diff != 0:
        # Adjust by adding/removing from the largest slabs
        sorted_indices = np.argsort([-v for v in volumes])
        for i in range(abs(diff)):
            idx = sorted_indices[i % len(delta_slabs)]
            samples_per_slab[idx] += 1 if diff > 0 else -1

    # Sample each slab
    samples = []
    for slab, n in zip(delta_slabs, samples_per_slab):
        if n <= 0:
            continue
        sampler = qmc.LatinHypercube(d=dim)
        X = sampler.random(n)
        scaled_X = np.array([
            lo + (hi - lo) * X[:, i]
            for i, (lo, hi) in enumerate(slab)
        ]).T
        # scaled_X = qmc.scale(X, [lo for (lo, hi) in slab], [hi for (lo, hi) in slab])
        samples.append(scaled_X)

    return np.vstack(samples) if samples else np.empty((0, dim))

def get_zoomed_bounds(center, old_bounds, zoom_fraction=0.2, min_widths=None):
    """
    Contract bounds around a center point (best solution) for final refinement.
    """
    zoom_bounds = []
    # This is for debugging - can be removed later
    if not old_bounds or len(old_bounds) == 0:
        raise ValueError("Error in get_zoomed_bounds: old_bounds is empty or None.")
    for i, (low, high) in enumerate(old_bounds):
        full_width = high - low
        zoom_width = zoom_fraction * full_width

        if min_widths is not None:
            zoom_width = max(zoom_width, min_widths[i])

        new_low = center[i] - zoom_width / 2
        new_high = center[i] + zoom_width / 2
        zoom_bounds.append([new_low, new_high])

    return zoom_bounds

def reshape_for_init(x):
    return doublet_test(x.reshape(1, -1))

def remove_duplicates(X, Y, dec=8):
    _, idx = np.unique(np.round(X, decimals=dec), axis=0, return_index=True)
    return X[idx], Y[idx]

def do_GP(Vdot_Vr_1, Qk_1, Cain_1, Tin_1):
    print('Most recent time:', time.ctime())
    Kc_b, Ki_b, Kd_b, Ca_ss_GP, Cb_ss_GP, Tr_ss_GP, Tk_ss_GP = get_sys(Vdot_Vr_1, Qk_1, Cain_1, Tin_1)
    Kcb_str = str(Kc_b) + ','
    Kib_str = str(Ki_b) + ','
    Kdb_str = str(Kd_b) + ','
    
    if np.sign(Kc_b) == -1:
        Kc_lb, Kc_ub = (Kc_b - 2), 2

    elif np.sign(Kc_b) == 1:
        Kc_lb, Kc_ub = -2, Kc_b + 2

    else:
        Kc_lb, Kc_ub = -10, 10

    if np.sign(Ki_b) == -1:
        Ki_lb, Ki_ub = (Ki_b - 2), 2

    elif np.sign(Ki_b) == 1:
        Ki_lb, Ki_ub = -2, Ki_b + 2

    else:
        Ki_lb, Ki_ub = -10, 10

    if np.sign(Kd_b) == -1:
        Kd_lb, Kd_ub = (Kd_b - 2), 2

    elif np.sign(Kd_b) == 1:
        Kd_lb, Kd_ub = -2, Kd_b + 2

    else:
        Kd_lb, Kd_ub = -10, 10
        
    raw_bounds = [[Kc_lb, Kc_ub], [Ki_lb, Ki_ub], [Kd_lb, Kd_ub]]
    KcPar = ContinuousParameter("Kc", 0.0, 1.0)
    KiPar = ContinuousParameter("Ki", 0.0, 1.0)
    KdPar = ContinuousParameter("Kd", 0.0, 1.0)
    
    print("Kc Bounds:", Kc_lb, Kc_ub)
    print("Ki Bounds:", Ki_lb, Ki_ub)
    print("Kd Bounds:", Kd_lb, Kd_ub)

    # Number of input dimensions and of initial points
    n_d = 3
    n_init = 200

    # Declare boundaries and initialize three random points for GP to start
    newbounds = ParameterSpace([KcPar, KiPar, KdPar])
    
    # Initialize LHS sampler
    sampler = qmc.LatinHypercube(d=n_d)
    
    # Generate samples
    init_samples = sampler.random(n_init)
    
    min_vals = [Kc_lb, Ki_lb, Kd_lb] # Minimum values for each dimension
    max_vals = [Kc_ub, Ki_ub, Kd_ub] # Maximum values for each dimension
    
    # Scale LHS samples to desired range
    X_init_1 = np.zeros_like(init_samples)
    for i in range(n_d):
        X_init_1[:, i] = min_vals[i] + (max_vals[i] - min_vals[i]) * init_samples[:, i]

    extrema_samples = []
    for i in range(2):  # For Kc
        for j in range(2):  # For Ki
            for k in range(2): # For Kd
                sample = np.array([min_vals[0] if i == 0 else max_vals[0],
                                   min_vals[1] if j == 0 else max_vals[1],
                                   min_vals[2] if k == 0 else max_vals[2]])
                extrema_samples.append(sample)
               
    # Append median extrema points
    # Medians
    Kc_med = (min_vals[0] + max_vals[0]) / 2
    Ki_med = (min_vals[1] + max_vals[1]) / 2
    Kd_med = (min_vals[2] + max_vals[2]) / 2
   
    # Median Kc
    for i in range(2):
        for j in range(2):
            extrema_samples.append(np.array([Kc_med,
                min_vals[1] if i == 0 else max_vals[1],
                min_vals[2] if j == 0 else max_vals[2]]))
    # Median Ki
    for i in range(2):
        for j in range(2):
            extrema_samples.append(np.array(
                [min_vals[0] if i == 0 else max_vals[0],
                    Ki_med,
                    min_vals[2] if j == 0 else max_vals[2]]))
    # Median Kd
    for i in range(2):
        for j in range(2):
            extrema_samples.append(np.array(
                [min_vals[0] if i == 0 else max_vals[0],
                    min_vals[1] if j == 0 else max_vals[1],
                    Kd_med]))
   
    esarr = np.array(extrema_samples)
   
    X_init = np.vstack([X_init_1, esarr])

    # print('Get initial samples start time:', time.ctime())
    Y_init = np.empty(len(X_init))
    for rc in range(len(X_init)):
        Y_init[rc] = doublet_test(X_init[rc].reshape(1, n_d))
            
    Y_init = Y_init.reshape(-1, 1)
    
    assert X_init.shape[0] == Y_init.shape[0], "Mismatch in sample/label count"
    
    normalizer = DynNorm(bounds=raw_bounds)
    X_norm_pre = normalizer.normalize(X_init)
    
    Y_normalizer = YNorm(Y_init)
    Y_norm_pre = Y_normalizer.normalize(Y_init)    

    X_norm, Y_norm = remove_duplicates(X_norm_pre, Y_norm_pre, dec=8)

    # Declare GPy kernel, create GPy model based on initial points, wrap emukit_model around.
    k = GPy.kern.Exponential(input_dim=n_d, ARD=True) # Make sure it takes the number of input dimensions
    gpy_model = GPy.models.GPRegression(X_norm, Y_norm, k) # Give it initial conditions
    # gpy_model.Gaussian_noise = (np.max(Y_init)-np.min(Y_init))/100
    #  gpy_model.Gaussian_noise.variance.fix()
    gpy_model.Gaussian_noise.variance.constrain_bounded(1e-8, 1e-4)
    # gpy_model.kern.lengthscale.constrain_bounded(1e-5, 10)
    # gpy_model.optimize_restarts(10, robust=True)
    gpy_model.optimize()
    
    print(gpy_model)
    # Your dimension names
    param_names = ['Kp', 'Ki', 'Kd']
    
    # Access the ARD lengthscales
    lengthscales = gpy_model.kern.lengthscale.values
    
    # Print each one with its label
    for name, l in zip(param_names, lengthscales):
        print(f"Lengthscale for {name}: {l:.6f}")
    
    emukit_model = GPyModelWrapper(gpy_model) # Get emukit model
    myBopt = BayesianOptimizationLoop(space=newbounds, model=emukit_model)
    
    sc1_bounds = np.array([[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]])
    stop_cond1 = Bound_or_Cluster_Stop(bounds=sc1_bounds, boundary_margin=0.10, N=8, M=10,
                                       top_count = 10)
    stop_cond2 = FixedIterationsStoppingCondition(500)

    def evaluate(x_norm):
        x_real = normalizer.denormalize(np.array(x_norm).reshape(1, -1))
        y_real = doublet_test(x_real)
        y = Y_normalizer.normalize(y_real)
        return y
        
    myBopt.run_loop(user_function=evaluate, stopping_condition=stop_cond1 | stop_cond2)
    
    X_final_norm = myBopt.loop_state.X
    X_final = normalizer.denormalize(X_final_norm)
    Y_final_norm = myBopt.loop_state.Y
    Y_final = Y_normalizer.denormalize(Y_final_norm)
    
    min_error_BO1 = Y_final[np.argmin(Y_final)]
    min_1_loc_in_list = np.argmin(Y_final)
    pid_BO1 = X_final[np.argmin(Y_final)]
    bounds_to_exp = stop_cond1.get_expansion_suggestions(X_final_norm, Y_final,
                    secondary_threshold = 0.5)
    
    while_counter = 0
    exp_frac = 0.50
    cont_frac = 0.30
    max_width = [100, 300, 10]
    
    # # Concatenate X and Y horizontally
    # data_for_testing = np.hstack((X_final, X_final_norm, Y_final))
    
    # # Create column names
    # num_x = X_final.shape[1]
    # column_names = [f"x{i}" for i in range(num_x)] + [f"x_norm{i}" for i in range(num_x)] + ["y"]
    
    # # Create DataFrame and save to CSV
    # df = pd.DataFrame(data_for_testing, columns=column_names)
    # df.to_csv("Documents/bo_pid/two_tanks_new_test/bo_first_pass_results.csv", index=False)
    
    while bounds_to_exp:
        while_counter += 1
        print('Most recent time:', time.ctime())
        print(f"\nIteration {while_counter}: Expanding due to recommended boundaries: {bounds_to_exp}")
        print('while counter:', while_counter)
        print('X length:', X_final.shape[0])
        print('Y length:', Y_final.shape[0])
        new_bounds_array = expand_bounds(raw_bounds, bounds_to_exp, exp_frac, cont_frac, max_width, pid_BO1)
        min_error_BO1a, pid_BO1a, min_err_loc, varBO1a, lsBO1kpa, lsBO1kia, lsBO1kda, gnBO1a, bounds_to_exp,\
            X_final_2, Y_final_2 = do_GP2(raw_bounds, new_bounds_array, X_final, Y_final)
        X_final = X_final_2
        Y_final = Y_final_2
        raw_bounds = new_bounds_array
        if min_error_BO1a < min_error_BO1:
            print('\nActivated if to reassign pid_BO1 and min_error\n')
            pid_BO1 = pid_BO1a
            min_error_BO1 = min_error_BO1a
            min_1_loc_in_list = min_err_loc
            ##### I think we don't even need the next 5 lines
            print(f"New bounds: {new_bounds_array}")

    pid_BO2, min_error_BO2, X_final_z, Y_final_z, zoom_frac, zoom_min_widths,\
        var_gp2_str, ls_gp2_kp_str, ls_gp2_ki_str, ls_gp2_kd_str, gn_gp2_str, len_BO_2_str,\
            rec_zoom_bounds = do_zoomed_BO(pid_BO1, raw_bounds)
    
    final_wc = 0
    while rec_zoom_bounds:
        final_wc += 1
        new_zbo_bounds = rec_zoom_bounds
        pid_BO2, min_error_BO2, X_final_z, Y_final_z, zoom_frac, zoom_min_widths,\
            var_gp2_str, ls_gp2_kp_str, ls_gp2_ki_str, ls_gp2_kd_str, gn_gp2_str, len_BO_2_str,\
                rec_zoom_bounds = do_zoomed_BO(new_zbo_bounds, pid_BO1)
                
    print('length of Y vector:', len(Y_final_z))
    print('Minimum error obtained:', min(Y_final_z))
    print('Parameters which produced them:', X_final_z[np.argmin(Y_final_z)])
    min_2_loc_in_list = np.argmin(Y_final_z)
    
    if min_error_BO1 < min_error_BO2:
        print("BO1 was better")
        Kcbest_str = str(round(pid_BO1[0], 5)) + ','
        Kibest_str = str(round(pid_BO1[1], 5)) + ','
        Kdbest_str = str(round(pid_BO1[2], 5)) + ','
        tebest_str = str(round(min_error_BO1[0], 5)) + ','
        bo_id_str = '1,' # Still has a comma
        min_iter_str = str(min_1_loc_in_list) # Last element - doesn't need comma

    else:
        print("BO2 was better")
        Kcbest_str = str(round(pid_BO2[0], 5)) + ','
        Kibest_str = str(round(pid_BO2[1], 5)) + ','
        Kdbest_str = str(round(pid_BO2[2], 5)) + ','
        tebest_str = str(round(min_error_BO2[0], 5)) + ','
        bo_id_str = '2,' # Still has a comma
        min_iter_str = str(min_2_loc_in_list) # Last element - doesn't need comma
    
    Kc2_str = str(round(X_final_z[np.argmin(Y_final_z)][0], 5)) + ','
    Ki2_str = str(round(X_final_z[np.argmin(Y_final_z)][1], 5)) + ','
    Kd2_str = str(round(X_final_z[np.argmin(Y_final_z)][2], 5)) + ','
    te2_str = str(round(Y_final_z[np.argmin(Y_final_z)][0], 5)) + ','

    Kc1_str = str(round(pid_BO1[0], 5)) + ','
    Ki1_str = str(round(pid_BO1[1], 5)) + ','
    Kd1_str = str(round(pid_BO1[2], 5)) + ','
    te1_str = str(round(min_error_BO1[0], 5)) + ','
    
    Vdot_Vr_str = str(Vdot_Vr_1) + ','
    Qk_str = str(Qk_1) + ','
    Cain_str = str(Cain_1) + ','
    Tin_str = str(Tin_1) + ','
    
    Ca_ss_str = str(Ca_ss_GP) + ','
    Cb_ss_str = str(Cb_ss_GP) + ','
    Tr_ss_str = str(Tr_ss_GP) + ','
    Tk_ss_str = str(Tk_ss_GP) + ','
    
    k2_str = 'Exponential,'
    k1_str = 'Exponential,'
    var_gp1_str = str(gpy_model.kern.variance.values[0]) + ','
    ls_gp1_kp_str = str(round(gpy_model.kern.lengthscale.values[0], 7)) + ','
    ls_gp1_ki_str = str(round(gpy_model.kern.lengthscale.values[1], 7)) + ','
    ls_gp1_kd_str = str(round(gpy_model.kern.lengthscale.values[2], 7)) + ','
    gn_gp1_str = str(gpy_model.likelihood.variance.values[0]) + ','
    len_BO_1_str = str(len(myBopt.loop_state.Y)) + ','
    
    wc_str = str(while_counter) + ','
    final_wc_str = str(final_wc) + ','
    exp_frac_str = str(exp_frac) + ','
    cont_frac_str = str(cont_frac) + ','
    max_width_kp_str = str(max_width[0]) + ','
    max_width_ki_str = str(max_width[1]) + ','
    max_width_kd_str = str(max_width[2]) + ','
    
    zoom_frac_str = str(zoom_frac) + ','
    zoom_min_width_kp_str = str(zoom_min_widths[0]) + ','
    zoom_min_width_ki_str = str(zoom_min_widths[1]) + ','
    zoom_min_width_kd_str = str(zoom_min_widths[2]) + ','
    
    str_to_print = Kcbest_str + Kibest_str + Kdbest_str + tebest_str +\
        bo_id_str + Kc2_str + Ki2_str + Kd2_str + te2_str + Kc1_str + Ki1_str +\
        Kd1_str + te1_str + Vdot_Vr_str + Qk_str + Cain_str + Tin_str + Ca_ss_str + Cb_ss_str + Tr_ss_str + Tk_ss_str +\
        k2_str + var_gp2_str + ls_gp2_kp_str + ls_gp2_ki_str + ls_gp2_kd_str + gn_gp2_str +\
        len_BO_2_str + k1_str + var_gp1_str + ls_gp1_kp_str + ls_gp1_ki_str + ls_gp1_kd_str + gn_gp1_str +\
        len_BO_1_str + wc_str + exp_frac_str + cont_frac_str + max_width_kp_str +\
        max_width_ki_str + max_width_kd_str + zoom_frac_str + final_wc_str + zoom_min_width_kp_str +\
        zoom_min_width_ki_str + zoom_min_width_kd_str + Kcb_str + Kib_str + Kdb_str +\
        min_iter_str
    print(str_to_print)
    return str_to_print

def do_GP2(old_bounds, new_bounds_raw, X_data_unfiltered, Y_data_unfiltered):
    def in_bounds(x, bounds):
        """Check if a sample x is inside all bounds."""
        return all(bounds[i][0] <= x[i] <= bounds[i][1] for i in range(len(bounds)))
    
    # Filter X_data and Y_data to retain only points inside new_bounds_raw
    X_data = []
    Y_data = []
    
    for x, y in zip(X_data_unfiltered, Y_data_unfiltered):
        if in_bounds(x, new_bounds_raw):
            X_data.append(x)
            Y_data.append(y)
    
    X_data = np.array(X_data)
    Y_data = np.array(Y_data)
    
    print('Most recent time:', time.ctime())
    print("\nIn do_GP2")
    print("Kc Bounds:", new_bounds_raw[0])
    print("Ki Bounds:", new_bounds_raw[1])
    print("Kd Bounds:", new_bounds_raw[2])
    
    Kc_lb2, Kc_ub2 = new_bounds_raw[0]
    Ki_lb2, Ki_ub2 = new_bounds_raw[1]
    Kd_lb2, Kd_ub2 = new_bounds_raw[2]
    
    KcPar = ContinuousParameter("Kc", 0, 1.0)
    KiPar = ContinuousParameter("Ki", 0, 1.0)
    KdPar = ContinuousParameter("Kd", 0, 1.0)
    
    # Number of input dimensions and of initial points
    n_d = 3
    n_init = 100

    # Declare boundaries and initialize three random points for GP to start
    newbounds = ParameterSpace([KcPar, KiPar, KdPar])
    
    # Generate samples
    X_max = 1000
    if len(X_data) > X_max - n_init:
        print(f"[do_GP2] Dataset too large (len(X) = {len(X_data)}). Resetting with 200 LHS samples.")
        
        best_indices = np.argsort(Y_data.flatten())[:100]
        X_best = X_data[best_indices]
        Y_best = Y_data[best_indices]
        
        n_lhs = 100
        # Initialize LHS sampler
        sampler = qmc.LatinHypercube(d=n_d)
        
        # Generate samples
        init_samples = sampler.random(n_lhs)
        
        min_vals = [b[0] for b in new_bounds_raw] # Minimum values for each dimension
        max_vals = [b[1] for b in new_bounds_raw] # Maximum values for each dimension
        
        # Scale LHS samples to desired range
        X_init_1 = np.zeros_like(init_samples)
        for i in range(n_d):
            X_init_1[:, i] = min_vals[i] + (max_vals[i] - min_vals[i]) * init_samples[:, i]

        extrema_samples = []
        for i in range(2):  # For Kc
            for j in range(2):  # For Ki
                for k in range(2): # For Kd
                    sample = np.array([min_vals[0] if i == 0 else max_vals[0],
                                       min_vals[1] if j == 0 else max_vals[1],
                                       min_vals[2] if k == 0 else max_vals[2]])
                    extrema_samples.append(sample)
                   
        # Append median extrema points
        # Medians
        Kc_med = (min_vals[0] + max_vals[0]) / 2
        Ki_med = (min_vals[1] + max_vals[1]) / 2
        Kd_med = (min_vals[2] + max_vals[2]) / 2
       
        # Median Kc
        for i in range(2):
            for j in range(2):
                extrema_samples.append(np.array([Kc_med,
                    min_vals[1] if i == 0 else max_vals[1],
                    min_vals[2] if j == 0 else max_vals[2]]))
        # Median Ki
        for i in range(2):
            for j in range(2):
                extrema_samples.append(np.array(
                    [min_vals[0] if i == 0 else max_vals[0],
                        Ki_med,
                        min_vals[2] if j == 0 else max_vals[2]]))
        # Median Kd
        for i in range(2):
            for j in range(2):
                extrema_samples.append(np.array(
                    [min_vals[0] if i == 0 else max_vals[0],
                        min_vals[1] if j == 0 else max_vals[1],
                        Kd_med]))
       
        extrema_array = np.array(extrema_samples)
       
        X_new_gen = np.vstack([X_init_1, extrema_array])
        
        Y_new_gen = np.empty(len(X_new_gen))
        for rc in range(len(X_new_gen)):
            Y_new_gen[rc] = doublet_test(X_new_gen[rc].reshape(1, n_d))
        
        Y_new_gen = Y_new_gen.reshape(-1,1)
        
        X_init = np.vstack([X_best, X_new_gen])
        Y_init = np.vstack([Y_best, Y_new_gen])
    else:
        # Generate samples
        X_new_samp = get_delta_LHS(old_bounds, new_bounds_raw, n_init)
        
        # --- Generate extrema + median points ---
        min_vals = [b[0] for b in new_bounds_raw] # Minimum values for each dimension
        max_vals = [b[1] for b in new_bounds_raw] # Maximum values for each dimension

        # 8 corners        
        extrema_samples = []
        for i in range(2):  # For Kc
            for j in range(2):  # For Ki
                for k in range(2): # For Kd
                    sample = np.array([min_vals[0] if i == 0 else max_vals[0],
                                       min_vals[1] if j == 0 else max_vals[1],
                                       min_vals[2] if k == 0 else max_vals[2]])
                    extrema_samples.append(sample)
                   
        # Append median extrema points
        # Medians
        Kc_med = (min_vals[0] + max_vals[0]) / 2
        Ki_med = (min_vals[1] + max_vals[1]) / 2
        Kd_med = (min_vals[2] + max_vals[2]) / 2
       
        # Median Kc
        for i in range(2):
            for j in range(2):
                extrema_samples.append(np.array([Kc_med,
                    min_vals[1] if i == 0 else max_vals[1],
                    min_vals[2] if j == 0 else max_vals[2]]))
        # Median Ki
        for i in range(2):
            for j in range(2):
                extrema_samples.append(np.array(
                    [min_vals[0] if i == 0 else max_vals[0],
                        Ki_med,
                        min_vals[2] if j == 0 else max_vals[2]]))
        # Median Kd
        for i in range(2):
            for j in range(2):
                extrema_samples.append(np.array(
                    [min_vals[0] if i == 0 else max_vals[0],
                        min_vals[1] if j == 0 else max_vals[1],
                        Kd_med]))
       
        extrema_array = np.array(extrema_samples)
       
        X_new_samp = np.vstack([X_new_samp, extrema_array])
        # print('X_new_samp:', X_new_samp)
        print('len(X_new_samp):', len(X_new_samp))
        print('X_new_samp.shape:', X_new_samp.shape)
        
        Y_new_samp = np.empty(len(X_new_samp))
        for rc in range(len(X_new_samp)):
            Y_new_samp[rc] = doublet_test(X_new_samp[rc].reshape(1, n_d))
    
        Y_new_samp = Y_new_samp.reshape(-1,1)
    
        X_init = np.vstack([X_data, X_new_samp])
        Y_init = np.vstack([Y_data, Y_new_samp])

    assert X_init.shape[0] == Y_init.shape[0], "Mismatch in sample/label count"    

    normalizer2 = DynNorm(bounds=new_bounds_raw)
    X_norm_pre = normalizer2.normalize(X_init)
    
    Y_normalizer = YNorm(Y_init)
    Y_norm_pre = Y_normalizer.normalize(Y_init)  

    X_norm, Y_norm = remove_duplicates(X_norm_pre, Y_norm_pre, dec=8)

    # Y_norm = np.atleast_2d(Y_norm).reshape(-1, 1)

    # Declare GPy kernel, create GPy model based on initial points, wrap emukit_model around.
    k = GPy.kern.Exponential(input_dim=n_d, ARD=True) # Make sure it takes the number of input dimensions
    gpy_model = GPy.models.GPRegression(X_norm, Y_norm, k) # Give it initial conditions
    gpy_model.Gaussian_noise.variance.constrain_bounded(1e-8, 1e-4)
    gpy_model.kern.lengthscale.constrain_bounded(1e-5, 100)
    # gpy_model.optimize_restarts(10, robust=True)
    gpy_model.optimize()
    
    print(gpy_model)
    # Your dimension names
    param_names = ['Kp', 'Ki', 'Kd']
    
    # Access the ARD lengthscales
    lengthscales = gpy_model.kern.lengthscale.values
    
    # Print each one with its label
    for name, l in zip(param_names, lengthscales):
        print(f"Lengthscale for {name}: {l:.6f}")
    
    emukit_model = GPyModelWrapper(gpy_model) # Get emukit model
    myBopt = BayesianOptimizationLoop(space=newbounds, model=emukit_model)

    sc1_bounds = np.array([[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]])
    stop_cond1 = Bound_or_Cluster_Stop(bounds=sc1_bounds, boundary_margin=0.10, N=8, M=10,
                                       top_count=10)
    stop_cond2 = FixedIterationsStoppingCondition(500)

    def evaluate(x_norm):
        x_real = normalizer2.denormalize(np.array(x_norm).reshape(1, -1))
        y_real = doublet_test(x_real)
        y = Y_normalizer.normalize(y_real)
        return np.array(y)

    myBopt.run_loop(user_function=evaluate, stopping_condition=stop_cond1 | stop_cond2)
    
    X_final_norm = myBopt.loop_state.X
    X_final = normalizer2.denormalize(X_final_norm)
    Y_final_norm = myBopt.loop_state.Y
    Y_final = Y_normalizer.denormalize(Y_final_norm)
    
    rec_bounds = stop_cond1.get_expansion_suggestions(X_final_norm, Y_final,
                            secondary_threshold=0.5)
    
    pid_ex = X_final[np.argmin(Y_final)]
    min_err_ex = Y_final[np.argmin(Y_final)]
    min_err_loc = np.argmin(myBopt.loop_state.Y)
    
    print("PID from GP2:", pid_ex)
    print("error from GP2:", min_err_ex)
    print("location of minimum error:", min_err_loc)
    
    return min_err_ex, pid_ex, min_err_loc, gpy_model.kern.variance.values[0],\
        gpy_model.kern.lengthscale.values[0], gpy_model.kern.lengthscale.values[1],\
            gpy_model.kern.lengthscale.values[2], gpy_model.likelihood.variance.values[0],\
            rec_bounds, X_final, Y_final
            
def do_zoomed_BO(pid_best, raw_bounds, n_init_z=200, zoom_fraction=0.2,\
                 min_widths=[1, 1, 0.5]):
    zoomed_bounds = get_zoomed_bounds(pid_best, raw_bounds, zoom_fraction, min_widths)
    print('Start do_zoomed_BO')
    print('Most recent time:', time.ctime())
    print('zoomed_bounds:', zoomed_bounds)
    
    # Latin Hypercube Sampling
    n_d = len(raw_bounds)
    sampler = qmc.LatinHypercube(d=n_d)
    init_samples_z = sampler.random(n_init_z)
    
    Kc_lb_z, Kc_ub_z = zoomed_bounds[0]
    Ki_lb_z, Ki_ub_z = zoomed_bounds[1]
    Kd_lb_z, Kd_ub_z = zoomed_bounds[2]
    
    min_vals = [Kc_lb_z, Ki_lb_z, Kd_lb_z] # Minimum values for each dimension
    max_vals = [Kc_ub_z, Ki_ub_z, Kd_ub_z] # Maximum values for each dimension
    
    # Scale LHS samples to desired range
    X_init_z1 = np.zeros_like(init_samples_z)
    for i in range(n_d):
        X_init_z1[:, i] = min_vals[i] + (max_vals[i] - min_vals[i]) * init_samples_z[:, i]
    
    extrema_samples = []
    for i in range(2):  # For Kc
        for j in range(2):  # For Ki
            for k in range(2): # For Kd
                sample = np.array([min_vals[0] if i == 0 else max_vals[0],
                                   min_vals[1] if j == 0 else max_vals[1],
                                   min_vals[2] if k == 0 else max_vals[2]])
                extrema_samples.append(sample)
               
    # Append median extrema points
    # Medians
    Kc_med = (min_vals[0] + max_vals[0]) / 2
    Ki_med = (min_vals[1] + max_vals[1]) / 2
    Kd_med = (min_vals[2] + max_vals[2]) / 2
   
    # Median Kc
    for i in range(2):
        for j in range(2):
            extrema_samples.append(np.array([Kc_med,
                min_vals[1] if i == 0 else max_vals[1],
                min_vals[2] if j == 0 else max_vals[2]]))
    # Median Ki
    for i in range(2):
        for j in range(2):
            extrema_samples.append(np.array(
                [min_vals[0] if i == 0 else max_vals[0],
                    Ki_med,
                    min_vals[2] if j == 0 else max_vals[2]]))
    # Median Kd
    for i in range(2):
        for j in range(2):
            extrema_samples.append(np.array(
                [min_vals[0] if i == 0 else max_vals[0],
                    min_vals[1] if j == 0 else max_vals[1],
                    Kd_med]))
   
    esarr = np.array(extrema_samples)
   
    X_init_z = np.vstack([X_init_z1, esarr])
    
    Y_init_z = np.empty(len(X_init_z))
    for rc in range(len(X_init_z)):
        Y_init_z[rc] = doublet_test(X_init_z[rc].reshape(1, n_d))

    Y_init_z = Y_init_z.reshape(-1,1)
    
    assert X_init_z.shape[0] == Y_init_z.shape[0], "Mismatch in sample/label count"

    normalizer_z = DynNorm(bounds=zoomed_bounds)
    X_norm_pre_z = normalizer_z.normalize(X_init_z)
    
    Y_normalizer_z = YNorm(Y_init_z)
    Y_norm_pre_z = Y_normalizer_z.normalize(Y_init_z)
    
    X_norm_z, Y_norm_z = remove_duplicates(X_norm_pre_z, Y_norm_pre_z, dec=8)

    KcPar2 = ContinuousParameter("Kc", 0.0, 1.0)
    KiPar2 = ContinuousParameter("Ki", 0.0, 1.0)
    KdPar2 = ContinuousParameter("Kd", 0.0, 1.0)

    # Declare boundaries and initialize three random points for GP to start
    newbounds2 = ParameterSpace([KcPar2, KiPar2, KdPar2])

    # Declare GPy kernel, create GPy model based on initial points, wrap emukit_model around.
    k2 = GPy.kern.Exponential(input_dim=n_d, ARD=True) # Make sure it takes the number of input dimensions
    gpy_model2 = GPy.models.GPRegression(X_norm_z, Y_norm_z, k2) # Give it initial conditions
#     gpy_model2.Gaussian_noise = (np.max(Y_init_z)-np.min(Y_init_z))/100
#     gpy_model2.Gaussian_noise.variance.fix()
    gpy_model2.Gaussian_noise.variance.constrain_bounded(1e-8, 1e-4)
    gpy_model2.kern.lengthscale.constrain_bounded(1e-5, 30)
    # gpy_model2.optimize_restarts(10, robust=True)
    gpy_model2.optimize()
    
    print(gpy_model2)
    # Your dimension names
    param_names = ['Kp', 'Ki', 'Kd']
    
    # Access the ARD lengthscales
    lengthscales2 = gpy_model2.kern.lengthscale.values
    
    # Print each one with its label
    for name, l in zip(param_names, lengthscales2):
        print(f"Lengthscale for {name}: {l:.6f}")
    
    emukit_model2 = GPyModelWrapper(gpy_model2) # Get emukit model
    myBoptz = BayesianOptimizationLoop(space=newbounds2, model=emukit_model2)
    sc1_bounds = np.array([[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]])
    stop_cond1 = Bound_or_Cluster_Stop(bounds=sc1_bounds, boundary_margin=0.05, N=8, M=10,
                                       top_count = 10)
    stop_cond2 = FixedIterationsStoppingCondition(500)
    
    def evaluate(x_norm):
        x_real = normalizer_z.denormalize(np.array(x_norm).reshape(1, -1))
        y = doublet_test(x_real)
        return np.array(y)

    myBoptz.run_loop(user_function=evaluate, stopping_condition=stop_cond1 | stop_cond2)

    X_final_norm_z = myBoptz.loop_state.X
    X_final_z = normalizer_z.denormalize(X_final_norm_z)
    Y_final_norm_z = myBoptz.loop_state.Y
    Y_final_z = Y_normalizer_z.denormalize(Y_final_norm_z)
    
    bounds_to_exp = stop_cond1.get_expansion_suggestions(X_final_norm_z, Y_final_z)
    
    min_error_BO2 = Y_final_z[np.argmin(Y_final_z)]
    pid_BO2 = X_final_z[np.argmin(Y_final_z)]
    
    var_gp2_str = str(gpy_model2.kern.variance.values[0]) + ','
    ls_gp2_kp_str = str(round(gpy_model2.kern.lengthscale.values[0], 7)) + ','
    ls_gp2_ki_str = str(round(gpy_model2.kern.lengthscale.values[1], 7)) + ','
    ls_gp2_kd_str = str(round(gpy_model2.kern.lengthscale.values[2], 7)) + ','
    gn_gp2_str = str(gpy_model2.likelihood.variance.values[0]) + ','
    len_BO_2_str = str(len(Y_final_z)) + ','

    return pid_BO2, min_error_BO2, X_final_z, Y_final_z, zoom_fraction, min_widths,\
        var_gp2_str, ls_gp2_kp_str, ls_gp2_ki_str, ls_gp2_kd_str, gn_gp2_str, len_BO_2_str, bounds_to_exp

def data_gen(ts_data:list) -> None:
    first_line = "total_time,Kc_best,Ki_best,Kd_best,total_error_best,bo_id,Kc_2,Ki_2,Kd_2,"\
        +"total_error_2,Kc_1,Ki_1,Kd_1,total_error_1,Vdot_Vr,Qk_dot,Ca_in,Ti_in,Ca_ss,Cb_ss,Tr_ss,Tk_ss,kernel_2,var_2,"\
        +"lengthscale_2_Kp,lengthscale_2_Ki,lengthscale_2_Kd,gaussian_noise_2,len_BO_2,kernel_1,"\
        +"var_1,lengthscale_1_Kp,lengthscale_1_Ki,lengthscale_1_Kd,gaussian_noise_1,"\
        +"len_BO_1,no_expansions,expansion_frac,contraction_frac,maxw_kp,maxw_ki,maxw_kd,"\
        +"zoom_frac,no_final_exp,minw_kp,minw_ki,minw_kd,Kc_init,Ki_init,Kd_init,min_iter_num,\n"
    filepath = 'Documents/bo_pid/cstr_new_test/vvr_hightemp_dataset_Exponential_06-27-25.csv'
    file_exists = os.path.isfile(filepath)
    if not file_exists:      
        f1 = open(filepath,'w')
        f1.write(first_line)
        f1.close()
    for i in range(len(ts_data)):
        print('Start time:', time.ctime())
        print(ts_data[i])
        starttime = time.time()
        data_string = do_GP(ts_data[i], -4250, 5100, 150)
        endtime = time.time()
        total_time_str = str(endtime - starttime) + ','
        f2 = open(filepath,'a')
        f2.write(total_time_str + data_string + '\n')
        f2.close()
        print('End time:', time.ctime(), '\n\n')

# # # Initialize LHS sampler
# num_params = 4
# splr = qmc.LatinHypercube(d=num_params) # 4 dimensions are vvr, qk, cain, tin

# # # Generate samples
# training_set_size = 100
# init_samples = splr.random(training_set_size)

# vvr_min, vvr_max = 15, 25 # 1/hr
# # qk_min, qk_max = -8200, -300 # kJ/hr
# cain_min, cain_max = 5000, 5200 # mol/L
# tin_min, tin_max = 100, 110 # Celsius

# min_vals = [vvr_min, cain_min, tin_min] # Minimum values for each dimension
# max_vals = [vvr_max, cain_max, tin_max] # Maximum values for each dimension

# # Scale LHS samples to desired range
# ts_init = np.zeros_like(init_samples)
# for i in range(num_params):
#     ts_init[:, i] = min_vals[i] + (max_vals[i] - min_vals[i]) * init_samples[:, i]

# def load_csv_as_2d_array_float(fp):
#     data = []
    
#     with open(fp, newline='', encoding='utf-8') as csvfile:
#         csvreader = csv.reader(csvfile)
#         for row in csvreader:
#             data.append([float(value) for value in row])
            
#     return data

# filepath8 = '/home/t778b526/Documents/bo_pid/training_data_backup_04-08-25.csv'
# ts_init = load_csv_as_2d_array_float(filepath8)

# print('maximum values for each column:\n', np.max(ts_init, 0))
# print('minimum values for each column:\n', np.min(ts_init, 0))

# print(ts_init[18])
# print(ts_init[19])
# print(ts_init[20])

# with open('restricted_trainingset.csv', 'w', newline='') as file_to_write:
#     writer = csv.writer(file_to_write)
#     writer.writerows(ts_init)

ts_init = [15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]
# ts_init = [21, 22, 23, 24, 25]

data_gen(ts_init)