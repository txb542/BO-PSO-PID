# -*- coding: utf-8 -*-
"""
Created on Wed Jun 25 11:57:27 2025

@author: tateb
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

from joblib import Parallel, delayed

import numpy as np
from scipy.stats import qmc
import itertools

import time
import warnings
from collections import deque

import control as ctl
from scipy.integrate import odeint
# import matplotlib.pyplot as plt
# import os


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
    
    lastCa = Ca[-1]
    lastCb = Cb[-1]
    lastTr = Tr[-1]
    lastTk = Tk[-1]
    
    return(lastCa, lastCb, lastTr, lastTk)

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
        
def doublet_test(kp_dt, ki_dt, kd_dt, Vdot_Vr, Qkdot, Ca_in, T_in, Ca_ss, Cb_ss, Tr_ss, Tk_ss):
    # Simulation time
    num_sec = 5
    num_hours = 4
    t = np.linspace(0,num_hours,round(num_hours*60*(60/num_sec)+1)) # Total of two hours of simulation time, sampling every 5 seconds
    
    # Manipulated variable conditions
    Cai = np.ones(len(t)) * Ca_in # mol/m3
    Ti = np.ones(len(t)) * T_in # Celsius

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
    Kp = kp_dt
    Ki = ki_dt
    Kd = kd_dt
    
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
        
    # total_error = np.array([sum(abs(error0)) + sum(abs(error1))])
    return float(np.sum(np.abs(error0)) + np.sum(np.abs(error1)))
    # return total_error
    # return total_error[:, None]
    
# def run_PID_PSO_test(reactor_inputs, raw_bounds=None, zoom=True, verbose=False):
def run_PID_PSO_test(Vdot_Vr_1, Qk_1, Cain_1, Tin_1, zoom=True, verbose=False):
    """
    Runs Adaptive PSO to tune PID parameters for a given reactor condition.
    
    Args:
        reactor_inputs: tuple or dict of system conditions (e.g., (Cai, F, Cac, Fc))
        raw_bounds: list of [min, max] for each of (Kp, Ki, Kd) in raw (unscaled) space
        zoom: whether to perform a zoomed-in refinement step after initial optimization
        verbose: whether to print output

    Returns:
        dict with best PID parameters, performance, and PSO metadata
    """
    print('Most recent time:', time.ctime())
    # Kc_b, Ki_b, Kd_b, Ca_ss_PSO, Cb_ss_PSO, Tr_ss_PSO, Tk_ss_PSO = get_sys(Vdot_Vr_1, Qk_1, Cain_1, Tin_1)
    
    Ca_ss_PSO, Cb_ss_PSO, Tr_ss_PSO, Tk_ss_PSO = get_ss(Vdot_Vr_1,Qk_1,Cain_1,Tin_1)
    sys = vdv_lin(Vdot_Vr_1, Qk_1, Cain_1, Tin_1, Ca_ss_PSO, Cb_ss_PSO, Tr_ss_PSO, Tk_ss_PSO)
    
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
    Kc_b = (1/gain)*((tau1+tau2)/(theta+tauc))
    Ki_b = Kc_b / (tau1 + tau2)
    Kd_b = Kc_b * (tau1*tau2/(tau1 + tau2))
    
    Kcb_str = str(Kc_b) + ','
    Kib_str = str(Ki_b) + ','
    Kdb_str = str(Kd_b) + ','
    
    Vdot_Vr_str = str(Vdot_Vr_1) + ','
    Qk_str = str(Qk_1) + ','
    Cain_str = str(Cain_1) + ','
    Tin_str = str(Tin_1) + ','
    
    Ca_ss_str = str(Ca_ss_PSO) + ','
    Cb_ss_str = str(Cb_ss_PSO) + ','
    Tr_ss_str = str(Tr_ss_PSO) + ','
    Tk_ss_str = str(Tk_ss_PSO) + ','
    
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
    # raw_bounds = [[Kc_lb, Ki_lb, Kd_lb], [Kc_lb, Ki_lb, Kd_lb]]

    def evaluate_PID_performance(pid_params):
        """Your system-specific objective function, e.g., total tracking error from simulation."""
        Kp, Ki, Kd = pid_params
        return doublet_test(Kp, Ki, Kd, Vdot_Vr_1, Qk_1, Cain_1, Tin_1, Ca_ss_PSO, Cb_ss_PSO, Tr_ss_PSO, Tk_ss_PSO) # Can add reactor params here - no need for global vars

    dictreturn = do_PSO(
        objective_func=evaluate_PID_performance,
        raw_bounds=raw_bounds,
        num_particles=100,
        num_iterations=100,
        c1=2.0,
        c2=2.0,
        num_best=15,
        boundary_trigger_frac=0.8,
        secondary_frac=0.5,
        boundary_margin=0.1,
        use_sobol=True,
        max_edge_points=26,
        reinject_edges=True,
    ) # seed=42
    
    total_first_iter_str = str(dictreturn.get('pso_iter')) + ','
    total_exp_str = str(len(dictreturn.get('expansion_log'))) + ','
    
    if zoom:
        zoomreturn = do_zoomed_PSO(
            objective_func=evaluate_PID_performance,
            center_point=dictreturn['best_position'],
            zoom_factor=0.25,
            raw_bounds=dictreturn['expanded_bounds'],
            num_particles=100,
            num_iterations=100,
            c1=1.0,
            c2=2.0,
            num_best=15,
            boundary_trigger_frac=0.8,
            secondary_frac=0.5,
            boundary_margin=0.1,
            max_edge_points=26,
            use_sobol=True,
            verbose=verbose
        )
        
        total_zoom_iter_str = str(zoomreturn.get('pso_iter')) + ','
        total_zoom_exp_str = str(len(zoomreturn.get('expansion_log'))) + ','
        Kc_zoom_str = str(zoomreturn['best_position'][0]) + ','
        Ki_zoom_str = str(zoomreturn['best_position'][1]) + ','
        Kd_zoom_str = str(zoomreturn['best_position'][2]) + ','
        total_err_zoom_str = str(zoomreturn['best_score']) + ',' # Last element, no need for comma
        
        if zoomreturn['best_score'] <= dictreturn['best_score']:
            Kc_best_str = str(zoomreturn['best_position'][0]) + ','
            Ki_best_str = str(zoomreturn['best_position'][1]) + ','
            Kd_best_str = str(zoomreturn['best_position'][2]) + ','
            total_err_best_str = str(zoomreturn['best_score']) + ','
            pso_id_str = '2,'
        else:
            Kc_best_str = str(dictreturn['best_position'][0]) + ','
            Ki_best_str = str(dictreturn['best_position'][1]) + ','
            Kd_best_str = str(dictreturn['best_position'][2]) + ','
            total_err_best_str = str(dictreturn['best_score']) + ','
            pso_id_str = '1,'
    else:
        # final_result = dictreturn
        Kc_best_str = str(dictreturn['best_position'][0]) + ','
        Ki_best_str = str(dictreturn['best_position'][1]) + ','
        Kd_best_str = str(dictreturn['best_position'][2]) + ','
        total_err_best_str = str(dictreturn['best_score']) + ','
        
        Kc_zoom_str = '0,'
        Ki_zoom_str = '0,'
        Kd_zoom_str = '0,'
        total_err_zoom_str = '0,'
        pso_id_str = '1,'
    
    Kc_first_str = str(dictreturn['best_position'][0]) + ','
    Ki_first_str = str(dictreturn['best_position'][1]) + ','
    Kd_first_str = str(dictreturn['best_position'][2]) + ','
    total_err_first_str = str(dictreturn['best_score']) + ','
    
    str_to_print = Kc_best_str + Ki_best_str + Kd_best_str + total_err_best_str + pso_id_str + \
        Kc_zoom_str + Ki_zoom_str + Kd_zoom_str + total_err_zoom_str + \
        Kc_first_str + Ki_first_str + Kd_first_str + total_err_first_str + \
        Vdot_Vr_str + Qk_str + Cain_str + Tin_str + Ca_ss_str + Cb_ss_str + Tr_ss_str + \
        Tk_ss_str + Kcb_str + Kib_str + Kdb_str + total_first_iter_str + total_zoom_iter_str + \
        total_exp_str + total_zoom_exp_str
        
    return str_to_print

def generate_edge_midpoints(bounds, max_n):
    bounds = np.array(bounds)
    dim = bounds.shape[0]
    lower = bounds[:, 0]
    upper = bounds[:, 1]
    mid = (lower + upper) / 2

    levels = [[l, m, u] for l, m, u in zip(lower, mid, upper)]
    full_grid = np.array(list(itertools.product(*levels)))

    center = mid
    mask = ~np.all(np.isclose(full_grid, center, rtol=1e-5), axis=1)
    grid_points = full_grid[mask]

    if len(grid_points) > max_n:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(grid_points), size=max_n, replace=False)
        grid_points = grid_points[idx]

    return grid_points

def scale_to_unit(x_raw, bounds):
    bounds = np.array(bounds)
    return (x_raw - bounds[:, 0]) / (bounds[:, 1] - bounds[:, 0])

def unscale_from_unit(x_unit, bounds):
    bounds = np.array(bounds)
    return bounds[:, 0] + x_unit * (bounds[:, 1] - bounds[:, 0])

class AdaptivePSO:
    def __init__(self, objective_func, bounds, num_particles=100, num_iterations=100,
                 w_start=0.9, w_end=0.4, c1=2.0, c2=2.0, num_best = 10,
                 boundary_trigger_frac=0.8, secondary_frac=0.5, boundary_margin=0.1, seed=None,
                 use_sobol=True, max_edge_points=26, reinject_edges=True, max_velocity_init=0.2,
                 max_velocity_final_factor=0.5):

        self.obj_func_raw = objective_func
        self.raw_bounds = np.array(bounds, dtype=float)
        self.bounds = np.array([[0.0, 1.0]] * len(bounds))
        self.num_particles = num_particles
        self.num_iterations = num_iterations
        self.dim = len(bounds)
        self.w_start = w_start
        self.w_end = w_end
        self.c1 = c1
        self.c2 = c2
        self.num_best = num_best
        self.boundary_trigger_frac = boundary_trigger_frac
        self.secondary_frac = secondary_frac
        self.boundary_margin = boundary_margin
        self.use_sobol = use_sobol
        self.max_edge_points = max_edge_points
        self.reinject_edges = reinject_edges
        self.rng = np.random.default_rng(seed)
        self.max_velocity_init=max_velocity_init
        self.max_velocity_final_factor=max_velocity_final_factor

        self.expanded_dims = np.zeros(self.dim, dtype=bool)
        self.reset_generations = False

        self.positions = self._init_positions()
        self.velocities = self._init_velocities()
        self.personal_best_positions = np.copy(self.positions)
        self.personal_best_scores = self._evaluate(self.positions)
        self.global_best_position = self.personal_best_positions[np.argmin(self.personal_best_scores)]
        self.global_best_score = np.min(self.personal_best_scores)
        self.expansion_log = []
        
        print("bounds:", self.bounds)

    def _init_positions(self):
        edge_particles = generate_edge_midpoints(self.bounds, self.max_edge_points)
        remaining = self.num_particles - len(edge_particles)

        lower, upper = self.bounds[:, 0], self.bounds[:, 1]

        if self.use_sobol:
            sampler = qmc.Sobol(d=self.dim, scramble=True, seed=self.rng.integers(0, 1e9))
        else:
            sampler = qmc.LatinHypercube(d=self.dim, seed=self.rng.integers(0, 1e9))

        sobol_sample = qmc.scale(sampler.random(n=remaining), lower, upper)

        return np.vstack([edge_particles, sobol_sample])

    def _init_velocities(self):
        return self.rng.uniform(-1, 1, size=(self.num_particles, self.dim)) * 0.1

    # def _evaluate(self, positions):
    #     raw_positions = unscale_from_unit(positions, self.raw_bounds)
    #     return np.apply_along_axis(self.obj_func_raw, 1, raw_positions)
    
    # def _evaluate(self, positions):
    #     raw_positions = unscale_from_unit(positions, self.raw_bounds)
    #     scores = np.apply_along_axis(self.obj_func_raw, 1, raw_positions)
    #     return scores.flatten() # Ensure shape is (num_particles,)
    
    # Parallel attempt
    def _evaluate(self, positions):
        raw_positions = unscale_from_unit(positions, self.raw_bounds)
    
        # Parallel evaluation
        results = Parallel(n_jobs=5)(delayed(self.obj_func_raw)(p) for p in raw_positions)
    
        # Flatten and convert to NumPy array
        scores = np.array([float(np.ravel(res)[0]) for res in results])
        return scores

    def _expand_bounds_if_needed(self, best_positions):
        print('time start checking expansion:', time.ctime())
        old_bounds = self.raw_bounds.copy() # Store current bounds before modification

        # Remember bounds are already defined as scaled - [0,1]
        raw_lower, raw_upper = self.raw_bounds[:, 0], self.raw_bounds[:, 1]
        raw_span = raw_upper - raw_lower
        print("raw span:", raw_span)
        lower, upper = self.bounds[:, 0], self.bounds[:, 1]
        span = upper - lower
        print("span:", span)

        near_lower = best_positions <= (lower + self.boundary_margin * span)
        near_upper = best_positions >= (upper - self.boundary_margin * span)
        near_bounds = np.logical_or(near_lower, near_upper)

        primary_trigger = np.sum(np.any(near_bounds, axis=1)) >= int(self.boundary_trigger_frac * len(best_positions))

        if primary_trigger:
            print('\nPrimary trigger activated')
            print('old bounds:', self.raw_bounds)
            self.expansion_log.append(self.current_iteration)
            self.reset_generations = True
            # self.velocities = self._init_velocities() # Does all velocities
            n_reset = self.num_particles // 2  # Or some other fraction
            reset_indices = np.argsort(self.personal_best_scores)[-n_reset:]
            self.velocities[reset_indices] = self._init_velocities()[reset_indices]
            
            per_dim_counts = np.sum(near_bounds, axis=0)
            secondary_trigger = per_dim_counts >= int(self.secondary_frac * len(best_positions))
            dims_to_check = np.where(np.logical_or(np.any(near_bounds, axis=0), secondary_trigger))[0]

            for dim in dims_to_check:
                if np.any(near_lower[:, dim]):
                    self.raw_bounds[dim, 0] -= 0.5 * raw_span[dim]
                    self.raw_bounds[dim, 1] -= 0.3 * raw_span[dim]
                    self.expanded_dims[dim] = True
                    print("for lower - raw_span[dim]:", raw_span[dim])
                if np.any(near_upper[:, dim]):
                    self.raw_bounds[dim, 1] += 0.5 * raw_span[dim]
                    self.raw_bounds[dim, 0] += 0.3 * raw_span[dim]
                    self.expanded_dims[dim] = True
                    print("for upper - raw_span[dim]:", raw_span[dim])

                # Ensure bounds are valid
                if self.raw_bounds[dim, 1] < self.raw_bounds[dim, 0]:
                    midpoint = np.mean(self.raw_bounds[dim])
                    delta = 0.1 * abs(span[dim])
                    self.raw_bounds[dim] = [midpoint - delta, midpoint + delta]

            if self.reinject_edges:
                edge_points = generate_edge_midpoints(self.bounds, self.max_edge_points * 2)
                mask = np.any([
                    np.isclose(edge_points[:, dim], self.bounds[dim, 0], rtol=1e-3) |
                    np.isclose(edge_points[:, dim], self.bounds[dim, 1], rtol=1e-3)
                    for dim in dims_to_check
                ], axis=0)
                reinjected = edge_points[mask]
                if len(reinjected) > 0:
                    n_reinject = min(len(reinjected), self.num_particles // 4)
                    elite_indices = np.argsort(self.personal_best_scores)[:n_reinject]
                    non_elite_indices = np.setdiff1d(np.arange(self.num_particles), elite_indices)[:n_reinject]

                    reinjected = reinjected[:n_reinject]
                    self.positions[non_elite_indices] = reinjected
                    self.velocities[non_elite_indices] = self.rng.uniform(-1, 1, size=(n_reinject, self.dim)) * 0.1
                    
            # Rescale particles to the new bounds
            raw_positions = unscale_from_unit(self.positions, old_bounds)
            self.positions = scale_to_unit(raw_positions, self.raw_bounds)

            raw_pbest = unscale_from_unit(self.personal_best_positions, old_bounds)
            self.personal_best_positions = scale_to_unit(raw_pbest, self.raw_bounds)

            raw_gbest = unscale_from_unit(self.global_best_position[None, :], old_bounds)[0]
            self.global_best_position = scale_to_unit(raw_gbest[None, :], self.raw_bounds)[0]
            
            print('new bounds:', self.raw_bounds)
            print('time after expansion:', time.ctime())

    def optimize(self, verbose=False):
        # for iter_ in range(self.num_iterations):
        generations_since_expansion = 0
        total_generations_done = 0
        generations_since_improvement = 0
        max_total_generations = 10000  # safety cap to avoid infinite loop
        max_gen_without_improvement = 10

        while generations_since_expansion < self.num_iterations and \
            total_generations_done < max_total_generations and \
            generations_since_improvement < max_gen_without_improvement:
            print('Most recent time:', time.ctime())
            print('generations_since_expansion:', generations_since_expansion)
            self.current_iteration = generations_since_expansion
            w = self.w_start - (self.w_start - self.w_end) * (generations_since_expansion / self.num_iterations)
            r1 = self.rng.uniform(size=(self.num_particles, self.dim))
            r2 = self.rng.uniform(size=(self.num_particles, self.dim))

            cognitive = self.c1 * r1 * (self.personal_best_positions - self.positions)
            social = self.c2 * r2 * (self.global_best_position - self.positions)
            self.velocities = w * self.velocities + cognitive + social
            new_max_velocity = self.max_velocity_init * (1 - (generations_since_expansion / self.num_iterations) * (1 - self.max_velocity_final_factor))
            self.velocities = np.clip(self.velocities, -new_max_velocity, new_max_velocity)
            
            self.positions += self.velocities
            self.positions = np.clip(self.positions, self.bounds[:, 0], self.bounds[:, 1])

            scores = self._evaluate(self.positions)
            # scores = self._evaluate(self.positions).flatten()
            # The above line may improve robustness, but needs testing first
            improved = scores < self.personal_best_scores-0.5
            self.personal_best_positions[improved] = self.positions[improved]
            self.personal_best_scores[improved] = scores[improved]

            min_idx = np.argmin(self.personal_best_scores)
            if self.personal_best_scores[min_idx] < self.global_best_score:
                self.global_best_score = self.personal_best_scores[min_idx]
                self.global_best_position = self.personal_best_positions[min_idx]
                generations_since_improvement = 0
            else:
                generations_since_improvement += 1

            best_subset = self.personal_best_positions[np.argsort(self.personal_best_scores)[:self.num_best]]
            self._expand_bounds_if_needed(best_subset)
            
            if getattr(self, "reset_generations", False):
                generations_since_expansion = 0
                self.reset_generations = False

            if verbose and generations_since_expansion % 10 == 0:
                print(f"Iter {total_generations_done}: Best score = {self.global_best_score:.5f}")
                
            generations_since_expansion += 1
            total_generations_done += 1
            
            print('best score:', self.global_best_score)
            print('scaled best position:', self.global_best_position)
            print('best position:', unscale_from_unit(self.global_best_position[None, :], self.raw_bounds)[0])
            print('time at end of iteration:', time.ctime(), '\n\n')

        final_unscaled = unscale_from_unit(self.global_best_position, self.raw_bounds)
        return final_unscaled, self.global_best_score, total_generations_done 

def do_PSO(objective_func, raw_bounds, num_particles=30, num_iterations=100, c1=2.0, c2=2.0,
           num_best=10, boundary_trigger_frac=0.8, secondary_frac=0.5, boundary_margin=0.1,
           use_sobol=True, max_edge_points=30, reinject_edges=True,
           seed=None, verbose=False):
    """
    Primary wrapper for AdaptivePSO. Handles initial search with boundary expansions.
    """
    # from adaptive_pso import AdaptivePSO  # If using an external file

    pso = AdaptivePSO(
        objective_func=objective_func,
        bounds=raw_bounds,
        num_particles=num_particles,
        num_iterations=num_iterations,
        c1=c1,
        c2=c2,
        num_best=num_best,
        boundary_trigger_frac=boundary_trigger_frac,
        secondary_frac=secondary_frac,
        boundary_margin=boundary_margin,
        use_sobol=use_sobol,
        max_edge_points=max_edge_points,
        reinject_edges=reinject_edges,
        seed=seed
    )
    
    ### NOTE that for first iteration, it zoomed into a range of 0.25 per dimension
    ### around the center point. This explains why it gets hyper accurate results.
    ### Not sure if I need to retest using the other method or not.
    best_position, best_score, totalgen1 = pso.optimize(verbose=verbose)
    return {
        'best_position': best_position,
        'best_score': best_score,
        'expanded_bounds': pso.raw_bounds,
        'expansion_log': pso.expansion_log,
        'pso_instance': pso,
        'pso_iter': totalgen1
    }

def do_zoomed_PSO(objective_func, center_point, zoom_factor=0.25, raw_bounds=None, num_particles=30, 
           num_iterations=100, c1=1.0, c2=2.0, num_best=10, boundary_trigger_frac=0.8, secondary_frac=0.5,
           boundary_margin=0.1, use_sobol=True, max_edge_points=26, reinject_edges=True,
           seed=None, verbose=False):

#     def do_zoomed_PSO(objective_func, center_point, zoom_factor=0.25, raw_bounds=None,
#                       num_particles=20, num_iterations=50, seed=None,
#                       use_sobol=True, verbose=False):
    """
    Zoomed-in local PSO search centered on `center_point`, using a percentage zoom_factor.
    """
    dim = len(center_point)
    center_point = np.array(center_point)

    if raw_bounds is None:
        raise ValueError("Must pass raw_bounds to do_zoomed_PSO to clip new search region.")

    raw_bounds = np.array(raw_bounds)
    lower_clip, upper_clip = raw_bounds[:, 0], raw_bounds[:, 1]

    span = (upper_clip - lower_clip) * zoom_factor
    local_lower = np.maximum(center_point - span / 2, lower_clip)
    local_upper = np.minimum(center_point + span / 2, upper_clip)
    zoom_bounds = np.stack([local_lower, local_upper], axis=1)

    # from adaptive_pso import AdaptivePSO  # If using an external file
    pso = AdaptivePSO(
        objective_func=objective_func,
        bounds=zoom_bounds,
        num_particles=num_particles,
        num_iterations=num_iterations,
        c1 = c1,
        c2 = c2,
        num_best=num_best,
        boundary_trigger_frac=boundary_trigger_frac,
        secondary_frac=secondary_frac,
        boundary_margin=boundary_margin,
        use_sobol=use_sobol,
        max_edge_points=max_edge_points,
        reinject_edges=reinject_edges,
        seed=seed
    )

    best_position, best_score, totalgen2 = pso.optimize(verbose=verbose)
    return {
        'best_position': best_position,
        'best_score': best_score,
        'zoom_bounds': zoom_bounds,
        'expansion_log': pso.expansion_log,
        'pso_instance': pso,
        'pso_iter': totalgen2
    }

def data_gen(ts_data:list, ts_data2:list, ts_data3:list) -> None:
    first_line = "total_time,Kc_best,Ki_best,Kd_best,total_error_best,pso_id,Kc_2,Ki_2,Kd_2,"\
        +"total_error_2,Kc_1,Ki_1,Kd_1,total_error_1,Vdot_Vr,Qk_dot,Ca_in,Ti_in,Ca_ss,Cb_ss,"\
        +"Tr_ss,Tk_ss,Kc_init,Ki_init,Kd_init,total_first_iter,total_zoom_iter,"\
        +"total_first_expansion,total_zoom_expansion\n"
        # +"no_expansions,expansion_frac,contraction_frac,maxw_kp,maxw_ki,maxw_kd,"\
        # +"zoom_frac,no_final_exp,minw_kp,minw_ki,minw_kd,Kc_init,Ki_init,Kd_init,min_iter_num,\n"
    filepath = 'Documents/bo_pid/pso_testing/vvr_tin_cain_grid_PSO_06-26-25.csv'
    file_exists = os.path.isfile(filepath)
    if not file_exists:      
        f1 = open(filepath,'w')
        f1.write(first_line)
        f1.close()
    for i in range(len(ts_data)):
        for j in range(len(ts_data2)):
            for k in range(len(ts_data3)):
                print('Start time:', time.ctime())
                print(ts_data[i])
                starttime = time.time()
                data_string = run_PID_PSO_test(ts_data[i], -4250, ts_data2[j], ts_data3[k])
                endtime = time.time()
                total_time_str = str(endtime - starttime) + ','
                f2 = open(filepath,'a')
                f2.write(total_time_str + data_string + '\n')
                f2.close()
                print('End time:', time.ctime(), '\n\n')
        
# ts_init = [25, 27, 29, 31, 33, 35]
# ts_init2 = [5300,5500]
ts_init = [35]
ts_init2 = [5300, 5500]
ts_init3 = [130,135,140,145,150]

data_gen(ts_init, ts_init2, ts_init3)

# best score: 721.788879216805
# scaled best position: [0.81654304 0.57421841 0.65905389]
# best position: [345.31498889 904.53264858   6.07936963]