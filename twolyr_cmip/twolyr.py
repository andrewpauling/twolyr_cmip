import numpy as np
import pandas as pd
import statsmodels.api as sm
import xarray as xr
from scipy.integrate import odeint
from scipy.stats import linregress
from scipy.optimize import least_squares


def new_linregress(x, y):
    # Wrapper around scipy linregress to use in apply_ufunc
    slope, intercept, r_value, p_value, std_err = linregress(x, y)
    return np.array([slope, intercept, r_value, p_value, std_err])


def xregress(x, y):
    """Perform linear regression for xarray DataArrays

    Args:
        x (xr.DataArray): independent variable
        y (xr.DataArray): dependent variable

    Returns:
        xr.DataArray: regression slope
        xr.DataArray: regression intercept
    """
    stats = xr.apply_ufunc(
        new_linregress,
        x,
        y,
        input_core_dims=[["year"], ["year"]],
        output_core_dims=[["parameter"]],
        output_sizes={"parameter": 5},
        vectorize=True
    )

    slope = stats.isel(parameter=0)
    intcpt = stats.isel(parameter=1)

    return slope, intcpt


class TwoLayerModel:

    def __init__(self, params=None, model_epsilon="unit", verbose=False):
        """Class containing methods to fit parameters for and run the Held two-layer model

        Args:
            params (dict, optional): Dictionary containing model parameters. Defaults to None.
            model_epsilon (str, optional): Whether to use epsilon=1 ("unit") or epsilon fit to model ("variable"). Defaults to "unit".
            verbose (bool, optional): Flag to print additional information. Defaults to False.
        """

        if params is dict:
            self.params = params
        elif params is not None:
            raise TypeError(
                "Params must be provided in a dict, or fit using fit_params"
            )
        else:
            self.params = {}

        self.verbose = verbose

        if model_epsilon in ["unit", "variable"]:
            self.model_epsilon = model_epsilon
        else:
            raise ValueError(
                "parameter model_epsilon must be one of ['unit', 'variable']"
            )

    def __repr__(self):
        summary = [f"{type(self).__name__}"]
        summary.append("---Parameters---")
        summary.append(f"F: {self.params['F_ref'].data}")
        summary.append(f"lambda: {self.params['lam'].data}")
        summary.append(f"T_eq: {self.params['Teq'].data}")
        summary.append(f"gamma: {self.params['gamma'].data}")
        summary.append(f"epsilon: {self.params['epsilon'].data}")

        return "\n".join(summary)

    def __gregory_regression(self, T, N):
        """Perform Gregory regression to get forcing, feedback and equilibrium T

        Args:
            T (array like): global-mean, annual-mean surface temperature anomaly
            N (array like): global-mean, annual-mean TOA radiative imbalance
        """
        
        if self.verbose:
            print("--Running Gregory regression")

        slope, intcpt = xregress(T, N)

        self.params["Teq"] = -intcpt / slope
        self.params["F_ref"] = intcpt
        self.params["lam"] = -slope
        
        if self.verbose:
            print(f"--Teq = {self.params['Teq']}")
            print(f"--F_ref = {self.params['F_ref']}")
            print(f"--lam = {self.params['lam']}")

    def __compute_prelim_params(self, T):
        """Compute preliminary parameters needed to fit model according to Geoffroy et al. (2013)

        Args:
            T (array like): global-mean, annual-mean surface temperature anomaly
        """
        
        if self.verbose:
            print("--Computing preliminary parameters")

        y = np.log(self.params["Teq"] - T)
        if np.isnan(y).any("year"):
            y = y.fillna(1e-8)

        if self.lambda_type == "fast":
            slope, intcpt = xregress(self.YRS[10:20], y.isel(year=slice(10, 20)))
        else:
            slope, intcpt = xregress(self.YRS[30:], y.isel(year=slice(30, None)))

        self.tau_s = -1 / slope
        self.a_s = np.exp(intcpt - np.log(self.params["Teq"]))

        self.a_f = 1 - self.a_s

        
        tmp = xr.DataArray(np.zeros((self.ens, 10)),
                           dims=("ensmem", "x"),
                           coords={"ensmem": ("ensmem", T.ensmem.data)})
        for t in range(1, 11):
            tmp[:, t - 1] = t / (
                np.log(self.a_f)
                - np.log(
                    1
                    - T.isel(year=(t - 1)) / self.params["Teq"]
                    - self.a_s * np.exp(-t / self.tau_s)
                )
            )

        self.tau_f = tmp.mean("x", skipna=True).squeeze()
        
        if self.verbose:
            print(f"--tau_s = {self.tau_s}")
            print(f"--tau_f = {self.tau_f}")
            print(f"--a_s = {self.a_s}")
            print(f"--a_f = {self.a_f}")

    def __compute_model_params(self):
        """Compute shallow and deep heat capacities and gamma"""

        if self.verbose:
            print("--Compute heat capacities and gamma")
        self.params["C"] = self.params["lam"] / (
            self.a_f / self.tau_f + self.a_s / self.tau_s
        )
        self.params["C_0"] = (
            self.params["lam"] * (self.tau_f * self.a_f + self.tau_s * self.a_s)
            - self.params["C"]
        )
        self.params["gamma"] = self.params["C_0"] / (
            self.tau_f * self.a_s + self.tau_s * self.a_f
        )

    def __compute_general_params(self):
        """Compiute intermediate parameters needed for subsequent calculations"""
        if self.verbose:
            print("--Compute general parameters")
        self.b = (self.params["lam"] + self.params["gamma"]) / self.params[
            "C"
        ] + self.params["gamma"] / self.params["C_0"]
        self.b_star = (self.params["lam"] + self.params["gamma"]) / self.params[
            "C"
        ] - self.params["gamma"] / self.params["C_0"]
        self.delta = self.b**2 - 4 * self.params["lam"] * self.params["gamma"] / (
            self.params["C"] * self.params["C_0"]
        )

    def __compute_mode_params(self):
        """Compute mode parameters according to Geoffroy et al. (2013)"""
        if self.verbose:
            print("--Compute mode parameters")
        self.phi_f = (
            self.params["C"]
            / (2 * self.params["gamma"])
            * (self.b_star - np.sqrt(self.delta))
        )
        self.phi_s = (
            self.params["C"]
            / (2 * self.params["gamma"])
            * (self.b_star + np.sqrt(self.delta))
        )

    def __compute_T_analytical(self):
        """Analytical solution for deep ocean temperature"""
        self.T_1 = self.params["Teq"] * (
            1 
            - self.a_f*np.exp(-self.YRS / self.tau_f) 
            - self.a_s*np.exp(-self.YRS / self.tau_s)
        )
        self.T_0 = self.params["Teq"] * (
            1
            - self.phi_f * self.a_f * np.exp(-self.YRS / self.tau_f)
            - self.phi_s * self.a_s * np.exp(-self.YRS / self.tau_s)
        )

    def __compute_H(self):
        """Compute heat uptale by deep ocean"""
        return self.params["gamma"] * (self.T_1 - self.T_0)
    

    def __compute_epsilon(self, T, N):
        """Iteratively fit epsilon with iterative procedure of Geoffroy et al. (2013b)

        Args:
            T (array like): global-mean, annual-mean surface temperature anomaly
            N (array like): global-mean, annual-mean TOA radiative imbalance
        """

        self.params["epsilon"] = xr.DataArray(np.ones((self.ens)),
                                              dims="ensmem",
                                              coords={"ensmem": ("ensmem", T.ensmem.data)})
        
        def eps_func(x):
            return N.squeeze() - x[0] + x[1] * T.squeeze() + (x[2] - 1.) * H.squeeze()
    
        for e in range(self.ens):
            init_guess = [self.params["F_ref"].isel(ensmem=e).data, 
                          self.params["lam"].isel(ensmem=e).data,
                          1]
            for i in range(10):
                self.__compute_T_analytical()
                H = self.__compute_H()
                
                updated_guess = least_squares(eps_func, x0=init_guess)
                init_guess[:] = updated_guess.x
                
                self.params["F_ref"][e] = init_guess[0]
                self.params["lam"][e] = init_guess[1]
                self.params["epsilon"][e] = init_guess[2]
                self.params["Teq"][e] = self.params["F_ref"][e] / self.params["lam"][e]
                
                self.__compute_prelim_params(T)
                self.__compute_model_params()
                self.__compute_general_params()
                self.__compute_mode_params()
                
                if self.verbose:
                    print(f"Iteration {i}: epsilon = {init_guess[2]}, F = {init_guess[0]}, lam = {init_guess[1]}")

    def fit_params(self, T, N, lambda_type="slow", run_type="abrupt-4xCO2"):
        """Fit the parameters of the two-layer model using the methods of Geoffroy et al. (2013)

        Args:
            T (array like): global-mean, annual-mean surface temperature anomaly
            N (array like): global-mean, annual-mean TOA radiative imbalance
            lambda_type (str): flag for using full 150 years to fit lambda or just first 20
        """
        
        if self.verbose:
            print("Fitting parameters")
            print(f"--Using {lambda_type} lambda")
        if len(T.shape) > 1:
            self.ens = T["ensmem"].size
        else:
            self.ens = 1
            
        self.YRS = T["year"]
        self.lambda_type = lambda_type
        self.run_type = run_type
            
        if self.lambda_type == "fast":
            self.__gregory_regression(T.isel(year=slice(None, 20)), N.isel(year=slice(None, 20)))
        else:
            self.__gregory_regression(T, N)
        self.__compute_prelim_params(T)
        self.__compute_model_params()
        self.__compute_general_params()
        self.__compute_mode_params()

        if self.model_epsilon == "variable":
            self.__compute_epsilon(T, N)
        else:
            self.params["epsilon"] = xr.DataArray(
                [1.0],
                dims="ensmem",
                coords={"ensmem": ("ensmem", T.ensmem.data)}
            )

    def __twolayersystem(self, state, t, R, tstep, ens):
        """Function describing the two layer model of Held et al. (2010)

        Args:
            state (list): current values of shallow and deep temperature
            t (array-like): time
            R (array-like): radiative forcing
            tstep (float): time step in units of years

        Returns:
            list: change in shallow and deep surface temperature
        """
        T1, T2 = state

        c1 = self.params["C"].isel(ensmem=ens) / tstep
        c2 = self.params["C_0"].isel(ensmem=ens) / tstep

        dT1 = (
            1
            / c1
            * (
                -self.params["lam"].isel(ensmem=ens) * T1
                + R
                + self.params["epsilon"].isel(ensmem=ens) * self.params["gamma"].isel(ensmem=ens) * (T2 - T1)
            )
        )
        dT2 = self.params["gamma"].isel(ensmem=ens) / c2 * (T1 - T2)

        return [dT1, dT2]

    def solve_model(self, forcing=None, tstep=1):
        """Solve the two-layer model of Held et al. (2010)

        Args:
            forcing (float, np.array, optional): radiative forcing. Can be constant or time-varying. Defaults to None.
            tstep (int, optional): time step in units of years. Defaults to 1.

        Returns:
            np.array: time vector
            np.array: shallow temperature
            np.array: deep temperature
        """
        
        if self.verbose:
            print("--Solving model")
            print(f"--Forcing type: {type(forcing)}")
        
        enssize = self.params["C"].ensmem.size
        
        for ens in range(enssize):
            
            if type(forcing) in [list, xr.DataArray]:
                T1 = np.array([])
                T2 = np.array([])
                tvec = np.array([])

                Nyrs = len(forcing)
                time = np.arange(Nyrs)

                for n in range(Nyrs - 1):
                    R = forcing[n]

                    t = np.arange(time[n], time[n + 1], 0.1)

                    if n == 0:
                        init_state = [0, 0]
                    else:
                        init_state = [stateF[-1, 0], stateF[-1, 1]]

                    stateF = odeint(
                        self.__twolayersystem, 
                        init_state, 
                        t, 
                        args=(R, tstep, ens))

                    T1 = np.append(T1, stateF[:, 0])
                    T2 = np.append(T2, stateF[:, 1])

                    tvec = np.append(tvec, t)
                    
                if ens == 0:
                    T1out = np.zeros((enssize, len(tvec)))
                    T2out = np.zeros((enssize, len(tvec)))
                    tvecout = np.zeros(len(tvec))
                    
                T1out[ens, :] = T1
                T2out[ens, :] = T2
                tvecout[:] = tvec
                
            else:
                init_state = [0, 0]
                tvec = np.arange(0, 150, 0.1)
                
                F = self.params['F_ref'].isel(ensmem=ens)
                
                if self.verbose:
                    print(f"--F = {F}")

                state = odeint(
                    self.__twolayersystem,
                    init_state,
                    tvec,
                    args=(self.params["F_ref"].isel(ensmem=ens), tstep, ens),
                )

                T1 = state[:, 0]
                T2 = state[:, 1]
                
                if ens == 0:
                    T1out = np.zeros((enssize, len(tvec)))
                    T2out = np.zeros((enssize, len(tvec)))
                    tvecout = np.zeros(len(tvec))
                    
                T1out[ens, :] = T1
                T2out[ens, :] = T2
                tvecout[:] = tvec
                
        xT1 = xr.DataArray(
            T1out, 
            dims=("ensmem", "time"),
            coords={"ensmem": ("ensmem", self.params["lam"].ensmem.data),
                    "time": ("time", tvecout)}
        )
        
        xT2 = xr.DataArray(
            T2out, 
            dims=("ensmem", "time"),
            coords={"ensmem": ("ensmem", self.params["lam"].ensmem.data),
                    "time": ("time", tvecout)}
        )

        return xT1, xT2

    def fit_lambda_sw(self, T, SW):
        slope, intcpt = xregress(T, SW)

        self.params["lam_sw"] = -slope.data
