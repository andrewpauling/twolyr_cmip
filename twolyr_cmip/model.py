import xarray as xr
import numpy as np
import os
from glob import glob
from shutil import copy
from datetime import datetime
from twolyr_cmip.util import DATADIR

import warnings

warnings.filterwarnings("ignore")


class Model:
    def __init__(self, name, experiment, variable, frequency="mon"):

        valid_vars = self.__get_downloaded_vars()
        valid_models = self.__get_downloaded_models()
        valid_experiments = self.__get_downloaded_experiments()

        if isinstance(name, str):
            if name in valid_models:
                self.name = name
            else:
                raise ValueError(
                    f"Model '{name}' is either invalid or has not been downloaded yet"
                )
        else:
            raise TypeError(
                f"Parameter 'name' must be of type 'str' not type {type(name)}"
            )

        if isinstance(experiment, str):
            if experiment in valid_experiments:
                self.experiment = experiment
            else:
                raise ValueError(
                    f" Experiment '{experiment}' is either invalid or has not been downloaded yet"
                )
        else:
            raise TypeError(f"Parameter 'experiment' must be of type 'str'")

        if isinstance(variable, str):
            if variable in valid_vars:
                self.variable = variable
            else:
                raise ValueError(
                    f"Variable '{variable}' is either invalid or has not been downloaded yet"
                )
        else:
            raise TypeError(f"Parameter 'variable' must be of type 'str'")

        if isinstance(frequency, str):
            self.frequency = frequency
        else:
            raise TypeError(f"Parameter 'frequency' must be of type 'str'")

        self.data_path = self._specify_data_path(name, experiment, variable, frequency)
        self.raw_file_list = self._get_file_list()
        self.num_files = len(self.raw_file_list)

        self.ensemble_members = self._get_ensemble_members(self.raw_file_list)
        self.num_members = len(self.ensemble_members)

        self.IS_PROCESSED = False
        self.processed_files = []
        self._unprocessed_members = []
        self.__check_processed_data_status()
        
        if self.num_files == 0 and self.IS_PROCESSED == False:
            raise ValueError(f"No files present at path {self.data_path}")

    def __repr__(self):
        summary = [f"<model.{type(self).__name__}>"]
        summary.append(f"Model: '{self.name}'")
        summary.append(f"Experiment: '{self.experiment}'")
        summary.append(f"Variable: '{self.variable}'")
        summary.append(f"Frequency: '{self.frequency}'")
        summary.append(f"Data Location: '{self.data_path}'")
        summary.append(f"Number of files: {self.num_files}")
        summary.append(f"Number of ensemble members: {self.num_members}")
        return "\n".join(summary)

    def __get_downloaded_vars(self):
        ddirs = sorted(glob(os.path.join(DATADIR, "v_*")))
        varlist = []
        for ddir in ddirs:
            varlist.append(ddir.split("/")[-1].split("_")[1])

        return set(varlist)

    def __get_downloaded_models(self):
        ddirs = sorted(glob(os.path.join(DATADIR, "v_*")))
        modlist = []
        for ddir in ddirs:
            modlist.append(ddir.split("/")[-1].split("_")[-1])
        return set(modlist)

    def __get_downloaded_experiments(self):
        ddirs = sorted(glob(os.path.join(DATADIR, "v_*")))
        explist = []
        for ddir in ddirs:
            explist.append(ddir.split("/")[-1].split("_")[5])

        return set(explist)

    def _specify_data_path(self, nam, exp, var, freq):
        filename = "v_" + var + "_f_" + freq + "_e_" + exp + "_s_" + nam
        return os.path.join(DATADIR, filename)

    def _get_file_list(self):
        return sorted(glob(os.path.join(self.data_path, "*.nc")))

    def _get_ensemble_members(self, file_list):
        enslist = []
        for file in file_list:
            enslist.append(file.split("/")[-1].split("_")[4])
        return set(enslist)

    def process_data(self):
        if not self.IS_PROCESSED:
            self._process_data()

    def _process_data(self):

        if self._unprocessed_members:
            members = self._unprocessed_members
        else:
            members = self.ensemble_members

        for ens in members:
            print(f"Processing ensemble member {ens}")
            files = sorted(glob(os.path.join(self.data_path, "*" + ens + "*.nc")))

            pathout = os.path.join(self.data_path, "processed")
            if not os.path.isdir(pathout):
                os.mkdir(pathout)

            if len(files) > 1:
                start_year = files[0].split("/")[-1].split("_")[-1][:6]
                end_year = files[-1].split("/")[-1].split("_")[-1][7:13]
                namestart = "_".join(files[0].split("/")[-1].split("_")[:-1])
                nameout = namestart + "_" + start_year + "-" + end_year + ".nc"

                dout = os.path.join(pathout, nameout)

                print(dout)
                mfds = xr.open_mfdataset(files, combine="by_coords", parallel=True, use_cftime=True)
                mfds.to_netcdf(dout)
            else:
                fname = files[0].split("/")[-1]
                dest = os.path.join(self.data_path, "processed", fname)
                print(dest)
                copy(files[0], dest)

        print(f"Processing complete for model {self.name}")
        self.IS_PROCESSED = True

    def __check_processed_data_status(self):
        process_path = os.path.join(self.data_path, "processed")

        if not os.path.isdir(process_path):
            print(f"Data for model {self.name} not yet processed")
        elif not os.listdir(process_path):
            print(f"Data for model {self.name} not yet processed")
        else:
            self.processed_files = sorted(glob(os.path.join(process_path, "*.nc")))
            enstmp = []
            for file in self.processed_files:
                enstmp.append(file.split("/")[-1].split("_")[4])
            processed_ens = set(enstmp)

            self._unprocessed_members = list(self.ensemble_members - processed_ens)

            if self._unprocessed_members:
                print(f"Members {self._unprocessed_members} not yet processed")
                self.IS_PROCESSED = False
            else:
                self.IS_PROCESSED = True
                self.ensemble_members = self._get_ensemble_members(self.processed_files)
                self.num_members = len(self.ensemble_members)

    def load_ensemble(self):
        if self.experiment == "historical":
            mfds = xr.open_mfdataset(
                self.processed_files,
                combine="nested",
                concat_dim="ensmem",
                coords="minimal",
                preprocess=preprocess,
                parallel=True,
                compat="override"
            )
        else:
            mfds = xr.open_mfdataset(
                self.processed_files,
                combine="nested",
                concat_dim="ensmem",
                parallel=True,
                use_cftime=True)
            
        mfds = mfds.assign_coords({"ensmem": sorted(list(self.ensemble_members))})

        self.dataset = mfds

    def time_subset(self, start_time, end_time):
        self.dataset = self.dataset.sel(time=slice(start_time, end_time))


def preprocess(ds):
    time = ds.time.data
    if type(time[0]) != np.datetime64:
        newtime = [np.datetime64(t) for t in time]

        ds = ds.assign_coords({"time": newtime})
        
    if ds.attrs["source_id"] == "EC-Earth3":
        ds = ds.drop("lat")
    return ds

def preprocess2(ds):
    return ds.drop("lat")

