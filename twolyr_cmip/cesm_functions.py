#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  5 13:10:27 2018

@author: andrewpauling
"""

import numpy as np
import xarray as xr
import scipy.interpolate as interp
from xgcm import Grid


def convert_hs2p(dataset, varin, press_levels, psfile=None):
    """
    Function to interpolate height-varying model output variables to desired
    pressure levels and calculate zonal mean and multi-year mean. Takes some
    inspiration from code by Dan Vimont at
    http://www.aos.wisc.edu/~dvimont/matlab/

    Function modified from geomipFunctions.py written by Rick Russotto
    Source at https://zenodo.org/record/1328272#.W5BBnn5lAQ9

    Parameters
    ----------

    dataset : xarray or netCDF4 Dataset
              Contains the data to interpolate, as well as the parameters
              needed for interpolation (ps,bk)

    varin : string
            String containing the name of the variable to output

    press_levels : array or list
                   Pressure levels to interpolate to, in hPa

    psfile (optional) : int
                        File for surface pressure if not in dataset

    Returns
    -------

    interpMat : ndarray
                Field interpolated to the specified pressure levels
    """

    if psfile is None:
        psfile = dataset.copy()

    # Extract the data from the objects as well as the surface pressure

    if isinstance(varin, str):
        datavar = dataset[varin][:].data
    else:
        datavar = varin   # time, lev, lat, lon

    p_surf = psfile['PS'][:].data  # time, lat, lon; Surface pressure (Pa)

    # Calculate pressure at the points where the Data are located.
    # Conversion from hybrid sigma levels to pressure levels
    # based on equation: #P[i,j,k] = a[k]*p0 + b[k]*PS[i,j]
    # But P and PS vary in time whereas a and b don't, so need to use
    # broadcasting.
    hyam = dataset['hyam'][:].data[0, :]   # Dimension: lev
    hybm = dataset['hybm'][:].data[0, :]   # dimension: lev
    pref = dataset['P0'].data[0]       # Reference pressure (Pa)
    akp0 = hyam*pref
    bkpsij = hybm[None, :, None, None]*p_surf[:, None, :, :]
    # Result: 4D array (time, lev, lat, lon)
    print('Shape of 4D array to interpolate:')
    print(np.shape(bkpsij))
    # Divide by 100 to go from Pa to hPa
    pressure_mat = (akp0[None, :, None, None] + bkpsij)/100.

    # Okay, now the hard part:
    # Interpolate the data from the model's native pressure levels to the
    # desired pressure levels. The model's pressure levels vary with
    # time, latitude, and longitude, whereas we need consistent
    # pressure levels for time and zonal averaging. So we have to do a
    # linear interpolation in the vertical coordinate that varies with
    # latitude, longitude and time. But doing this in nested loops
    # is way too slow, so we need to use matrix operations.
    #
    # Further complicating the picture is the fact that sometimes the desired
    # pressure level lies outside the range of the model pressure levels,
    # e.g. due to terrain. To account for this we need to put nans in these
    # places where there is no data, and use nanmean at the end for zonal
    # and time mean.

    # General strategy: only one loop, over the new pressure levels.
    # At each desired pressure level, calculate the difference between the
    # native pressures and the desired pressure, and find the vertical
    # indices of the native pressure closest to the desired pressure above
    # and below. Find the native pressures and the data values corresponding
    # to these indices in order to do the linear interpolation. In order to
    # put nans where there is no data, keep track of indices columns where
    # all native pressures are either above or below the desired pressure,
    # and slot in nans in the appropriate place right before the final
    # interpolation calculation.

    # First: reshape the native pressures and data into 2D arrays,
    # with time/latitude/longitude all on one axis and vertical coordinate
    # on the other axis. This makes it simpler to extract data at the
    # particular vertical index we want (which varies with time, latitude,
    # and longitude) later on.
    # But we will still use the 4D arrays in other parts of the calculation.

    ntime, nlev, nlat, nlon = pressure_mat.shape

    # Make vertical level the last axis so that columns remain intact when
    # matrices are reshaped
    pressure2d = np.swapaxes(pressure_mat, 1, 3)  # Now it's time,lon,lat,lev
    pressure2d = np.reshape(pressure2d, (ntime*nlat*nlon, nlev))
    data2d = np.swapaxes(datavar, 1, 3)
    data2d = np.reshape(data2d, (ntime*nlat*nlon, nlev))

    # Preallocate array to hold interpolated data
    interp_mat = np.empty((pressure_mat.shape[0], press_levels.size,
                           pressure_mat.shape[2], pressure_mat.shape[3]))

    # Now: loop over the desired pressure levels and interpolate data to
    # each one
    for k in range(press_levels.size):

        # This code block: find the upper boundaries of the native pressure
        # and data for interpolation
        pmat_diffpos = pressure_mat - press_levels[k]
        # Result: 4D array of differences between native and desired pressure.
        # Positive values: higher native pressure than the level we're filling
        # Negative values: lower  native pressure than the level we're filling
        # We're only interested in positive values (higher pressure end)
        # for now, so set negative ones to a very high number and then find
        # the index associated with the minimum along the vertical coordinate.
        pmat_diffpos[pmat_diffpos < 0] = 1.e10
        upper_index = np.argmin(pmat_diffpos, axis=1)

        # upperIndex is 3D array in time, lat, lon
        # Next: Record indices where we're trying to interpolate to greater
        # than the maximum native pressure in the column.
        # If this is the case, every value of pressureMatDiffPos in the column
        # will have been set to 1.e10, so the difference between the max and
        # min in the column will be zero.
        # We'll create an array of boolean values that are true if this is one
        # such column. They'll be used later to slot in nans.
        nan_upper_index_bool = np.all(pressure_mat < press_levels[k], axis=1)
        # Now, convert the 3D arrays containing vertical indices of interest
        # and boolean values for nans to 1D vectors
        # switch lat and lon to match reshaped matrices above
        upper_index1d = np.swapaxes(upper_index, 1, 2)
        # Convert to a vector
        upper_index1d = np.reshape(upper_index1d, ntime*nlat*nlon)
        nan_upper_index1d = np.swapaxes(nan_upper_index_bool, 1, 2)
        nan_upper_index1d = np.reshape(nan_upper_index1d, ntime*nlat*nlon)
        # Now, extract the native pressure and data values at the upper bound
        # indices we found
        # (I tested this method for extracting data using a sample 2D array
        # in the command line)
        upper_pressure_bound = pressure2d[np.arange(upper_index1d.size),
                                          upper_index1d]  # Result: 1D vector
        upper_data_bound = data2d[np.arange(upper_index1d.size), upper_index1d]
        # Set the pressure and data boundary data to nans where we are trying
        # to interpolate to outside the data range
        upper_pressure_bound[nan_upper_index1d] = np.nan
        upper_data_bound[nan_upper_index1d] = np.nan

        # This code block: same as above but for the lower boundaries.
        # (far less comments)
        pmat_diffneg = pressure_mat - press_levels[k]
        # This time we are only interested in negative values
        pmat_diffneg[pmat_diffneg > 0] = -1.e10

        lower_index = np.argmax(pmat_diffneg, axis=1)
        # lowerIndex = pressureMatDiffNeg.argmax(axis=1)
        nan_lower_index_bool = np.all(pressure_mat > press_levels[k], axis=1)
        lower_index1d = np.swapaxes(lower_index, 1, 2)
        lower_index1d = np.reshape(lower_index1d, ntime*nlat*nlon)
        nan_lower_index1d = np.swapaxes(nan_lower_index_bool, 1, 2)
        nan_lower_index1d = np.reshape(nan_lower_index1d, ntime*nlat*nlon)
        lower_pressure_bound = pressure2d[np.arange(lower_index1d.size),
                                          lower_index1d]
        lower_data_bound = data2d[np.arange(lower_index1d.size), lower_index1d]
        lower_pressure_bound[nan_lower_index1d] = np.nan
        lower_data_bound[nan_lower_index1d] = np.nan

        # Now: linearly interpolate the data in log pressure space
        # (interpolating in log pressure means interpolating linearly w.r.t.
        # height)
        interp_vec = lower_data_bound + \
            (upper_data_bound-lower_data_bound) * \
            (np.log(press_levels[k]) - np.log(lower_pressure_bound)) / \
            (np.log(upper_pressure_bound)-np.log(lower_pressure_bound))

        # Finally: Reshape to 3D matrix to put in the later 4D matrix
        interp_slice = np.reshape(interp_vec, (ntime, nlon, nlat))
        interp_slice = np.swapaxes(interp_slice, 1, 2)  # switch lat and lon
        interp_mat[:, k, :, :] = interp_slice  # Populate the returned matrix

        # Now we have a 4D matrix of interpolated data in time, pressure,
        # latitude and longitude.

    interp_mat = np.squeeze(interp_mat)

    return interp_mat


def compute_baroclinic_criterion(tfile, ufile):
    """Compute baroclinic instability criterion for CESM output. Taken from
    NCL code by Hansi Singh

    Parameters
    ----------

    tfile : xarray or netCDF4 Dataset
            Dataset containing temperature data

    ufile : string
            Dataset containing wind data

    Returns
    -------

    b_crit : ndarray
             Baroclinic criterion
    """


    # Define pressure levels to interpolate to in hPa
    press = np.array([10., 20., 30., 50., 70., 100., 150., 200., 250.,
                      300., 400., 500., 600., 650., 700., 750., 800., 850.,
                      900., 950., 1000.])

    # Read in coordinate vars
    lat = tfile['lat'][:].data
    lon = tfile['lon'][:].data

    nlat = len(lat)
    nlon = len(lon)

    # Interpolate T and U to pressure levels using code from Rick Russotto
    t_c = convert_hs2p(tfile, 'T', press)
    u_c = convert_hs2p(ufile, 'U', press)

    # Get surface and reference pressure
    p_surf = np.squeeze(tfile['PS'])/100  # hPa
    p_ref = tfile['P0'].data             # Pa

    # Define sigma levels to use
    sig_top = 0.25
    sig_bot = 0.75
    sig_mid1 = 0.55
    sig_mid2 = 0.45

    # Get pressure values at each sigma level
    myp = np.zeros((4, nlat, nlon))
    myp[0, :, :] = p_surf*sig_top
    myp[3, :, :] = p_surf*sig_bot
    myp[2, :, :] = p_surf*sig_mid1
    myp[1, :, :] = p_surf*sig_mid2

    # Initialize matrices for interpolated T and U
    t_top = np.zeros((nlat, nlon))
    t_bot = np.zeros((nlat, nlon))
    t_mid = np.zeros((2, nlat, nlon))

    u_top = np.zeros((nlat, nlon))
    u_bot = np.zeros((nlat, nlon))

    # Loop over grid cells and interpolate T and U at the pressure for each
    # of the sigma levels defined above
    for i in range(nlat):
        for j in range(nlon):
            myt = np.interp(myp[:, i, j], press, t_c[:, i, j])
            t_top[i, j] = myt[0]
            t_bot[i, j] = myt[3]
            t_mid[:, i, j] = myt[1:2]

            myu = np.interp(myp[:, i, j], press, u_c[:, i, j])
            u_top[i, j] = myu[0]
            u_bot[i, j] = myu[3]

    # Define some constants
    omega = 7.3e-5
    r_e = 6.37e6
    degtorad = np.pi/180
    latrad = lat*degtorad
    lat0 = -np.pi/4

    # Compute beta and f
    yval = r_e*(latrad-lat0)
    f_cor = 2*omega*np.sin(latrad)
    f_ref = 2*omega*np.sin(lat0)
    beta0 = (f_cor-f_ref)/yval
    beta = np.tile(beta0[:, None], [1, 288])

    # Use modified beta for sigma coordinates as used in Singh et al. (2015)
    avg_p_surf = np.mean(p_surf.data, axis=1)
    my_lat = -45
    mylat_ind = np.where(np.logical_and(lat < my_lat+0.5,
                                        lat > my_lat-0.5))[0][0]

    p00 = avg_p_surf[mylat_ind]*100
    d_mat = (p_surf.data*100-p00)/np.tile(yval[:, None], [1, 288])
    betap = beta - (f_ref*d_mat/p00)

    # Compute theta at each relevant level
    rcp = 0.286

    theta_midt = t_mid[0, :, :]*(p_ref/myp[1, :, :])**rcp
    theta_midb = t_mid[1, :, :]*(p_ref/myp[2, :, :])**rcp
    theta_mid = (theta_midt+theta_midb)/2

    # Compute S and lambda**2
    dtds = -(theta_midt-theta_midb)/0.1
    alpha = 0.77
    s_p = -alpha*p00*dtds/theta_mid
    lambdap2 = (f_ref/0.5)**2/s_p

    # Compute B
    b_crit = lambdap2*(u_top-u_bot)/betap

    return b_crit


def compute_ep_vectors(rname):
    """Compute EP-Flux vectors and their divergence. Taken from NCL code by
    Hansi Singh

    Parameters
    ----------

    rname : string
            Name of run to use

    Returns
    -------

    f_y : ndarray
          y-component of EP flux

    f_z : ndarray
          z-component of EP flux

    div_f : ndarray
            EP flux divergence
    """

    if rname == 'CCSM41degcont':
        dname = 'CCSMcont/'
        suff = '0071-0100avg.nc'
    elif rname == 'pd700katopo':
        dname = 'pd700ka/'
        suff = '0071-0100avg.nc'
    elif rname == 'pd700ka_halfdiff':
        dname = 'pdhalfdiff/'
        suff = '0071-0100avg.nc'
    elif rname == 'pd700ka_negdiff':
        dname = 'pdnegdiff/'
        suff = '0071-0100avg.nc'
    elif rname == 'SOMcontrol':
        dname = 'som/'
        suff = '0031-0060avg.nc'
    elif rname == 'FlatWAIS_0.9h_SOM':
        dname = 'som/'
        suff = '0031-0060avg.nc'
    elif rname == 'TallWAIS_0.9h_SOM':
        dname = 'som/'
        suff = '0031-0060avg.nc'

    ddir = '/Users/andrewpauling/Documents/PhD/isotope/data/' + dname

    dfilevuc = rname + '.cam.h0.VU.' + suff
    dfileuvc = rname + '.cam.h0.UV.' + suff
    dfilez3c = rname + '.cam.h0.Z3.' + suff
    dfilevtc = rname + '.cam.h0.VTvars.' + suff
    dfiletc = rname + '.cam.h0.T.' + suff

    nc_vu = ddir + dfilevuc
    nc_uv = ddir + dfileuvc
    nc_z3 = ddir + dfilez3c
    nc_vt = ddir + dfilevtc
    nc_t = ddir + dfiletc

    vu_file = xr.open_dataset(nc_vu, decode_times=False)
    uv_file = xr.open_dataset(nc_uv, decode_times=False)
    z3_file = xr.open_dataset(nc_z3, decode_times=False)
    vt_file = xr.open_dataset(nc_vt, decode_times=False)
    t_file = xr.open_dataset(nc_t, decode_times=False)

    lat = vu_file['lat'][:].data
    # lon = vu_file['lon'][:].data
    p_ref = vu_file['P0'].data/100

    # nlon = len(lon)
    nlat = len(lat)
    dlat = lat[1] - lat[0]

    # Load variables and do coordinate conversion
    press = np.array([10., 20., 30., 50., 70., 100., 150., 200., 250.,
                      300., 400., 500., 600., 650., 700., 750., 800., 850.,
                      900., 950., 1000.])
    nlev = len(press)

    u_c = convert_hs2p(uv_file, 'U', press)
    v_c = convert_hs2p(uv_file, 'V', press)
    vu_c = convert_hs2p(vu_file, 'VU', press)
    vt_c = convert_hs2p(vt_file, 'VT', press)
    z3_c = convert_hs2p(z3_file, 'Z3', press)
    t_c = convert_hs2p(t_file, 'T', press)

    # Get perturbation momentum flux
    upvp_c = vu_c - v_c*u_c

    # Get perturbation heat flux
    vptp_c = vt_c - v_c*t_c

    # Take zonal means
    u_c = np.nanmean(u_c, axis=2)
    v_c = np.nanmean(v_c, axis=2)
    upvp_c = np.nanmean(upvp_c, axis=2)
    vptp_c = np.nanmean(vptp_c, axis=2)
    z3_c = np.nanmean(z3_c, axis=2)
    t_c = np.nanmean(t_c, axis=2)

    press_big = np.tile(press[:, None], (1, nlat))

    # calculate potential temperature
    rdcp = 0.286
    tpot_c = t_c*(p_ref/press_big)**rdcp

    # Flip z3 and Tpot so that height is increasing
    z3_c_flip = np.flip(z3_c, axis=0)
    tpot_c_flip = np.flip(tpot_c, axis=0)

    # Compute potential temperature gradient
    dthdz = np.nan*np.zeros((tpot_c.shape))

    for xlat in range(nlat):
        gind = np.where(np.isnan(z3_c_flip[:, xlat]) == 0)
        z3_tmp = np.squeeze(z3_c_flip[gind, xlat])
        tpot_tmp = np.squeeze(tpot_c_flip[gind, xlat])

        dthdz[gind, xlat] = interp.CubicSpline(z3_tmp,
                                               tpot_tmp).derivative()(z3_tmp)

    dthdz = np.flip(dthdz, axis=0)
    grav = 9.81

    # Compute Brunt-Vaisala frequency
    n_2 = grav*dthdz/tpot_c
    print('Minimum N2 = ' + str(np.nanmin(n_2)))
    print('Maximum N2 = ' + str(np.nanmax(n_2)))

    r_e = 6.371e6
    degtorad = np.pi/180
    coslat = np.cos(lat*degtorad)
    coslat_big = np.tile(coslat[None, :], (nlev, 1))
    lat_std = -45
    omega = 7.292e-5
    f_ref = omega*np.sin(lat_std*degtorad)
    r_gas = 287
    h_scale = 8.5e3

    f_y = -coslat_big*r_e*upvp_c
    f_z = f_ref*r_gas*coslat_big*r_e*vptp_c/(n_2*h_scale)

    div_f = np.zeros(f_z.shape)

    for j in np.arange(1, nlat-1):
        for i in np.arange(1, nlev-1):
            delz = z3_c[i-1, j] - z3_c[i+1, j]
            div_f[i, j] = (f_z[i-1, j]-f_z[i+1, j])/delz + \
                (f_y[i, j+1]-f_y[i, j-1])/dlat

    return f_y, f_z, div_f


def fixmonth(dfile):

    """Fix CESM months since by default the timestamp is for the first day of
    the next month

    Parameters
    ----------

    dfile : xarray dataset
            Dataset containing time to fix

    Returns
    -------

    dfile : xarray dataset
            Fixed dataset
    """

    mytime = dfile['time'][:].data
    for time in range(mytime.size):
        if mytime[time].month > 1:
            mytime[time] = mytime[time].replace(month=mytime[time].month-1)
        elif mytime[time].month == 1:
            mytime[time] = mytime[time].replace(month=12)
            mytime[time] = mytime[time].replace(year=mytime[time].year-1)

    dfile = dfile.assign_coords(time=mytime)

    return dfile

def pop_add_cyclic(ds):
    
    nj = ds.TLAT.shape[0]
    ni = ds.TLONG.shape[1]

    xL = int(ni/2 - 1)
    xR = int(xL + ni)

    tlon = ds.TLONG.data
    tlat = ds.TLAT.data
    
    tlon = np.where(np.greater_equal(tlon, min(tlon[:,0])), tlon-360., tlon)    
    lon  = np.concatenate((tlon, tlon + 360.), 1)
    lon = lon[:, xL:xR]

    if ni == 320:
        lon[367:-3, 0] = lon[367:-3, 0] + 360.        
    lon = lon - 360.
    
    lon = np.hstack((lon, lon[:, 0:1] + 360.))
    if ni == 320:
        lon[367:, -1] = lon[367:, -1] - 360.

    #-- trick cartopy into doing the right thing:
    #   it gets confused when the cyclic coords are identical
    lon[:, 0] = lon[:, 0] - 1e-8

    #-- periodicity
    lat = np.concatenate((tlat, tlat), 1)
    lat = lat[:, xL:xR]
    lat = np.hstack((lat, lat[:,0:1]))

    TLAT = xr.DataArray(lat, dims=('nlat', 'nlon'))
    TLONG = xr.DataArray(lon, dims=('nlat', 'nlon'))
    
    dso = xr.Dataset({'TLAT': TLAT, 'TLONG': TLONG})

    # copy vars
    varlist = [v for v in ds.data_vars if v not in ['TLAT', 'TLONG']]
    for v in varlist:
        v_dims = ds[v].dims
        if not ('nlat' in v_dims and 'nlon' in v_dims):
            dso[v] = ds[v]
        else:
            # determine and sort other dimensions
            other_dims = set(v_dims) - {'nlat', 'nlon'}
            other_dims = tuple([d for d in v_dims if d in other_dims])
            lon_dim = ds[v].dims.index('nlon')
            field = ds[v].data
            field = np.concatenate((field, field), lon_dim)
            field = field[..., :, xL:xR]
            field = np.concatenate((field, field[..., :, 0:1]), lon_dim)       
            dso[v] = xr.DataArray(field, dims=other_dims+('nlat', 'nlon'), 
                                  attrs=ds[v].attrs)


    # copy coords
    for v, da in ds.coords.items():
        if not ('nlat' in da.dims and 'nlon' in da.dims):
            dso = dso.assign_coords(**{v: da})
                
            
    return dso


def regrid_xgcm(ds, dp_sigma, ps, p_target):
    """
    Regrid data from hybrid-sigma coordinates to constant pressure levels using xgcm
    Uses xgcm.Grid.transform
    
    Parameters
    ----------
    ds: xarray Dataset
        A dataset containing the variables to be regridded
    dp_sigma: xarray DataArray
        DataArray containing thickness of hybrid-sigma levels in hPa
    ps: xarray DataArray
        DataArray containing surface pressure data
    p_target: array-like
        Array containing pressure levels in hPa to interpolate to
        
    Returns
    -------
    dsout: xarray Dataset
        A dataset containing the variables regridded onto pressure levels        
    
    """
    
    p = (ds['hyam']*ds['P0'] + ds['hybm']*ps)/100
    ds = ds.assign({'p': np.log(p)})
    grid = Grid(ds, coords={'Z': {'center': 'lev'}}, periodic=False)
    
    dsout = xr.Dataset(coords={"time": ("time", ds.time.data),
                              "plev": ("plev", p_target),
                              "lat": ("lat", ds.lat.data),
                              "lon": ("lon", ds.lon.data)})
    
    for var in ds.data_vars:
        if var in ["Q", "T"] or var[0] == "F":
            if var[0] == "F":
                data = ds[var]/dp_sigma
            else:
                data = ds[var]
            varout = grid.transform(data,
                                    'Z',
                                    np.log(p_target),
                                    target_data=ds.p)
            varout = varout.rename({"p": "plev"})
            varout = varout.assign_coords({'plev': p_target})
        else:
            varout = ds[var]
        
        dsout = dsout.assign({var: varout})
    return dsout


def calcsatspechum(t, p):
    """
    Compute saturateed specific humidity
    T is temperature, P is pressure in hPa
    """

    ## Formulae from Buck (1981):
    es = (1.0007+(3.46e-6*p))*6.1121*np.exp(17.502*(t-273.15)/(240.97+(t-273.15)))
    wsl = .622*es/(p-es) # saturation mixing ratio wrt liquid water (g/kg)

    es = (1.0003+(4.18e-6*p))*6.1115*np.exp(22.452*(t-273.15)/(272.55+(t-273.15)))

    wsi = .622*es/(p-es) # saturation mixing ratio wrt ice (g/kg)

    ws = wsl
    ws = ws.where(t>=273.15, wsi)

    qs=ws/(1+ws) # saturation specific humidity, g/kg
    
    return qs

def globalmean(da):
    return da.weighted(np.cos(np.deg2rad(da.lat))).mean(("lat", "lon"))
