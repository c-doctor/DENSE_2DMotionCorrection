import numpy as np
import random
import math
import os
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from scipy import stats 
import cv2
import scipy.optimize as so
from operator import attrgetter
# okay so we want very small displacements, less than 2 mm
# translational shifts would just be a global addition or subtraction of phase in the different encoding directions (or base directions x,y,z), 
#       and would be ' ? mm * 2.02 (mm/cycle) = ? cycles ' then either ' ? cycles * 360 degrees = ? degree phase shift ' 
#       or (more likely) ' ? cycles * 2pi = ? radians ' (since our phase data is in radians for most of the dense_combine/process codes)
i = 1j
pi = math.pi

def convert_deg_to_rad(angle_deg):
    angle_rad = (angle_deg/180)*pi
    return angle_rad

def convert_rad_to_deg(angle_rad):
    angle_deg = (angle_rad/pi)*180
    return angle_deg

def convert_mm_to_phase(meas_mm):
    meas_phase =  meas_mm*2.02*(2*pi)
    return meas_phase

def convert_phase_to_mm(meas_phase):
    meas_mm = (meas_phase/2*pi)/2.02
    return meas_mm

def convert_rad_to_cycles(meas_phase):
    meas_cycles = (meas_phase/2*pi)
    return meas_cycles


def rmssd(data):
    """Calculates the RMSSD of a given time series data."""
    differences = np.diff(data)
    squared_differences = differences ** 2
    mean_squared_difference = np.nanmean(squared_differences)
    return np.sqrt(mean_squared_difference)


def poly_regression2D(full_data_to_fit,data_mask,poly_degree):
    if (full_data_to_fit.shape) == (data_mask.shape):
        data_points = full_data_to_fit[data_mask]
        size_x = full_data_to_fit.shape[0]
        size_y = full_data_to_fit.shape[1]
        x_coord_array = np.full((size_x,size_y),0)
        y_coord_array = np.full((size_x,size_y),0)
        for a in range(size_x):
            x_coord_array[a,:] = range(size_x)
        for b in range(size_y):
            y_coord_array[:,b] = range(size_y)
        poly = PolynomialFeatures(degree=poly_degree,include_bias=True)
        poly_coord_array = np.column_stack((poly.fit_transform(x_coord_array.reshape(-1,1)),poly.fit_transform(y_coord_array.reshape(-1,1))))
        poly_features = np.column_stack((poly.fit_transform((x_coord_array[data_mask]).reshape(-1,1)),poly.fit_transform((y_coord_array[data_mask]).reshape(-1,1))))
        model = LinearRegression()
        model.fit(poly_features,data_points)
        poly_fit = model.predict(poly_coord_array).reshape(64, 64)
        return poly_fit, model.coef_, model.intercept_



# SECTION WITH NUMBA
import numba as nb
from numba import prange

# ~ functions with numba ~
@nb.jit(forceobj=True)
def setup2Dcoordarraysj(N): 
    x_coord_array = np.full((N,N),0)
    y_coord_array = np.full((N,N),0)
    for a in range(N):
        x_coord_array[a,:] = range(N)
        y_coord_array[:,a] = range(N)
    return x_coord_array, y_coord_array

@nb.jit(nopython=True)
def err_model_full2Dj(params, x, y, z):
    interc, slopex, slopey = params
    err = (interc + x*slopex +y*slopey - z + (pi)) % (2*pi) - (pi)
    error = np.dot(err, err)
    return error

@nb.jit(nopython=True)
def err_model2Dj(interc, slopes, x, y, z):
    slopex, slopey = slopes
    err = (interc + x*slopex +y*slopey - z + (pi)) % (2*pi) - (pi)
    error = np.dot(err, err)
    return error

@nb.jit(nopython=True)
def minimize_err_model2Dj(slopes, x, y, z):
    interc_it_number = 75 #100
    intercepts = np.full(slopes.shape[1],np.nan)
    err = np.full(slopes.shape[1],np.nan)
    for num in prange(slopes.shape[1]):
        sl = slopes[:,num]
        slopex, slopey = sl
        interc_list = np.linspace(-(2*pi), (2*pi), interc_it_number)
        error_list = np.full((interc_it_number),np.nan)
        for interc_indx in range(interc_it_number):
            base_error = (interc_list[interc_indx] + x*slopex +y*slopey - z + (pi)) % (2*pi) - (pi)
            error_squared = np.dot(base_error, base_error)
            error_list[interc_indx] = error_squared
        min_error_indx = np.argmin(error_list)
        intercepts[num] = interc_list[min_error_indx]
        err[num] = error_list[min_error_indx]
    return intercepts, err

@nb.jit(nopython=True)
def getfitj(interc, slopex, slopey, x, y): 
    z = np.full((x.shape),np.nan)
    z = (interc + x*slopex + y*slopey + (pi)) % (2*pi) - (pi)
    return z

@nb.jit(forceobj=True)
def complex_regression2Dj_new(full_data_to_fit,data_mask,num_of_slopes):
    if full_data_to_fit.shape == data_mask.shape:
        if full_data_to_fit.shape[0] == full_data_to_fit.shape[1]:
            N = int(full_data_to_fit.shape[0])
    x_coord_array, y_coord_array = setup2Dcoordarraysj(N)
    # masking the coordinate arrays and the data
    data_to_fit = full_data_to_fit[data_mask]
    x_coords_for_fit = x_coord_array[data_mask]
    y_coords_for_fit = y_coord_array[data_mask]
    # making lists of slopes to iterate through
    total_slopes_list = np.linspace(-1.0, 1.0, num_of_slopes)
    slopes_x = np.full((num_of_slopes,num_of_slopes),np.nan)
    slopes_y = np.full((num_of_slopes,num_of_slopes),np.nan)
    for a in range(num_of_slopes):
        slopes_x[a,:] = total_slopes_list[a]
        slopes_y[:,a] = total_slopes_list[a]
    slopes_x = slopes_x.ravel()
    slopes_y = slopes_y.ravel()
    slopes = np.vstack((slopes_x,slopes_y))
    # finding the initial guess for the intercept with each set of slopes
    intercepts, err = minimize_err_model2Dj(slopes, x_coords_for_fit, y_coords_for_fit, data_to_fit)
    best = np.argmin(err)
    x0 = np.array((intercepts[best], slopes_x[best], slopes_y[best]),dtype="float64")
    res = so.minimize(err_model_full2Dj, x0, args = (x_coords_for_fit, y_coords_for_fit, data_to_fit))
    ic_rec, slx_rec, sly_rec = res.x # these are the 'final' fit parameters
    err_rec_2 = res.fun
    z_rec = getfitj(ic_rec, slx_rec, sly_rec, x_coord_array, y_coord_array) # calculating the final full fit
    return z_rec, ic_rec, slx_rec, sly_rec



print('hello from the functions file! :)')

def find_RSS(x_data, y_data):
    reg_xy = stats.linregress(x_data,y_data)
    obs_values = y_data
    pred_values = ((reg_xy.slope)*x_data)+reg_xy.intercept
    residuals = obs_values - pred_values
    RSS = np.sum(residuals**2)
    return RSS, residuals

def find_RSS_noregression(x_data, y_data):
    obs_values = y_data
    pred_values = x_data
    residuals = obs_values - pred_values
    RSS = np.sum(residuals**2)
    return RSS, residuals

def linreg(x_data,y_data):
    reg_xy = stats.linregress(x_data,y_data)
    R2 = (reg_xy.rvalue)**2
    slope = reg_xy.slope
    intercept = reg_xy.intercept
    return reg_xy, R2, slope, intercept



def bland_altman_output(data1, data2, *args, **kwargs):
    data1     = np.asarray(data1)
    data2     = np.asarray(data2)
    mean      = np.mean([data1, data2], axis=0)
    diff      = data1 - data2                   # Difference between data1 and data2
    mean_diff = np.mean(diff)                   # Mean of the difference
    sd        = np.std(diff, axis=0)            # Standard deviation of the difference
    upper_bound = mean_diff + 1.96*sd
    lower_bound = mean_diff - 1.96*sd
    return mean_diff, upper_bound, lower_bound

def bland_altman_plot(data1, data2, *args, **kwargs):
    data1     = np.asarray(data1)
    data2     = np.asarray(data2)
    mean      = np.mean([data1, data2])
    diff      = data1 - data2                   # Difference between data1 and data2
    md        = np.mean(diff)                   # Mean of the difference
    sd        = np.std(diff, axis=0)            # Standard deviation of the difference

    plt.scatter(mean, diff, s=10, *args, **kwargs)
    plt.axhline(md,           color='red',alpha=0.6,linestyle='--',label=('mean = '+str(md)))
    plt.axhline(md + 1.96*sd, color='gray',linestyle='--',label=('+1.96 SD = '+str(md + 1.96*sd)))
    plt.axhline(md - 1.96*sd, color='gray',linestyle='--',label=('-1.96 SD = '+str(md - 1.96*sd)))
    print('mean = '+str(md))
    print('+1.96 SD = '+str(md + 1.96*sd))
    print('-1.96 SD = '+str(md - 1.96*sd))


def mask_erosion(mask, num_pixels_to_erode):
    mask_to_erode = mask.astype('uint8') #confirming it's in the right format for erosion
    if num_pixels_to_erode < 10:
        odd_num_list = np.array(range(3,24,2))
    kernel_number = odd_num_list[(num_pixels_to_erode-1)]
    print('kernel number is: '+str(kernel_number))
    kernel = np.ones((kernel_number,kernel_number), np.uint8)
    eroded_mask = (cv2.erode(mask, kernel)).astype(bool)
    return eroded_mask



