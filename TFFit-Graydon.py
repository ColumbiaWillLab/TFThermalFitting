from numpy import sum, power, array, pi, exp, subtract, divide, argmin, argmax, log, mean, linspace, round, absolute, sqrt, maximum, zeros, diag, inf
from numpy.fft import fft2, ifft2, fftshift
import matplotlib.pyplot as plt
from matplotlib.image import imread
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter, fourier_ellipsoid
from os import listdir
from os.path import isfile, join
from time import time

# constants of the universe
mu_0 = 4 * pi * 10.**-7
hbar = 1.0545718 * 10.**-34
c = 299792458
mu_b = hbar * 2 * pi * 1.39962460 * 10.**6
k_b = 1.38 * 10**-23

# sodium constants
Isat = 6.26 * 10
Gamma = 2 * pi * 9.7946 * 10.**6
f0 = 508.8487162 * 10.**12
k = 2 * pi * f0 / c
m = 22.989769 * 1.672623 * 10**-27

# Experiment constants
pixel = 0.00375

def sigma(v, sigma_0, t):
    return sigma_0 + v * t

def get_Directory(mypath):
    return [f for f in listdir(mypath) if isfile(join(mypath, f))]

def group_names(list_names):
    names = []
    for i in range(int(len(list_names)/3)):
        names.append([list_names[3*i], list_names[3*i+1], list_names[3*i+2]])
    return names

def transmission(directory, names):
    data, laser, dark = imread(directory + '/' + names[0]), imread(directory + '/' + names[1]), imread(directory + '/' + names[2])
    data, laser, dark = data[:,:,0].astype('float'), laser[:,:,0].astype('float'), dark[:,:,0].astype('float')
    laser = gaussian_filter(laser, sigma = 3)
    data = gaussian_filter(data, sigma = 3)
    atoms = subtract(data, dark)
    light = subtract(laser, dark)
    threshold = 7
    t = divide(atoms, light, where = light > threshold)
    t[light <= threshold] = 1
    t[t > 1] = 1
    return t

def find_center(image):
    x_project = sum(image, 0)
    y_project = sum(image, 1)
    return argmin(x_project), argmin(y_project)

def log_integrate_1D(image):
    """
    takes -log and then takes mean of each axis to obtain normalised 1D distributions
    also ensures image values are not zero (log0 = -inf) by setting those to the minimum positive value (1/255)
    
    params:
        image: image array
        
    returns:
        x_1D, y_1D: 1D projections to each axis of the image
    """
    for i in range(len(image)):
        for j in range(len(image[i])):
            if image[i][j] == 0:
                image[i][j] = 1 / 255
    log_image = -log(image)
    x_1D = mean(log_image, 0)
    y_1D = mean(log_image, 1)
    return x_1D, y_1D

def AOI_crop(image, center, widths):
    if widths[0] < 250:
        widths[0] = 250
    if widths[1] < 250:
        widths[1] = 250
    
    x1, x2, y1, y2 = center[0] - widths[0] / 2, center[0] + widths[0] / 2, center[1] - widths[1] / 2, center[1] + widths[1] / 2
    
    if x1 < 0:
        x1 = 0
    if y1 < 0:
        y1 = 0
    x_max, y_max = image.shape
    if x2 > x_max:
        x2 = x_max
    if y2 > y_max:
        y2 = y_max
        
    return image[int(y1):int(y2),int(x1):int(x2)]

def AOI_integration(image, center, widths, detuning, mag, v = 'no'):
        
    cropped_image = gaussian_filter(AOI_crop(image, center, widths), sigma = 3)
    
    s1 = -sum(sum(log(cropped_image), 0), 0)
    sigma = ( 3 * (2 * pi / k)**2 / (2 * pi) ) / (1 + (2 * detuning * 2 * pi * 10.**6 / Gamma)**2)
    Area = (3.75 * mag * 10.**-6)**2
    
    if v == 'yes':
        if widths[0] < 250:
            widths[0] = 250
        if widths[1] < 250:
            widths[1] = 250
        
        x1, x2, y1, y2 = center[0] - widths[0] / 2, center[0] + widths[0] / 2, center[1] - widths[1] / 2, center[1] + widths[1] / 2
        plot_cropped_image(image, cropped_image, x1, x2, y1, y2)
    
    return s1 * 10.**-6 * Area / sigma 

def integration(image, detuning, mag):
        
    cropped_image = gaussian_filter(image, sigma = 3)
    
    s1 = -sum(sum(log(cropped_image), 0), 0)
    sigma = ( 3 * (2 * pi / k)**2 / (2 * pi) ) / (1 + (2 * detuning * 2 * pi * 10.**6 / Gamma)**2)
    Area = (3.75 * mag * 10.**-6)**2
    
    return s1 * 10.**-6 * Area / sigma

def gaussian_x(x, A, sigma_0, h, x0):
    return A * exp( - power( (x - x0)/sigma_0 , 2) / 2 ) + h

def gaussian_no_h(x, A, sigma_0, x0):
    return A * exp( - power( (x - x0)/sigma_0 , 2) / 2 )

def fit_1D_gaussians(image, center, no_h = 'no'):
    x_project, y_project = log_integrate_1D(image) ### changed
    
    xs = list(range(len(x_project)))
    ys = list(range(len(y_project)))
    
    if no_h == 'yes':
        popt_x, pcov_x = curve_fit(gaussian_no_h, xs, x_project, p0 = [1, 100, center[0]])
        popt_y, pcov_y = curve_fit(gaussian_no_h, ys, y_project, p0 = [1, 100, center[1]])
        
        A_x, sigma_x, x0 = popt_x
        error_sigma_x = sqrt(pcov_x[1,1]) ### added sqrt
        A_y, sigma_y, y0 = popt_y
        error_sigma_y = sqrt(pcov_y[1,1])
        
        h_x, h_y = 0, 0
    else:
        popt_x, pcov_x = curve_fit(gaussian_x, xs, x_project, p0 = [1, 100, 0, center[0]])
        popt_y, pcov_y = curve_fit(gaussian_x, ys, y_project, p0 = [1, 100, 0, center[1]])
        
        A_x, sigma_x, h_x, x0 = popt_x
        error_sigma_x = sqrt(pcov_x[1,1])
        A_y, sigma_y, h_y, y0 = popt_y
        error_sigma_y = sqrt(pcov_y[1,1])
    
    return A_x, sigma_x, h_x, x0, error_sigma_x, A_y, sigma_y, h_y, y0, error_sigma_y

def plot_cropped_image(image, cropped_image, x1, x2, y1, y2):
    fig, (ax1, ax2) = plt.subplots(2)
    ax1.imshow(cropped_image)
    ax2.imshow(image)
    
    ax2.plot([x1,x1], [y1,y2])
    ax2.plot([x1,x2], [y1,y1])
    ax2.plot([x2,x2], [y1,y2])
    ax2.plot([x1,x2], [y2,y2])
    
    plt.show()    

def plot_1D_fits(atoms, center, A_x, sigma_x, h_x, x0_x, A_y, sigma_y, h_y, x0_y):
    atoms = -log(atoms)
    fig, (ax1, ax2, ax3) = plt.subplots(3)
    fig.set_size_inches(15, 15, forward=True)
    
    ax1.imshow(atoms)
    
    x_project = mean(atoms, 0)
    x = range(len(x_project))
    x_fit = gaussian_x(x, A_x, sigma_x, h_x, x0_x)
    ax2.plot(x, x_project)
    ax2.plot(x, x_fit)
    
    y_project = mean(atoms, 1)
    y = range(len(y_project))
    y_fit = gaussian_x(y, A_y, sigma_y, h_y, x0_y)
    ax3.plot(y, y_project)
    ax3.plot(y, y_fit)
    
    plt.show()

def TF_1D(x, R, x0, TF0, h):
    """
    1D Thomas-Fermi function (ref: Atomic Gas Density Profile, David Chen)
    
    params:
        x: x position
        R: "radius" of the parabola at zero
        x0: centre of the parabola
        TF0: maximum height
        h: vertical offset
        
    returns: value of Thomas-Fermi function at x = x
    """
    return TF0 * power(maximum(0, 1 - power((x - x0) / R, 2)), 2) + h

def TF_1D_no_h(x, R, x0, TF0):
    """
    1D Thomas-Fermi function with no offset
    
    params:
        x: x position
        R: "radius" of the parabola at zero
        x0: centre of the parabola
        TF0: maximum height
        
    returns: value of Thomas-Fermi function at x = x
    """
    return TF0 * power(maximum(0, 1 - power((x - x0) / R, 2)), 2)

def TFG_1D(x, R, sigma_0, x0, TF0, A, h):
    """
    sum of Thomas-Fermi and Gaussian functions
    
    params:
        x: x position
        R, sigma_0, x0, TF0, A, h: same as in gaussian_x() and TF_1D()
        
    returns:
        value of this sum at x = x
    """
    return TF0 * power(maximum(0, 1 - power((x - x0) / R, 2)), 2) + A * exp(- power((x - x0) / sigma_0, 2) / 2) + h

def rms_width(project):
    """
    finds approximate RMS width of function (1/sqrt(2) max height)
    
    params:
        project: 1D distribution
        
    returns:
        width_guess: approximate RMS width of bell-shaped function
    """
    x0_guess = argmax(project)
    rms_height = max(project) / sqrt(2)
    left_guess = (absolute(project[0:x0_guess] - rms_height)).argmin()
    right_guess = (absolute(project[x0_guess + 1:len(project)] - rms_height)).argmin() + x0_guess + 1
    width_guess = int(mean([absolute(x0_guess - left_guess), absolute(right_guess - x0_guess)]))
    return width_guess

def prelim_fit(project):
    """
    fits 1D projection to the sum of TF and Gaussian function TFG_1D() in order to extract rough parameters
    
    fitting guess, (lower bound, upper bound):
        R: rms_width() value, (0, length of project)
        sigma_0: rms_width() value, (0, length of project)
        x0: index of maximum height, (guess - project length, guess + project length)
        TF0: maximum height / 2, (0, maximum height)
        A: maximum height / 2, (0, maximum height)
        h: 0, (0, infinity)
    
    params:
        project: 1D distribution
        
    returns:
        popt: array of fitting parameters in the order R, sigma_0, x0, TF0, A, h
        pcov: covariance array of popt
        peak_max: maximum height of project (used to limit fitting height)
    """
    x0_guess = argmax(project)
    width_guess = rms_width(project)
    peak_max = max(project)# * 2
    peak_guess = peak_max / 2
    xs = list(range(len(project)))
    popt, pcov = curve_fit(TFG_1D, xs, project, p0 = (width_guess, width_guess, x0_guess, peak_guess, peak_guess, 0), bounds = ((0, 0, x0_guess - len(project), 0, 0, 0), (len(project), len(project), x0_guess + len(project), peak_max, peak_max, inf)))
    return popt, pcov, peak_max

def subtract_BEC(project, R, x0, h, a):
    """
    removes data points (except for fit offset h) within a * R of the centre
    ensures there is at least one data point in each wing to prevent errors when fitting Gaussian (although this ideally does not occur)
    
    params:
        project: 1D distribution
        R, x0, h: parameters returned from initial prelim_fit()
        a: determines wings from fit R to remove distorted density region (ref: Szczepkowski et al. 2008), a = 1.1 often works
        
    returns:
        no_BEC: 1D distribution with the data in the centre set to h (roughly zero)
        left_bound, right_bounds: bounds to define the "centre" (BEC) and the "wings" (thermal cloud) - the centre includes these bounds
    """
    no_BEC = zeros(len(project))
    left_bound = max(x0 - int(absolute(R * a)), 1)
    right_bound = min(x0 + int(absolute(R * a)), len(project) - 2)
    for i in range(len(project)):
        if i < left_bound:
            no_BEC[i] = project[i]
        elif i <= right_bound:
            no_BEC[i] = h
        else:
            no_BEC[i] = project[i]
    return no_BEC, left_bound, right_bound

def thermal_only_fit(no_BEC, left_bound, right_bound, peak_max, sigma_0, x0, A, h):
    """
    fits outside wings to "1D" Gaussian and returns arrays of only the outside data points
    
    fitting guess, (lower bound, upper bound):
        A: A obtained from prelim_fit(), (0, maximum height)
        sigma_0: sigma_0 obtained from prelim_fit(), (0, length of project)
        h: h obtained from prelim_fit(), (0, infinity)
        x0: x0 obtained from prelim_fit(), (guess - project length, guess + project length)
    
    params:
        no_BEC: 1D distribution with the centre data set to h
        left_bound, right_bounds: bounds to define the "centre" (BEC) and the "wings" (thermal cloud)
        peak_max: maximum height of project (used to limit fitting height)
        sigma_0, x0, A, h: parameters extracted from initial prelim_fit()
        
    returns:
        popt: array of fitting parameters in the order A, sigma_0, h, x0
        pcov: covariance array of popt
        array(xs): indices that correspond to the length of the outside points
        array(ys): 1D distribution of only the outside wings from no_BEC
    """
    xs = []
    ys = []
    for i in range(len(no_BEC)):
        if i < left_bound:
            xs.append(i)
            ys.append(no_BEC[i])
        elif i > right_bound:
            xs.append(i)
            ys.append(no_BEC[i])
    
    popt, pcov = curve_fit(gaussian_x, xs, ys, p0 = (A, sigma_0, h, x0), bounds = ((0, 0, 0, x0 - len(no_BEC)), (peak_max, len(no_BEC), inf, x0 + len(no_BEC))))
    return popt, pcov, array(xs), array(ys)

def subtract_thermal(project, sigma_0, x0, A):
    """
    subtracts 1D Gaussian (excluding offset h) from overall profile
    
    params:
        project: 1D distribution
        sigma_0, x0, A: parameters extracted from thermal_only_fit()
        
    returns:
        no_thermal: original image 1D distribution with the thermal fit subtracted off
    """
    no_thermal = zeros(len(project))
    for i in range(len(project)):
        no_thermal[i] = project[i] - gaussian_x(i, A, sigma_0, 0, x0)
    return no_thermal

def TF_fit(no_thermal, peak_max, R, x0, TF0, h, no_h = 'no'):
    """
    fits to a 1D Thomas-Fermi profile
    
    fitting guess, (lower bound, upper bound):
        R: input R, (0, length of project)
        x0: input x0, (guess - project length, guess + project length)
        TF0: input TF0, (0, maximum height)
        h: input h, (0, infinity)
    
    params:
        no_thermal: 1D distribution with no thermal part
        peak_max: maximum allowed height of fit
        R, x0, TF0, h: guess parameters (either extracted from prelim_fit() or directly input)
        no_h: fits with zero vertical offset (set to 'no' by default)
        
    returns:
        popt: array of fitting parameters in the order R, x0, TF0, h
        pcov: covariance array of popt
    """
    xs = list(range(len(no_thermal)))
    if no_h == 'yes':
        popt, pcov = curve_fit(TF_1D_no_h, xs, no_thermal, p0 = (R, x0, TF0), bounds = ((0, x0 - len(no_thermal), 0), (len(no_thermal), x0 + len(no_thermal), peak_max)))
    else:
        popt, pcov = curve_fit(TF_1D, xs, no_thermal, p0 = (R, x0, TF0, h), bounds = ((0, x0 - len(no_thermal), 0, 0), (len(no_thermal), x0 + len(no_thermal), peak_max, inf)))
    
    return popt, pcov

def TF_plot(image, a = 1.1, plot_true = True):
    """
    performs fitting routing for Thomas-Fermi and Gaussian sum
    note each dimension is called "x"
    
    params:
        image: transmission image
        a: determines size of wings, set to 1.1 by default
        plot_true: produces a plot of the various fits if True, set to True by default
        
    returns:
        array(popt_thermal): parameter array in form [x, y] where each of x and y contains [A, sigma_0, h, x0] from thermal fit
        array(pcov_thermal): [x, y] where each of x and y is the covariance matrix of popt_thermal
        array(popt_TF): parameter array in form [x, y] where each of x and y contains [R, x0, TF0, h] from TF fit
        array(pcov_TF): [x, y] where each of x and y is the covariance matrix of popt_TF
    """
    if plot_true == True:
        log_image = -log(image)
        plt.imshow(log_image)
        plt.show()
    x_proj, y_proj = log_integrate_1D(image)
    
    popt_thermal = []
    pcov_thermal = []
    popt_TF = []
    pcov_TF = []
    
    for proj in [x_proj, y_proj]:
        x_1D = proj
        popt, pcov, peak_max = prelim_fit(x_1D)
        R, sigma_0, x0, TF0, A, h = popt[0], popt[1], popt[2], popt[3], popt[4], popt[5]
        x_no_BEC, left_bound, right_bound = subtract_BEC(x_1D, R, x0, h, a)
        popt2, pcov2, x_outside, y_outside = thermal_only_fit(x_no_BEC, left_bound, right_bound, peak_max, sigma_0, x0, A, h)
        A2, sigma_02, h2, x02 = popt2[0], popt2[1], popt2[2], popt2[3]
        x_no_thermal = subtract_thermal(x_1D, sigma_02, x02, A2)
        popt3, pcov3 = TF_fit(x_no_thermal, peak_max, R, x0, TF0, h, no_h = 'no')
        R3, x03, TF03, h3 = popt3[0], popt3[1], popt3[2], popt3[3]

        if plot_true == True:
            x = range(len(x_no_BEC))
            outside_fit = array([gaussian_x(x, A2, sigma_02, h2, x02) for x in range(len(x_no_BEC))])
            remainder_fit = array([TF_1D(x, R3, x03, TF03, h3) for x in range(len(x_no_BEC))])
            
            plt.plot(x, x_1D, label = 'data')
            plt.plot(x, x_no_BEC, label = 'wings')
            plt.plot(x, outside_fit, label = 'thermal fit')
            plt.plot(x, x_no_thermal, label = 'TF only')
            plt.plot(x, remainder_fit, label = 'TF fit')
            plt.legend()
            plt.show()
        
        popt_thermal.append(popt2)
        pcov_thermal.append(pcov2)
        popt_TF.append(popt3)
        pcov_TF.append(pcov3)
        
    return array(popt_thermal), array(pcov_thermal), array(popt_TF), array(pcov_TF)

def TFOnly_plot(image, plot_true = True, no_h = 'no'):
    """
    performs fitting routing for Thomas-Fermi profile only (no thermal cloud)
    note each dimension is called "x"
    
    fitting guess:
        R: rms_width() value
        x0: index of maximum height
        TF0: maximum height
        h: 0
    
    params:
        image: transmission image
        plot_true: produces a plot of the fit if True, set to True by default
        no_h: fits with zero vertical offset (set to 'no' by default)
        
    returns:
        array(popt_TF): parameter array in form [x, y] where each of x and y contains [R, x0, TF0, h] from TF fit
        array(pcov_TF): [x, y] where each of x and y is the covariance matrix of popt_TF
    """
    if plot_true == True:
        log_image = -log(image)
        plt.imshow(log_image)
        plt.show()
    x_proj, y_proj = log_integrate_1D(image)
    
    popt_TF = []
    pcov_TF = []
    
    for proj in [x_proj, y_proj]:
        x_1D = proj
        peak_max = max(x_1D) * 2 # larger than max to allow for noise at peak
        R_guess = rms_width(x_1D)
        x0_guess = argmax(x_1D)
        TF0_guess = max(x_1D)
        h_guess = 0
        
        if no_h == 'yes':
            popt, pcov = TF_fit(x_1D, peak_max, R_guess, x0_guess, TF0_guess, h_guess, no_h = 'yes')
            R, x0, TF0, h = popt[0], popt[1], popt[2], 0
        else:
            popt, pcov = TF_fit(x_1D, peak_max, R_guess, x0_guess, TF0_guess, h_guess, no_h = 'no')
            R, x0, TF0, h = popt[0], popt[1], popt[2], popt[3]

        if plot_true == True:
            x = range(len(x_1D))
            TF_points = array([TF_1D(x, R, x0, TF0, h) for x in range(len(x_1D))])
            
            plt.plot(x, x_1D, label = 'data') # data
            plt.plot(x, TF_points, label = 'TF fit') # TF fit
            plt.legend()
            plt.show()
        
        popt_TF.append(popt)
        pcov_TF.append(pcov)
        
    return array(popt_TF), array(pcov_TF)

def get_temp(ts, sigmas, error_sigma):
    popt, pcov = curve_fit(sigma, ts, sigmas, sigma = error_sigma)
    return m / k_b * popt[1] * 10.**6, m / k_b * pcov[1,1] * 10.**6, popt[0], popt[1]

def fit_widths(mypath, mag, detuning, i_s = [-1], time = 10, v = 'no', AOI = [0, 0, 1291, 963]):
    names = get_Directory(mypath)
    names = group_names(names)
    x1, y1, x2, y2 = AOI
    Ts = []
    Ns = []
    for i in i_s:
        atoms = gaussian_filter(transmission(mypath, names[int(i)])[int(y1):int(y2),int(x1):int(x2)], sigma = 3)
        center = find_center(atoms)
        sigma_x, sigma_y, error_x, error_y, A_x, h_x, x0_x, A_y, h_y, x0_y = fit_1D_gaussians(atoms, center)
        
        if v == 'yes':
            plot_1D_fits(atoms, center, A_x, sigma_x, h_x, x0_x, A_y, sigma_y, h_y, x0_y)
            
        T = ((sigma_x + sigma_y) / 2 * pixel * mag / time)**2 * 0.5 * 1.67 * 23 * 100 / 1.38
        N = AOI_integration(atoms, center, [int(6 * sigma_x), int(6 * sigma_y)], detuning, mag, v = 'no')
        print( ((sigma_x) * pixel * mag / time)**2 * 0.5 * 1.67 * 23 * 100 / 1.38 )
        Ts.append(T)
        Ns.append(N)
    return Ns, Ts

def fit_progression(mypath, mag, detuning, times, v = 'no', offset = 0, AOI = [0, 0, 1291, 963], fit = 'thermal'):
    """
    3 types of fits: "thermal" (Gaussian only), "thermalBEC" (Gaussian + TF), "BECOnly" (TF only)
        "thermal": fit_1D_gaussians() and plot_1D_fits()
        "thermalBEC": TF_plot()
        "BECOnly": TFOnly_plot()
    """
    sigma_xs = []
    sigma_ys = []
    error_xs = []
    error_ys = []
    atom_num = []
    
    x1, y1, x2, y2 = AOI
    
    names = get_Directory(mypath)
    if len(names) > 3 * len(times):
        names = names[int(len(names)- 3 * len(times) - 3 * len(times) * offset):int(len(names) - 3 * offset * len(times))]
        
    names = group_names(names)
    print (names)
    atoms = gaussian_filter(transmission(mypath, names[0])[int(y1):int(y2),int(x1):int(x2)], sigma = 3)
    center = find_center(atoms)
    
    if fit == 'thermal' or fit == 'thermalBEC':
        for i in names:
            atoms = gaussian_filter(transmission(mypath, i)[int(y1):int(y2),int(x1):int(x2)], sigma = 3)

            if fit == 'thermal':
                A_x, sigma_x, h_x, x0, error_sigma_x, A_y, sigma_y, h_y, y0, error_sigma_y = fit_1D_gaussians(atoms, center, no_h = 'no')
                if v == 'yes':
                    print(sigma_x, sigma_y, error_sigma_x, error_sigma_y, A_x, h_x, x0, A_y, h_y, y0)
                    plot_1D_fits(atoms, center, A_x, sigma_x, h_x, x0, A_y, sigma_y, h_y, y0)

            elif fit == 'thermalBEC':
                if v == 'yes':
                    popt_thermal, pcov_thermal, popt_TF, pcov_TF = TF_plot(atoms, a = 1.1, plot_true = True)
                if v == 'no':
                    popt_thermal, pcov_thermal, popt_TF, pcov_TF = TF_plot(atoms, a = 1.1, plot_true = False)
                A_x, sigma_x, h_thermal_x, x0_thermal_x, R_x, x0_TF_x, TF0_x, h_TF_x = popt_thermal[0, 0], popt_thermal[0, 1], popt_thermal[0, 2], popt_thermal[0, 3], popt_TF[0, 0], popt_TF[0, 1], popt_TF[0, 2], popt_TF[0, 3]
                A_y, sigma_y, h_thermal_y, x0_thermal_y, R_y, x0_TF_y, TF0_y, h_TF_y = popt_thermal[1, 0], popt_thermal[1, 1], popt_thermal[1, 2], popt_thermal[1, 3], popt_TF[1, 0], popt_TF[1, 1], popt_TF[1, 2], popt_TF[1, 3]
                error_sigma_x = sqrt(pcov_thermal[0, 1, 1])
                error_sigma_y = sqrt(pcov_thermal[1, 1, 1])

            atom_num.append( AOI_integration( atoms, center, [sigma_x * 6, sigma_y * 6], detuning, mag , v = v ) )
            sigma_xs.append(sigma_x * pixel * mag)
            sigma_ys.append(sigma_y * pixel * mag)
            error_xs.append(error_sigma_x * pixel * mag)
            error_ys.append(error_sigma_y * pixel * mag)

        sigma_xs, sigma_ys = power(array(sigma_xs), 2), power(array(sigma_ys), 2) 

        error_xs, error_ys = power(array(error_xs), 2), power(array(error_ys), 2)

        t_x, error_t_x, sigma_0_x, v_x = get_temp(times, sigma_xs, error_xs)
        fits_x = sigma(v_x, sigma_0_x, times)

        t_y, error_t_y, sigma_0_y, v_y = get_temp(times, sigma_ys, error_ys)
        fits_y = sigma(v_y, sigma_0_y, times)

        print (sqrt(sigma_0_x), sqrt(sigma_0_y))

        if v == 'yes':
            plot_progression(times, sigma_xs, error_xs, fits_x, sigma_ys, error_ys, fits_y)
            print ((t_x + t_y) / 2, atom_num)

        return t_x, t_y, atom_num
    
    elif fit == 'BECOnly':
        for i in names:
            atoms = gaussian_filter(transmission(mypath, i)[int(y1):int(y2),int(x1):int(x2)], sigma = 3)
            no_h_value = 'no'
            
            if v == 'yes':
                popt_TFOnly, pcov_TFOnly = TFOnly_plot(atoms, plot_true = True, no_h = no_h_value)
            elif v == 'no':
                popt_TFOnly, pcov_TFOnly = TFOnly_plot(atoms, plot_true = False, no_h = no_h_value)
                
            if no_h_value == 'yes':
                R_x, x0_x, TF0_x, h_x = popt_TFOnly[0, 0], popt_TFOnly[0, 1], popt_TFOnly[0, 2], 0
                R_y, x0_y, TF0_y, h_y = popt_TFOnly[1, 0], popt_TFOnly[1, 1], popt_TFOnly[1, 2], 0
            else:
                R_x, x0_x, TF0_x, h_x = popt_TFOnly[0, 0], popt_TFOnly[0, 1], popt_TFOnly[0, 2], popt_TFOnly[0, 3]
                R_y, x0_y, TF0_y, h_y = popt_TFOnly[1, 0], popt_TFOnly[1, 1], popt_TFOnly[1, 2], popt_TFOnly[1, 3]
            
            atom_num.append( AOI_integration( atoms, center, [R_x * 1.5, R_y * 1.5], detuning, mag, v = v ) )
            
        return 0, 0, atom_num
    
    else:
        return

def plot_progression(times, sigma_xs, error_xs, fits_x, sigma_ys, error_ys, fits_y):
    fig, (ax1, ax2) = plt.subplots(2)
    
    ax1.scatter(times, sigma_xs)
    ax1.errorbar(times, sigma_xs, yerr = error_xs, fmt = 'o')
    ax1.plot(times, fits_x)    
    
    ax2.scatter(times, sigma_ys)
    ax2.errorbar(times, sigma_ys, yerr = error_ys, fmt = 'o')
    ax2.plot(times, fits_y)
    
    plt.show()

def main():
    t_Start = time()
    mag = 2.5
    detuning = 1.
    repump_time = 2.
    #times = power(array([2,3,4,5,6,7]) + repump_time, 2)
    #times = power(array([11, 12, 13]) + repump_time, 2)
    #times = power(array([8,9,10,11,12,13,14,15,16,17,18]) + repump_time, 2)
    times = power(array([4, 5, 6, 7, 8]) + repump_time, 2)
    
    mypath = 'C:/Users/Columbia/Documents/Python Scripts/for cal'
    
    n = 1
    AOI = [200, 300, 900, 764]
    
    #Ns, Ts = fit_widths(mypath, mag, detuning, i_s = linspace(-n, -1, n), v = 'yes', time = 10.4)
    #print (round(array(Ns), 2).tolist())
    #print (round(array(Ts), 2).tolist())
    
    #offsets = list(range(23))
    #for i in offsets:    
    #    t_x, t_y, num = fit_progression(mypath, mag, detuning, times, v = 'no', offset = i)
    #    print (i, t_x, t_y, t_x / 2 + t_y / 2, num)
    
    #t_x, t_y, num = fit_progression(mypath, mag, detuning, times, v = 'yes', offset = 0, fit = 'thermal')
    t_x, t_y, num = fit_progression(mypath, mag, detuning, times, v = 'yes', offset = 0, fit = 'thermalBEC')
    #t_x, t_y, num = fit_progression(mypath, mag, detuning, times, v = 'yes', offset = 0, fit = 'BECOnly')
    
    plt.show()
    #print ('temperature (uK)', 'Number (10^6)')
    #print (t_x / 2 + t_y / 2, mean(num), num, t_x, t_y)
    
    #plot_progression(times, sigma_xs, error_xs, fits_x, sigma_ys, error_ys, fits_y)
    
if __name__ == "__main__":
    main()