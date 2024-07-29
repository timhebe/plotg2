import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from scipy.ndimage import zoom
from scipy.optimize import curve_fit
import scipy.signal as signal
from scipy.signal import argrelmax, argrelmin
import pylablib as pll
import os
import os.path
from matplotlib import animation
import math
from lmfit import Model


class LaserScan(object):

    def __init__(self, path):
        object.__init__(self)

        self.path = path
        self.data = None
        self.load_data(path)

    def load_data(self, path):
        self.data = pd.read_csv(path, delimiter='\t')

    def filter_frequency_jumps(self, threshold=1e8, freq_jump_length=100, padding=3):
        freq = np.array(self.data['Frequency'])
        df = freq[1:] - freq[:-1]
        jumps = np.array(abs(df) > threshold)
        jump_locs = np.append(jumps.nonzero()[0], [len(df) - 1])
        prev_jump = -1
        include = np.ones(len(df) + 1).astype("bool")
        for jl in jump_locs:
            if jl - prev_jump < freq_jump_length:
                start = max(prev_jump + 1 - padding, 0)
                end = min(jl + 1 + padding, len(include))
                include[start:end] = False
            prev_jump = jl

        self.data = self.data[include]
        return include

    def find_smooth_fit_param(self, x, y):
        '''
        function finds automatically starting values for frequency smoothing fit
        --> only works for single scan
        '''
        x = np.array(x)
        y = np.array(y)
        b = y.argmin()
        c = y.min()
        a = ((y[0] - y.min()) / b + (y[-1] - y.min()) / b) / 2

        return a, b, c

    def fit_smooth_freq(self, x, y, a, b, c, fix_a=False, fix_b=False, fix_c=False, figure=False):
        '''
        x - list of x values
        y - list of y values
        a - slope
        b - x center
        c - offset

        '''
        # Fitfunc
        function = lambda x, a, b, c: a * np.abs(x - b) + c
        fmodel = Model(function)

        header = ['a', 'b', 'c']
        header_err = ['a error', 'b error', 'c error']
        params = fmodel.make_params(a=a, b=b, c=c)
        if a:
            params['a'].vary = False
        if b:
            params['b'].vary = False
        if c:
            params['c'].vary = False

        result = fmodel.fit(y, params, x=x)
        param = []

        for i in header:
            if i in result.params:
                param.append(result.params[i].value)
            else:
                print('Parameters are not identified.', i)

        for i in header:
            if i in result.params:
                param.append(result.params[i].stderr)
            else:
                print('Parameters are not identified.', i)

        header = header + header_err
        par = pd.DataFrame(columns=header)
        par.loc[0] = param

        if figure:
            fig, ax = plt.subplots()
            ax.scatter(x, y, color='tab:red', s=4)
            ax.plot(x, result.eval(), color='tab:blue')
        return par, result.eval()

    def fit_smooth_freq_auto_parameter(self, x, y, a=None, b=None, c=None, fix_a=False, fix_b=False, fix_c=False,
                                       figure=True):
        '''
        function first finds starting parameters
        '''
        a_new, b_new, c_new = self.find_smooth_fit_param(x, y)
        if a is None:
            a = a_new
        if b is None:
            b = b_new
        if c is None:
            c = c_new
        parameters, fit = self.fit_smooth_freq(x, y, a, b, c, fix_a=False, fix_b=False, fix_c=False, figure=figure)

        return parameters, fit

    def fit_smooth_freq_batch(self, frequencies, fix_a=False, fix_b=False, fix_c=False, figure=True):
        num = len(frequencies)
        start_index = 0
        for i in range(num):
            index = np.arange(len(frequencies[i]))
            freq = frequencies[i]
            parameters, fit = self.fit_smooth_freq_auto_parameter(index, freq, a=None, b=None, c=None, fix_a=False,
                                                                  fix_b=False, fix_c=False, figure=True)
            frequencies[i] = pd.Series(fit, index=np.arange(len(frequencies[i])) + start_index)
            start_index = start_index + len(frequencies[i]) + 1
            frequencies[i].name = "Frequency"
        return frequencies

    def fit_lorentzian(self, x, y, gamma, a, x0, y0, fix_gamma=False, fix_a=False, fix_x0=False, fix_y0=False, Hz=True,
                       figure=True):
        '''
        x - list of x values
        y - list of y values
        p -list of start parameters 
        gamma - FWHM (sigma)
        a - amplitude
        x0 - center wavelength (frequency) (mu - expected value)
        y0 - offset

        '''
        #Convert to GHz
        if Hz:
            x = x * 1e-9
            x0 = x0 * 1e-9
            gamma = gamma * 1e-9
        x0_init = x0
        # Fitfunc
        function = lambda x, gamma, a, x0, y0: a * (0.5 * gamma) ** 2 / (
                    np.pi * ((x - x0) ** 2 + (0.5 * gamma) ** 2)) + y0

        header = ['gamma', 'a', 'x0', 'y0']
        header_err = ['gamma error', 'a error', 'x0 error', 'y0 error']
        fmodel = Model(function)
        params = fmodel.make_params(gamma=gamma, a=a, x0=0, y0=y0)
        #params['y0'].min =0
        params['gamma'].min = 0
        if fix_y0:
            params['y0'].vary = False
        if fix_a:
            params['a'].vary = False
        if fix_x0:
            params['x0'].vary = False
        if fix_gamma:
            params['gamma'].vary = False
        result = fmodel.fit(y, params, x=x - x0)
        param = []

        for i in header:
            param.append(result.params[i].value)
        param[2] = param[2] + x0_init

        param.append(result.params['gamma'].value)

        for i in header:
            param.append(result.params[i].stderr)

        param.append(result.params['gamma'].stderr)

        header.append('FWHM')

        header_err.append('FWHM err')

        header = header + header_err
        par = pd.DataFrame(columns=header)
        par.loc[0] = param
        if figure:
            fig, ax = plt.subplots()
            ax.plot(x, y)
            ax.plot(x, result.eval())
        return par, result.eval()

    def find_fit_param(self, x, y, starting=1, ending=0):
        '''
        function finds automatically starting values for Lorentz fit
        --> probably only works for isolated molecules
        '''
        x = np.array(x)
        y = np.array(y)
        #filter out unwanted data
        x = x[starting:-1 - ending]
        y = y[starting:-1 - ending]
        x0 = x[y.argmax()]
        y0 = (y[0] + y[-1]) / 2
        a = y.max() - y0

        #approx FWHM
        idx1 = np.abs(y[:y.argmax()] - (y.max() + y0) / 2).argmin()
        idx2 = np.abs(y[y.argmax():] - (y.max() + y0) / 2).argmin()
        gamma = np.abs(x[idx1] - x[idx2 + y.argmax()])
        return gamma, a, x0, y0

    def fit_lorentzian_auto_parameter(self, x, y, gamma=None, a=None, x0=None, y0=None, fix_gamma=False, fix_a=False,
                                      fix_x0=False, fix_y0=False, Hz=True, figure=True):
        '''
        function first finds starting parameters
        '''
        gamma_new, a_new, x0_new, y0_new = self.find_fit_param(x, y)
        if gamma is None:
            gamma = gamma_new
        if a is None:
            a = a_new
        if x0 is None:
            x0 = x0_new
        if y0 is None:
            y0 = y0_new
        parameters, fit = self.fit_lorentzian(x, y, gamma, a, x0, y0, fix_gamma=False, fix_a=False, fix_x0=False,
                                              fix_y0=False, Hz=True, figure=figure)

        return parameters, fit

    def fit_lorentzian_batch_old(self, data_freq, data_intensity, gamma, a, x0, y0, freq_range=[0, 1e15],
                                 fix_gamma=False,
                                 fix_a=False, fix_x0=False, fix_y0=False, Hz=True, figure=True):
        """
        data_ - list Frequencies,'intensities
        p -list of start parameters
        gamma - FWHM (sigma)
        a - amplitude
        x0 - center wavelength (frequency) (mu - expected value)
        y0 - offset

        """

        data_fits = []
        # range
        if figure:
            color1 = iter(cm.Blues(np.linspace(1, 0, 2 * len(data_freq))))
            color2 = iter(cm.Reds(np.linspace(1, 0, 2 * len(data_freq))))
            fig, ax = plt.subplots()

        for i in range(len(data_freq)):
            freq = data_freq[i].reset_index(drop=True)
            inten = data_intensity[i].reset_index(drop=True)
            low_idx = freq.sub(freq_range[0]).abs().idxmin()
            high_idx = freq.sub(freq_range[1]).abs().idxmin()
            if i == 0:
                param, fits = self.fit_lorentzian(data_freq[i].iloc[low_idx:high_idx],
                                                  data_intensity[i].iloc[low_idx:high_idx], gamma=gamma, a=a,
                                                  x0=x0, y0=y0, fix_gamma=fix_gamma, fix_a=fix_a, fix_x0=fix_x0,
                                                  fix_y0=fix_y0, figure=False)
                parameters = param
            else:
                param, fits = self.fit_lorentzian(data_freq[i].iloc[low_idx:high_idx],
                                                  data_intensity[i].iloc[low_idx:high_idx], gamma=gamma, a=a,
                                                  x0=x0, y0=y0, fix_gamma=fix_gamma, fix_a=fix_a, fix_x0=fix_x0,
                                                  fix_y0=fix_y0, figure=False)
                parameters = parameters.append(param)

            data_fits.append(fits)

            if figure:
                c1 = next(color1)
                c2 = next(color2)
                ax.scatter(data_freq[i].iloc[low_idx:high_idx] / 1e12, data_intensity[i].iloc[low_idx:high_idx] / 1000,
                           color=c1)
                ax.plot(data_freq[i].iloc[low_idx:high_idx] / 1e12, data_fits[i] / 1000, color=c2)

        if figure:
            ax.set_xlabel('Frequency (THz)')
            ax.set_ylabel('Intensity (kcps)')

        return data_fits, parameters

    def fit_lorentzian_batch(self, data_freq, data_intensity, gamma=None, a=None, x0=None, y0=None,
                             freq_range=[0, 1e15], fix_gamma=False, fix_a=False, fix_x0=False, fix_y0=False, Hz=True,
                             figure=True):
        '''
        data_ - list Frequencies,'intensities
        p -list of start parameters 
        gamma - FWHM (sigma)
        a - amplitude
        x0 - center wavelength (frequency) (mu - expected value)
        y0 - offset

        '''

        data_fits = []
        data_frequencies = []
        ##range
        if figure:
            color1 = iter(cm.Blues(np.linspace(1, 0, 2 * len(data_freq))))
            color2 = iter(cm.Reds(np.linspace(1, 0, 2 * len(data_freq))))
            fig, ax = plt.subplots()

        for i in range(len(data_freq)):
            freq = data_freq[i].reset_index(drop=True)
            low_idx = freq.sub(freq_range[0]).abs().idxmin()
            high_idx = freq.sub(freq_range[1]).abs().idxmin()
            low_idx = 0
            high_idx = len(freq)
            if i == 0:
                param, fits = self.fit_lorentzian_auto_parameter(data_freq[i].iloc[low_idx:high_idx],
                                                                 data_intensity[i].iloc[low_idx:high_idx], gamma=gamma,
                                                                 a=a, x0=x0, y0=y0, fix_gamma=fix_gamma, fix_a=fix_a,
                                                                 fix_x0=fix_x0, fix_y0=fix_y0, figure=False)
                parameters = param
            else:
                param, fits = self.fit_lorentzian_auto_parameter(data_freq[i].iloc[low_idx:high_idx],
                                                                 data_intensity[i].iloc[low_idx:high_idx], gamma=gamma,
                                                                 a=a, x0=x0, y0=y0, fix_gamma=fix_gamma, fix_a=fix_a,
                                                                 fix_x0=fix_x0, fix_y0=fix_y0, figure=False)
                parameters = parameters.append(param)

            data_fits.append(fits)
            data_frequencies.append(data_freq[i].iloc[low_idx:high_idx])

            if figure:
                c1 = next(color1)
                c2 = next(color2)
                ax.scatter(data_freq[i].iloc[low_idx:high_idx] / 1e12, data_intensity[i].iloc[low_idx:high_idx] / 1000,
                           color=c1)
                ax.plot(data_freq[i].iloc[low_idx:high_idx] / 1e12, data_fits[i] / 1000, color=c2)

        if figure:
            ax.set_xlabel('Frequency (THz)')
            ax.set_ylabel('Intensity (kcps)')

        return data_fits, data_frequencies, parameters

    def find_fit_sat_param(self, x, y_intensity, y_FWHM):
        '''
        function finds automatically starting values for saturation curve fitting
        '''
        x = np.log10(x)
        diff_intensity = np.gradient(y_intensity, x)
        max_diff_pos = diff_intensity.argmax()
        Is = 10 ** x[max_diff_pos]
        y_FWHM = np.array(y_FWHM)
        y_intensity = np.array(y_intensity)
        v0 = y_FWHM[max_diff_pos] / np.sqrt(2)
        y0 = y_intensity[0]
        R_inf = (y_intensity[max_diff_pos] - y0) * 2
        return R_inf, Is, y0, v0

    def fit_saturation(self, data_power, data_intensity, data_FWHM, R_inf, Is, y0, v0, fix_R_inf=False, fix_Is=False,
                       fix_y0=False, fix_v0=False, Hz=False, nW=True, figure=False):
        '''
        data_power - list of power values
        data_intensity - list of intensity (count) values
        data_FWHM - list of linewidth values
        R_inf - saturated intensity
        Is - saturation power
        y0 - offset
        v0 - zero intensity linewidth

        '''
        # Convert to GHz
        if Hz:
            data_FWHM = data_FWHM * 10 ** 9
        if nW:
            data_power = data_power * 1000
        # Fitfunc
        function_intensity = lambda x, R_inf, Is, y0: R_inf * x / Is / (1 + x / Is) + y0
        function_FWHM = lambda x, Is_FWHM, v0: v0 * (1 + x / Is_FWHM) ** 0.5

        header = ['R_inf', 'Is', 'y0', 'Is_FWHM', 'v0']
        header_err = ['R_inf error', 'Is error', 'y0 error', 'Is_FWHM error', 'v0 error']
        fmodel_intensity = Model(function_intensity)
        fmodel_FWHM = Model(function_FWHM)
        params_intensity = fmodel_intensity.make_params(R_inf=R_inf, Is=Is, y0=y0)
        params_FWHM = fmodel_FWHM.make_params(Is_FWHM=Is, v0=v0)
        params_intensity['y0'].min = 0
        if fix_R_inf:
            params_intensity['R_inf'].vary = False
        if fix_Is:
            params_intensity['Is'].vary = False
            params_FWHM['Is_FWHM'].vary = False
        if fix_y0:
            params_intensity['y0'].vary = False
        if fix_v0:
            params_FWHM['v0'].vary = False

        result_intensity = fmodel_intensity.fit(data_intensity, params_intensity, x=data_power)
        result_FWHM = fmodel_FWHM.fit(data_FWHM, params_FWHM, x=data_power)
        param = []

        for i in header:
            if i in result_intensity.params:
                param.append(result_intensity.params[i].value)
            elif i in result_FWHM.params:
                param.append(result_FWHM.params[i].value)
            else:
                print('Parameters are not identified.', i)

        for i in header:
            if i in result_intensity.params:
                param.append(result_intensity.params[i].stderr)
            elif i in result_FWHM.params:
                param.append(result_FWHM.params[i].stderr)
            else:
                print('Parameters are not identified.', i)

        header = header + header_err
        par = pd.DataFrame(columns=header)
        par.loc[0] = param

        if figure:
            fig, ax = plt.subplots()
            ax.scatter(data_power, data_intensity, color='tab:red')
            ax.plot(data_power, result_intensity.eval(), color='tab:red')
            ax1 = ax.twinx()
            ax1.scatter(data_power, data_FWHM, color='tab:blue')
            ax1.plot(data_power, result_FWHM.eval(), color='tab:blue')
            ax.set_xscale('log')
        return par, result_intensity.eval(), result_FWHM.eval()

    def fit_saturation_auto_parameter(self, data_power, data_intensity, data_FWHM, R_inf=None, Is=None, y0=None,
                                      v0=None, fix_R_inf=False, fix_Is=False,
                                      fix_y0=False, fix_v0=False, Hz=False, nW=True, figure=False):
        '''
        function first finds starting parameters
        '''
        R_inf_new, Is_new, y0_new, v0_new = self.find_fit_sat_param(data_power, data_intensity, data_FWHM)
        if R_inf is None:
            R_inf = R_inf_new
        if Is is None:
            Is = Is_new
        if y0 is None:
            y0 = y0_new
        if v0 is None:
            v0 = v0_new
        parameters, fit_intensity, fit_FWHM = self.fit_saturation(data_power, data_intensity, data_FWHM, R_inf, Is, y0,
                                                                  v0, fix_R_inf=fix_R_inf, fix_Is=fix_Is, fix_y0=fix_y0,
                                                                  fix_v0=fix_v0, Hz=Hz, nW=nW, figure=figure)

        return parameters, fit_intensity, fit_FWHM

    def get_index_extrema_frequency(self, data):
        idx, _ = signal.find_peaks(np.array(data.Frequency - data.Frequency.shift(-1)), height=5e9, distance=20)
        if len(idx) == 1:
            idx = [0, len(data.Frequency)]
        elif len(idx) == 0:
            idx = [0, len(data.Frequency)]
        return idx[0], idx[1] - 2

    def get_all_power_spectra(self, intensity='APD_1', intensity2='APD_2', power='Voltage_Input_3', rep=0, figure=True):
        data = self.data
        waveplate_angles = data['RMount1'].unique()
        frequencies = []
        intensities = []
        average_power = []

        for angle in waveplate_angles.tolist():
            if 'Rep' in data.columns:
                data_angle = data[(data['RMount1'] == angle) & (data['Rep'] == rep)]
            else:
                data_angle = data[(data['RMount1'] == angle)]
            idx_0 = data_angle.index[0]
            index_min, index_max = self.get_index_extrema_frequency(data_angle)
            average_power.append(np.mean(np.array(data_angle[power])))
            frequencies.append(data.Frequency.loc[idx_0 + index_min:idx_0 + index_max])
            intensities.append(data[intensity].loc[idx_0 + index_min:idx_0 + index_max] + data[intensity2].loc[
                                                                                          idx_0 + index_min:idx_0 + index_max])

        ##plotting
        color1 = iter(cm.Blues(np.linspace(1, 0, 2 * len(waveplate_angles))))
        if figure:
            fig, ax = plt.subplots()
            for i in range(len(waveplate_angles)):
                c1 = next(color1)
                ax.plot(frequencies[i] / 1e12, intensities[i] / 1000, color=c1)
            ax.set_xlabel('Frequency (THz)')
            ax.set_ylabel('Intensity (kcps)')

        return frequencies, intensities, waveplate_angles, average_power

    def get_all_iteration_spectra(self, intensity='APD_1', intensity2='APD_2', power='Voltage_Input_3', intensity_num=2,
                                  rep=0, figure=True):
        data = self.data
        iteration = data['Rep'].unique()
        frequencies = []
        intensities = []
        scan_num = np.array(0)
        i = 0
        for num in iteration.tolist():
            data_iteration = data[(data['Rep'] == num)]
            idx_0 = data_iteration.index[0]
            index_min, index_max = self.get_index_extrema_frequency(data_iteration)
            if i == 0:
                scan_num = np.array(data['Rep'].loc[idx_0 + index_min:idx_0 + index_max])
                frequencies = np.array(data.Frequency.loc[idx_0 + index_min:idx_0 + index_max])
                if intensity_num == 2:
                    intensities = np.array(
                        data[intensity].loc[idx_0 + index_min:idx_0 + index_max] + data[intensity2].loc[
                                                                                   idx_0 + index_min:idx_0 + index_max])
                else:
                    intensities = np.array(data[intensity].loc[idx_0 + index_min:idx_0 + index_max])
                i = 1
            else:
                scan_num = np.append(scan_num, np.array(data['Rep'].loc[idx_0 + index_min:idx_0 + index_max]))
                frequencies = np.append(frequencies, np.array(data.Frequency.loc[idx_0 + index_min:idx_0 + index_max]))
                if intensity_num == 2:
                    intensities = np.append(intensities, np.array(
                        data[intensity].loc[idx_0 + index_min:idx_0 + index_max] + data[intensity2].loc[
                                                                                   idx_0 + index_min:idx_0 + index_max]))
                else:
                    intensities = np.append(intensities,
                                            np.array(data[intensity].loc[idx_0 + index_min:idx_0 + index_max]))

        ##plotting
        if figure:
            fig, ax = plt.subplots()
            ax.tricontourf(frequencies, scan_num, intensities, )
            ax.set_xlabel('Frequency (THz)')
            ax.set_ylabel('Iteration')

        return frequencies, intensities, scan_num

    def find_peaks(self, y=None, height=6000, distance=100):
        y = np.array(y)
        peaks = signal.find_peaks(y, height=height, distance=distance)
        return peaks[0]

    def find_power(self, filter_pos, angle_pos, power_file='power_summary.xlsx'):
        power1 = pd.read_excel(power_file, index_col=None)
        power = power1['Filter' + str(filter_pos)]
        power = power[angle_pos]
        return power

    def perform_power_analysis(self, gamma=None, a=None, x0=None, y0=None, freq_range=[0, 1e15], fix_gamma=False,
                               fix_a=False, fix_x0=False, fix_y0=False, intensity_monitor='APD_1',
                               power_monitor='Voltage_Input_3', rep=0, smooth_freq=False, figure=True, figure_sat=True):

        frequencies, intensities, angles, power = self.get_all_power_spectra(intensity=intensity_monitor,
                                                                             power=power_monitor, rep=rep,
                                                                             figure=figure)
        if smooth_freq:
            frequencies = self.fit_smooth_freq_batch(frequencies, fix_a=False, fix_b=False, fix_c=False, figure=False)
        data_fits, data_frequencies, parameters = self.fit_lorentzian_batch(frequencies, intensities, gamma=gamma, a=a,
                                                                            x0=x0, y0=y0, freq_range=freq_range,
                                                                            fix_gamma=fix_gamma, fix_a=fix_a,
                                                                            fix_x0=fix_x0, fix_y0=fix_y0, Hz=True,
                                                                            figure=figure)

        #subtract background power
        # needs to be interpolated in the future
        power = power - min(power)

        ### Figure
        if figure_sat:
            fig, ax = plt.subplots()
            ax.plot(power, parameters.gamma * 1000)
            ax0 = ax.twinx()
            ax0.plot(power, parameters.a / 1000, color='tab:red')
            ax0.set_ylabel('Intensity (a.u.)', color='tab:red')
            ax.set_ylabel('FWHM (MHz)')
            ax.set_xlabel('Photodiode Voltage(V)')
            ax.set_xscale('log')

        parameters['Power PD (V)'] = power
        parameters['Angles'] = angles

        data_and_fits = []
        for i in range(len(frequencies)):
            df = pd.DataFrame()
            # df['Frequency data']=frequencies[i]
            # df['Intensity data']=intensities[i]
            df['Frequency fit'] = data_frequencies[i]
            df['Intensity fit'] = data_fits[i]
            data_and_fits.append(df)
        return parameters, data_and_fits


class CamReader(object):
    """
    Reader class for .cam files.

    Allows transparent access to frames by reading them from the file on the fly (without loading the whole file).
    Supports determining length, indexing (only positive single-element indices) and iteration.

    Args:
        path(str): path to .cam file.
        same_size(bool): if ``True``, assume that all frames have the same size, which speeds up random access and obtaining number of frames;
            otherwise, the first time the length is determined or a large-index frame is accessed can take a long time (all subsequent calls are faster).
    """

    def __init__(self, path, same_size=False):
        object.__init__(self)
        self.path = self.normalize_path(path)
        self.frame_offsets = [0]
        self.frames_num = None
        self.same_size = same_size
        self.channel_intensities = None

    def eof(f, strict=False):
        """
        Standard EOF function.
        
        Return ``True`` if the the marker is at the end of the file.
        If ``strict==True``, only return ``True`` if the marker is exactly at the end of file; otherwise, return ``True`` if it's at the end of further.
        """
        p = f.tell()
        f.seek(0, 2)
        ep = f.tell()
        f.seek(p)
        return (ep == p) or (ep <= p and not strict)

    def _read_cam_frame(self, f, skip=False):
        size = np.fromfile(f, "<u4", count=2)
        if len(size) == 0 and self.eof(f):
            raise StopIteration
        if len(size) < 2:
            raise IOError("not enough cam data to read the frame size")
        w, h = size
        if not skip:
            data = np.fromfile(f, "<u2", count=w * h)
            if len(data) < w * h:
                raise IOError(
                    "not enough cam data to read the frame: {} pixels available instead of {}".format(len(data), w * h))
            return data.reshape((w, h))
        else:
            f.seek(w * h * 2, 1)
            return None

    def normalize_path(self, p):
        """Normalize filesystem path (case and origin). If two paths are identical, they should be equal when normalized."""
        return os.path.normcase(os.path.abspath(p))

    def _read_frame_at(self, offset):
        with open(self.path, "rb") as f:
            f.seek(offset)
            return self._read_cam_frame(f)

    def _read_next_frame(self, f, skip=False):
        data = self._read_cam_frame(f, skip=skip)
        self.frame_offsets.append(f.tell())
        return data

    def _read_frame(self, idx):
        idx = int(idx)
        if self.same_size:
            if len(self.frame_offsets) == 1:
                with open(self.path, "rb") as f:
                    self._read_next_frame(f, skip=True)
            offset = self.frame_offsets[1] * idx
            return self._read_frame_at(offset)
        else:
            if idx < len(self.frame_offsets):
                return self._read_frame_at(self.frame_offsets[idx])
            next_idx = len(self.frame_offsets) - 1
            offset = self.frame_offsets[-1]
            with open(self.path, "rb") as f:
                f.seek(offset)
                while next_idx <= idx:
                    data = self._read_next_frame(f, next_idx < idx)
                    next_idx += 1
            return data

    def _fill_offsets(self):
        if self.frames_num is not None:
            return
        if self.same_size:
            file_size = os.path.getsize(self.path)
            if file_size == 0:
                self.frames_num = 0
            else:
                with open(self.path, "rb") as f:
                    self._read_next_frame(f, skip=True)
                if file_size % self.frame_offsets[1]:
                    raise IOError("File size {} is not a multiple of single frame size {}".format(file_size,
                                                                                                  self.frame_offsets[
                                                                                                      1]))
                self.frames_num = file_size // self.frame_offsets[1]
        else:
            offset = self.frame_offsets[-1]
            try:
                with open(self.path, "rb") as f:
                    f.seek(offset)
                    while True:
                        self._read_next_frame(f, skip=True)
            except StopIteration:
                pass
            self.frames_num = len(self.frame_offsets) - 1

    def size(self):
        """Get the total number of frames"""
        self._fill_offsets()
        return self.frames_num

    __len__ = size

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return list(self.iterrange(idx.start or 0, idx.stop, idx.step or 1))
        try:
            return self._read_frame(idx)
        except StopIteration:
            raise IndexError("index {} is out of range".format(idx))

    def get_data(self, idx):
        """Get a single frame at the given index (only non-negative indices are supported)"""
        return self[idx]

    def __iter__(self):
        return self.iterrange()

    def iterrange(self, *args):
        """
        iterrange([start,] stop[, step])

        Iterate over frames starting with `start` ending at `stop` (``None`` means until the end of file) with the given `step`.
        """
        start, stop, step = 0, None, 1
        if len(args) == 1:
            stop, = args
        elif len(args) == 2:
            start, stop = args
        elif len(args) == 3:
            start, stop, step = args
        if step < 0:
            raise IndexError("format doesn't support reversed indexing")
        try:
            n = start
            while True:
                yield self._read_frame(n)
                n += step
                if stop is not None and n >= stop:
                    break
        except StopIteration:
            pass

    def read_all(self):
        """Read all available frames"""
        return list(self.iterrange())
        ####### End General

        ####### End extract channels
        ####### Movie

    def binned_name(self, pfx, nbin, ext="bin"):
        """Get the name of the file with the binned data for a given prefix, binning number, and extention"""
        return os.path.join(pfx, "binned_{}.{}".format(nbin, ext))

    def decfunc(self, mode):
        """Get the appropriate decimation function"""
        if mode == "max":
            return np.max
        if mode == "mean":
            return np.mean
        if mode == "min":
            return np.min
        if mode == "skip":
            return lambda a, axis: a.take(0, axis=axis)

    def load_and_bin(self, path, nbin, ntot=None, mode="max"):
        """
        Load and bin .cam file.

        Args:
            path: source path
            nbin: binning factor
            ntot: total number of binned frames to load (so that total number of raw frames to read is ``ntot*nbin``);
                by default, load all frames
            mode: binning mode; can be ``"mean"``, ``"max"``, ``"min"``, or ``"skip"``.
        """
        dec = self.decfunc(mode)
        #reader=cam.CamReader(path,same_size=True)
        if ntot is None:
            ntot = len(self) // nbin
        ntot = min(ntot, len(self) // nbin)
        frame = self.get_data(0)
        result = np.full((ntot,) + frame.shape, 0,
                         dtype=float)  # use np.full instead of np.zeros to actually allocate RAM
        for i in range(ntot):
            frames = self[i * nbin:(i + 1) * nbin]
            binned = dec(frames, axis=0)
            result[i] = binned
        return result

    def save_prebinned(self, frames, pfx, nbin):
        """Save pre-binned data"""
        path = self.binned_name(pfx, nbin)
        os.makedirs(os.path.split(path)[0], exist_ok=True)
        frames.astype("<f8").tofile(path)

    def load_prebinned(self, pfx, sfx, nbin):
        """Load already pre-binned data"""
        bname = self.binned_name(pfx, nbin)
        if os.path.exists(bname):
            reader = cam.CamReader("{}_{}.cam".format(pfx, sfx), same_size=True)
            return np.fromfile(bname, "<f8").reshape((-1,) + reader[0].shape)
        return None

    def show_movie(self, frames, nshow=None, **kwargs):
        """
        Show movie from the frames.

        All ``kwargs`` parameters are passed to ``imshow``.
        """
        img = plt.imshow(frames[0], **kwargs)

        def update(i):
            img.set_data(frames[i])

        ani = animation.FuncAnimation(img.axes.figure, update, interval=1E3 / self.fps, frames=nshow or len(frames),
                                      repeat=True)
        return ani

    def make_movie(self, sfx='vid', nbin=100, fps=10, name='test', **kwargs):
        """Load and bin data (or load prebinned if exists and ``recalc==False``), save it, and save a movie"""
        self.fps = fps
        bdata = self.load_and_bin("{}_{}.cam".format(self.path, sfx), nbin)

        bdata -= np.median(bdata, axis=0)  # subtract median background (wide-field fluorescence)
        #self.save_prebinned(bdata,self.path,nbin)
        ani = self.show_movie(bdata, **kwargs)
        #ani.save(self.binned_name(self.path,nbin,ext="mp4"),"ffmpeg",fps=fps)
        ani.save(name + '.mp4', fps=fps)
        ##### End Movie

    def extract_channel(self, channel_h=[0, 10], channel_v=[0, 10]):
        '''

        '''
        intensity = []
        for i in range(len(self)):
            intensity.append(self[i][channel_h[0]:channel_h[1], channel_v[0]:channel_v[1]].mean())

        return intensity

    def extract_all_channel_intensities(self, pixels=8, pixel_range=None):
        data = []
        if pixel_range is None:
            for i in range(len(self)):
                data.append(zoom(self[i], 1 / pixels))
        else:
            for i in range(len(self)):
                data.append(
                    zoom(self[i][pixel_range[0][0]:pixel_range[0][1], pixel_range[1][0]:pixel_range[1][1]], 1 / pixels))
        data = np.array(data)
        returndata = pd.DataFrame()
        for i in range(np.shape(data[0])[0]):
            for j in range(np.shape(data[0])[0]):
                returndata[str(i) + ' ' + str(j)] = data[:, i, j]

        data = None
        self.channel_intensities = returndata
        self.get_max_channel_intensity(data=returndata, figure=True)
        return returndata

    def length(self):
        return len(self)

    def get_max_channel_intensity(self, data=None, figure=True):
        if data is None:
            data = self.channel_intensities
        maximum = data.max().max()
        col = data.max().idxmax()
        idx = data[col].idxmax()

        print('Maximum in %s at index %i', col, idx)

        if figure:
            fig, ax = plt.subplots()
            ax.plot(data[col])
            ax.scatter(idx, maximum, color='tab:red')
            ax.set_ylabel('Intensity (a.u.)')
            ax.set_ylabel('Index')
