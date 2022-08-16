#!/usr/bin/env python

# coding: utf-8

# In[136]:


# !/usr/bin/env python

# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as pl
import matplotlib.ticker as mtick
import os
import shutil
from matplotlib.animation import FuncAnimation
import mpl_toolkits.axes_grid1
import matplotlib.widgets
import scipy.stats
from math import log10, floor
from scipy.optimize import curve_fit

class Player(FuncAnimation):
    def __init__(self, fig, func, frames=None, init_func=None, fargs=None,
                 save_count=None, mini=0, maxi=100, pos=(0.125, 0.92), **kwargs):
        self.i = 0
        self.min = mini
        self.max = maxi
        self.runs = True
        self.forwards = True
        self.fig = fig
        self.func = func
        self.setup(pos)
        FuncAnimation.__init__(self, self.fig, self.update, frames=self.play(),
                               init_func=init_func, fargs=fargs,
                               save_count=save_count, **kwargs)

    def play(self):
        while self.runs:
            self.i = self.i + self.forwards - (not self.forwards)
            if self.i > self.min and self.i < self.max:
                yield self.i
            else:
                self.stop()
                yield self.i

    def start(self):
        self.runs = True
        self.event_source.start()

    def stop(self, event=None):
        self.runs = False
        self.event_source.stop()

    def forward(self, event=None):
        self.forwards = True
        self.start()

    def backward(self, event=None):
        self.forwards = False
        self.start()

    def oneforward(self, event=None):
        self.forwards = True
        self.onestep()

    def onebackward(self, event=None):
        self.forwards = False
        self.onestep()

    def onestep(self):
        if self.i > self.min and self.i < self.max:
            self.i = self.i + self.forwards - (not self.forwards)
        elif self.i == self.min and self.forwards:
            self.i += 1
        elif self.i == self.max and not self.forwards:
            self.i -= 1
        self.func(self.i)
        self.slider.set_val(self.i)
        self.fig.canvas.draw_idle()

    def setup(self, pos):
        playerax = self.fig.add_axes([pos[0], pos[1], 0.64, 0.04])
        divider = mpl_toolkits.axes_grid1.make_axes_locatable(playerax)
        bax = divider.append_axes("right", size="80%", pad=0.05)
        sax = divider.append_axes("right", size="80%", pad=0.05)
        fax = divider.append_axes("right", size="80%", pad=0.05)
        ofax = divider.append_axes("right", size="100%", pad=0.05)
        sliderax = divider.append_axes("right", size="500%", pad=0.07)
        self.button_oneback = matplotlib.widgets.Button(playerax, label='$\u29CF$')
        self.button_back = matplotlib.widgets.Button(bax, label='$\u25C0$')
        self.button_stop = matplotlib.widgets.Button(sax, label='$\u25A0$')
        self.button_forward = matplotlib.widgets.Button(fax, label='$\u25B6$')
        self.button_oneforward = matplotlib.widgets.Button(ofax, label='$\u29D0$')
        self.button_oneback.on_clicked(self.onebackward)
        self.button_back.on_clicked(self.backward)
        self.button_stop.on_clicked(self.stop)
        self.button_forward.on_clicked(self.forward)
        self.button_oneforward.on_clicked(self.oneforward)
        self.slider = matplotlib.widgets.Slider(sliderax, '',
                                                self.min, self.max, valinit=self.i)
        self.slider.on_changed(self.set_pos)

    def set_pos(self, i):
        self.i = int(self.slider.val)
        self.func(self.i)

    def update(self, i):
        self.slider.set_val(i)


# os.chdir('lanczos_diag_data')
# os.chdir('../')

def lanczos_ave_extract(file_name, size):
    os.chdir('lanczos_diag_data')
    table = np.loadtxt(file_name)
    index_arr = table[:, 0]
    h_x_range = table[:, 1]
    GS_energy_val = table[:, 2]
    first_excited_energy_val = table[:, 3]
    second_derivative_energy_val = table[:, 4]
    susceptibility_val = table[1:, 5]
    structure_factor_val = table[:, 6]
    os.chdir('../')
    return h_x_range, GS_energy_val / size, first_excited_energy_val / size, susceptibility_val / size, structure_factor_val / size

def chi_ii_extract(file_name):
    os.chdir('chi_ii_prob')
    with open(file_name, 'rb') as f:
        chi_ii_arr = np.load(f)
    os.chdir('../')
    return chi_ii_arr

def lanczos_all_extract(file_name):
    with open(file_name, 'rb') as f:
        ground_energy_all = np.load(f)
        first_energy_all = np.load(f)
        chi_arr_all = np.load(f)
        S_SG_arr_all = np.load(f)
        EE_arr_all = np.load(f)
    return ground_energy_all, first_energy_all, chi_arr_all, S_SG_arr_all, EE_arr_all

def test_plot(h_x_range, all_quantity, ylabel=None):
    fig = pl.figure(figsize=(8, 6))
    pl.figure(facecolor='white')
    pl.rcParams['font.size'] = '18'
    for h_x_quantity in all_quantity:
        pl.plot(h_x_range, h_x_quantity)
    pl.ylabel(ylabel, fontsize=22)
    pl.xlabel(r'$h_x/J_0$', fontsize=22)
    pl.xticks(fontsize=18)
    pl.yticks(fontsize=18)
    pl.tick_params('both', length=7, width=2, which='major')
    pl.tick_params('both', length=5, width=2, which='minor')
    pl.gca().yaxis.set_major_formatter(mtick.FormatStrFormatter('%.3f'))
    pl.grid(False)
    # pl.yscale("log")
    pl.legend(loc=0, prop={'size': 16}, numpoints=1, scatterpoints=1, ncol=1)
    fig.tight_layout(pad=0.5)
    # pl.savefig("susceptibility plot")
    pl.show()

def coeff_extract():
    table = np.loadtxt('series_d2.txt')
    chi_coeff = table[:, 1]
    E_coeff = table[:, 2]
    S_coeff = table[:, 3]
    # print(S_coeff)
    return chi_coeff, E_coeff, S_coeff

def expand(coeff_vec, h, max_order):
    result = 0.
    x = h**(-2.)
    for i in range(max_order):
        result += coeff_vec[i]*(x**i)
    return result

def h_x_expand(h_x_range, max_order):
    chi_coeff, E_coeff, S_coeff = coeff_extract()
    chi = np.array([expand(chi_coeff, h, max_order) for h in h_x_range])
    E = np.array([expand(E_coeff, h, max_order) for h in h_x_range])
    S = np.array([expand(S_coeff, h, max_order) for h in h_x_range])
    return chi, E, S

def errorbar(h_x_range, ave, std, normalize = False, ylabel=None, legend_list=None, yupperlim=None, ylowerlim=None,
             savefile=False, filename=None, ewidth=None, xupperlim=None, xlowerlim=None, expansion_result = False, chi_diverg = False, expansion = None):
    '''
    ave, std need to be dimensions (different_sizes, h_x_values)
    '''
    fig = pl.figure(figsize=(8, 6), facecolor='white')
    pl.rcParams['font.size'] = '18'
    if normalize:
        for i in range(np.shape(ave)[0]):
            ave[i] = ave[i]/np.max(ave[i])
            std[i] = std[i]/np.max(std[i])
    else:
        for i in range(np.shape(ave)[0]):
            ave[i] = ave[i]
            std[i] = std[i]
    for i in range(np.shape(ave)[0]):
        if legend_list != None:
            pl.errorbar(h_x_range, ave[i], yerr= std[i], label=str(legend_list[i]),
                        elinewidth=ewidth)
        else:
            pl.errorbar(h_x_range, ave[i], yerr= std[i], elinewidth=ewidth)
    pl.ylabel(ylabel, fontsize=22)
    pl.xlabel(r'$h_x/J_0$', fontsize=22)
    pl.xticks(fontsize=18)
    pl.yticks(fontsize=18)
    pl.tick_params('both', length=7, width=2, which='major')
    pl.tick_params('both', length=5, width=2, which='minor')
    pl.gca().yaxis.set_major_formatter(mtick.FormatStrFormatter('%.3f'))
    pl.grid(False)
    # pl.yscale("log")
    pl.ylim(bottom=ylowerlim, top=yupperlim)
    pl.xlim(left=xlowerlim, right=xupperlim)
    if expansion_result:
        pl.axvline(x = 4.521**0.5, color = 'k', linestyle = '--', label = r'$T_c$')
    if chi_diverg:
        pl.axvline(x=5.045**0.5, color = 'm', linestyle='--', label=r'$\chi_{SG}$ divergence')

    # linked cluster expansion comparison
    if expansion != None:
        max_order_arr = [10, 12, 14]
        lowfieldcut = 2.
        truncated_h_x = h_x_range[np.where(h_x_range >= lowfieldcut)]
        for max_order in max_order_arr:
            chi_expansion_arr, E_expansion_arr, S_expansion_arr = h_x_expand(truncated_h_x, max_order)
            expansion_arr = np.zeros(len(truncated_h_x))
            if expansion == 'chi':
                expansion_arr = np.divide(chi_expansion_arr, np.power(truncated_h_x, 2.))
            elif expansion == 'E':
                expansion_arr = np.multiply(E_expansion_arr, truncated_h_x)
            elif expansion == 'S':
                expansion_arr = S_expansion_arr
            else:
                print('expansion should be chi, E or S')
                pass
            pl.plot(truncated_h_x, expansion_arr, label = 'max order: {order}'.format(order = max_order))
    pl.legend(prop={'size': 16}, numpoints=1, scatterpoints=1, ncol=1)
    fig.tight_layout(pad=0.5)
    # pl.yscale('log')
    if savefile:
        pl.savefig(filename)
        pl.ioff()
    else:
        pl.show()

def distribution(h_x_index, data_all, size_list):
    # degeneracy histogram
    distribution_size = data_all[:, :, h_x_index]
    size_label = [str(size) for size in size_list]
    fig = pl.figure(figsize=(8, 6))
    # histogram on log scale.
    pl.hist(distribution_size, density=True, label=size_label)
    pl.ylabel('probability', fontsize=18)
    pl.xlabel('susceptibility', fontsize=18)
    pl.xticks(fontsize=18)
    pl.yticks(fontsize=18)
    pl.tick_params('both', length=7, width=2, which='major')
    pl.tick_params('both', length=5, width=2, which='minor')
    pl.grid(False)
    pl.gca().yaxis.set_major_formatter(mtick.FormatStrFormatter('%.6f'))
    pl.legend(loc=5, prop={'size': 16}, numpoints=1, scatterpoints=1, ncol=1)
    fig.tight_layout(pad=0.5)

def round_to_1(x):
    return round(x, -int(floor(log10(abs(x)))))

def general_plot(directory_name, h_x_range, plot = True, distribution_animation=False):
    '''
    This function reads all the data_all files and directly outputs the plots
    '''
    listdir = os.listdir(directory_name)
    os.chdir(directory_name)
    size_list = [int(''.join(filter(str.isdigit, string))) for string in listdir]
    ground_energy_all_size = []
    first_energy_all_size = []
    chi_arr_all_size = []
    S_SG_arr_all_size = []
    EE_arr_all_size = []
    for i, file_name in enumerate(listdir):
        ground_energy_all, first_energy_all, chi_arr_all, S_SG_arr_all, EE_arr_all = lanczos_all_extract(file_name)
        ground_energy_all_size.append(ground_energy_all/size_list[i])
        first_energy_all_size.append(first_energy_all/size_list[i])
        chi_arr_all_size.append(chi_arr_all/size_list[i])
        S_SG_arr_all_size.append(S_SG_arr_all/size_list[i])
        EE_arr_all_size.append(EE_arr_all/size_list[i]**0.5)
        # (EE_arr_all - np.log(2.)) / np.power(size_list[i], 0.5)
    os.chdir('../')
    def ave(all_size_arr):
        ave_size_arr = []
        for all_arr in all_size_arr:
            ave_size_arr.append(np.mean(all_arr, axis = 0))
        return ave_size_arr

    def std(all_size_arr):
        ave_size_arr = []
        for all_arr in all_size_arr:
            ave_size_arr.append(np.std(all_arr, axis = 0))
        return ave_size_arr

    ground_energy_ave_size = ave(ground_energy_all_size)
    first_energy_ave_size = ave(first_energy_all_size)
    chi_arr_ave_size = ave(chi_arr_all_size)
    S_SG_arr_ave_size = ave(S_SG_arr_all_size)
    EE_arr_ave_size = ave(EE_arr_all_size)

    ground_energy_std_size = std(ground_energy_all_size)
    first_energy_std_size = std(first_energy_all_size)
    chi_arr_std_size = std(chi_arr_all_size)
    S_SG_arr_std_size = std(S_SG_arr_all_size)
    EE_arr_std_size = std(EE_arr_all_size)

    if plot:
        # output files
        # check to see whether the output file already exists
        dir_name = 'errorbar_plots'
        if os.path.isdir(dir_name):
            os.chdir(dir_name)
        else:
            os.mkdir(dir_name)

        errorbar(h_x_range, ground_energy_ave_size, ground_energy_std_size, ylabel=r'$E_0$', legend_list=size_list,
                 savefile=True, filename='GS_energy', expansion = 'E')
        errorbar(h_x_range, first_energy_ave_size, first_energy_std_size, ylabel=r'$E_1$', legend_list=size_list,
                 savefile=True, filename='1st_energy')
        errorbar(h_x_range, [first_energy_ave_size[k]-ground_energy_ave_size[k] for k in range(len(size_list))], np.power(np.power(first_energy_std_size, 2) + np.power(ground_energy_std_size, 2), 0.5), ylabel=r'$\Delta E$',
                 legend_list=size_list,
                 savefile=True, filename='1st_energy_gap')
        errorbar(h_x_range, chi_arr_ave_size, chi_arr_std_size, normalize = False, ylabel=r'$\chi_{SG}$', legend_list=size_list,
                 yupperlim=10., ylowerlim=0., savefile=True, filename='susceptibility', ewidth=0.3, xlowerlim=1.5, xupperlim=4., expansion_result=True, chi_diverg=True, expansion='chi')
        errorbar(h_x_range, S_SG_arr_ave_size, S_SG_arr_std_size, ylabel=r'$S_{SG}$', legend_list=size_list,
                 yupperlim=10., ylowerlim=1., savefile=True, filename='structure factor', expansion_result=True, chi_diverg=False, expansion='S',ewidth=0.3)
        errorbar(h_x_range, EE_arr_ave_size, EE_arr_std_size, ylabel=r'$EE/\sqrt{N}$', legend_list=listdir, savefile=True,
                 filename='entanglement entropy',ewidth=0.5)

        if distribution_animation:
            # histogram on log scale.
            fig = pl.figure(figsize = (8,6))
            fig.tight_layout(pad=0.5)
            pl.ylabel('counts', fontsize=18)
            pl.xlabel('susceptibility', fontsize=18)
            pl.xticks(fontsize=18)
            pl.yticks(fontsize=18)
            pl.tick_params('both', length=7, width=2, which='major')
            pl.tick_params('both', length=5, width=2, which='minor')
            pl.grid(False)
            pl.gca().yaxis.set_major_formatter(mtick.FormatStrFormatter('%.3f'))
            size_label = [str(size) for size in size_list]
            # pl.xlim(left=-10, right=100)
            index = 20
            distribution_size = np.array(chi_arr_all_size)[:, :, index]
            n, bins, patches = pl.hist(distribution_size.T, bins=70, label=size_label, alpha=0.65)
            pl.legend(loc=0, prop={'size': 16}, numpoints=1, scatterpoints=1, ncol=1)
            pl.show()
            pl.savefig('dsitribution_hx={hx}.png'.format(hx = h_x_range[index]))
        os.chdir('../')

    return ground_energy_all_size, first_energy_all_size, chi_arr_all_size, S_SG_arr_all_size, EE_arr_all_size

def chi_ii_hist(directory_name, h_x_index, plot = True):
    '''
    This function reads all the data_all files and directly outputs the plots
    '''
    listdir = os.listdir(directory_name)
    size_list = [int(''.join(filter(str.isdigit, string))) for string in listdir]
    chi_ii_size = []
    for file_name in listdir:
        chi_ii_arr = chi_ii_extract(file_name)
        chi_ii_size.append(chi_ii_arr)

    if plot:
        # output files
        # check to see whether the output file already exists
        dir_name = 'chi_ii_plots'
        if os.path.isdir(dir_name):
            os.chdir(dir_name)
        else:
            os.mkdir(dir_name)
            os.chdir(dir_name)

        # chi_ii histogram

        size_label = [str(size) for size in size_list]
        fig = pl.figure(figsize=(8, 6))
        pl.rcParams['font.size'] = '18'
            # pl.hist(chi_ii_size[:, h_x_index, :], bins=70, label=size_label, stacked=stack)
        if h_x_range[h_x_index] < 2.1:
            hist = []
            for i in range(len(size_list)):
                chi_ii = chi_ii_size[i]
                chi_ii = chi_ii[h_x_index]
                hist.append(chi_ii)
            n, bins, patches = pl.hist(hist, density = True, label = size_label)
            pl.legend(loc='upper right', prop={'size': 16}, numpoints=1, scatterpoints=1, ncol=1)

            hist = np.array([])
            for i in range(len(size_list)):
                chi_ii_hist = chi_ii_size[i]
                hist = np.append(hist, chi_ii_hist[h_x_index, :])
            [mean, std] = scipy.stats.norm.fit(hist)
            x = np.linspace(min(bins), max(bins))
            pl.plot(x, scipy.stats.norm.pdf(x, mean, std), label='normal')
        else:
            hist = np.array([])
            for i in range(len(size_list)):
                chi_ii_hist = chi_ii_size[i]
                hist = np.append(hist, chi_ii_hist[h_x_index, :])
            n, bins, patches = pl.hist(hist, density=True)
            [mean, std] = scipy.stats.norm.fit(hist)
            x = np.linspace(min(bins), max(bins))
            pl.plot(x, scipy.stats.norm.pdf(x, mean, std), label = 'normal')

        # popt, pcov = curve_fit(func_powerlaw, bins[:-1], n)
        # pl.plot(x, func_powerlaw(x, *popt), '--', label = 'powerlaw')
        pl.ylabel(r'probability density', fontsize=18)
        pl.xlabel(r'$\chi_{ii}$', fontsize=18)
        pl.xticks(fontsize=18)
        pl.yticks(fontsize=18)
        pl.tick_params('both', length=7, width=2, which='major')
        pl.tick_params('both', length=5, width=2, which='minor')
        pl.grid(False)
        fig.tight_layout(pad=0.5)
        title = r"$h_x$ = {hx}, mean = {m}, std = {s}".format(hx = round(h_x_range[h_x_index], 3), m = round(mean, 3), s = round_to_1(std))
        pl.title(title)
        pl.ioff()
        pl.savefig("chi_ii_histogram_{index}.png".format(index = h_x_index), bbox_inches='tight')
        os.chdir('../')
    return chi_ii_size



init = 0.1
final = 4.
num_steps = 50
h_x_range = np.concatenate((np.linspace(init, final, num_steps), np.linspace(final+0.5, final+6., 5)))

# if __name__ == '__main__':
#     for h_x_index in range(len(h_x_range)):
#         chi_ii_hist('chi_ii_prob', h_x_index)
