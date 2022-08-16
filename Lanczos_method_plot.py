#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as pl
import matplotlib.ticker as mtick
import os

N = 16

# os.chdir("../")
os.chdir('lanczos_ave_data')
table = np.loadtxt("lanczos_ave data_[4, 4] seed from 2 to 99.txt")
index_arr = table[: ,0]
h_x_range = table[:, 1]
GS_energy_val = table[:, 2]
first_excited_energy_val = table[:, 3]
second_derivative_energy_val = table[:, 4]
susceptibility_val = table[:, 5]
structure_factor_val = table[:, 6]

# ground energy plot
fig = pl.figure(figsize=(8, 6))
pl.rcParams['font.size'] = '18'
pl.plot(h_x_range,GS_energy_val/float(16), lw=1.3, ls='-', color="blue", label= "ground state energy per site")
pl.plot(h_x_range,second_derivative_energy_val/float(N), lw=1.3, ls='-', color="red", label= "second derivative")
pl.ylabel(r'energy$/J_0$', fontsize=22)
pl.xlabel(r'$h_x/J_0$', fontsize=22)
pl.xticks(fontsize=18)
pl.yticks(fontsize=18)
pl.tick_params('both', length=7, width=2, which='major')
pl.tick_params('both', length=5, width=2, which='minor')
pl.gca().yaxis.set_major_formatter(mtick.FormatStrFormatter('%.6f'))
pl.grid(False)
# pl.xscale('log')
pl.legend(loc=0, prop={'size': 16}, numpoints=1, scatterpoints=1, ncol=1)
fig.tight_layout(pad=0.5)
# pl.savefig("GS energy plot")
pl.show()

# First energy gap plot
fig = pl.figure(figsize=(8, 6))
pl.rcParams['font.size'] = '18'
pl.plot(h_x_range, (first_excited_energy_val - GS_energy_val)/float(N), lw=1.3, ls='-', color="blue", label= "first energy gap per site")
pl.ylabel(r'first energy gap $/J_0$', fontsize=22)
pl.xlabel(r'$h_x/J_0$', fontsize=22)
pl.xticks(fontsize=18)
pl.yticks(fontsize=18)
pl.tick_params('both', length=7, width=2, which='major')
pl.tick_params('both', length=5, width=2, which='minor')
pl.gca().yaxis.set_major_formatter(mtick.FormatStrFormatter('%.6f'))
pl.grid(False)
# pl.xscale('log')
pl.legend(loc=0, prop={'size': 16}, numpoints=1, scatterpoints=1, ncol=1)
fig.tight_layout(pad=0.5)
# pl.savefig("First energy gap plot")
pl.show()

# Susceptibility plot
fig = pl.figure(figsize = (8, 6))
pl.rcParams['font.size'] = '18'
pl.plot(h_x_range, susceptibility_val, lw = 1.3, ls='-', color="blue", label= "susceptibility")
pl.plot(h_x_range, N/h_x_range, lw = 1.3, ls='-', color="red", label= "lattice.N/h_x")
pl.ylabel(r'$\chi_{SG}$', fontsize=22)
pl.xlabel(r'$h_x/J_0$', fontsize=22)
pl.xticks(fontsize=18)
pl.yticks(fontsize=18)
pl.tick_params('both', length=7, width=2, which='major')
pl.tick_params('both', length=5, width=2, which='minor')
pl.gca().yaxis.set_major_formatter(mtick.FormatStrFormatter('%.6f'))
pl.grid(False)
# pl.yscale("log")
pl.ylim((-10, 100))
pl.legend(loc=0, prop={'size': 16}, numpoints=1, scatterpoints=1, ncol=1)
fig.tight_layout(pad=0.5)
# pl.savefig("susceptibility plot")
pl.show()

# structure factor plot
fig = pl.figure(figsize = (8, 6))
pl.rcParams['font.size'] = '18'
pl.plot(h_x_range, structure_factor_val, lw = 1.3, ls='-', color="blue", label= "structure factor")
pl.ylabel(r'$S_{SG}$', fontsize=22)
pl.xlabel(r'$h_x/J_0$', fontsize=22)
pl.xticks(fontsize=18)
pl.yticks(fontsize=18)
pl.tick_params('both', length=7, width=2, which='major')
pl.tick_params('both', length=5, width=2, which='minor')
pl.gca().yaxis.set_major_formatter(mtick.FormatStrFormatter('%.6f'))
pl.grid(False)
# pl.xscale('log')
pl.legend(loc=0, prop={'size': 16}, numpoints=1, scatterpoints=1, ncol=1)
fig.tight_layout(pad=0.5)
# pl.savefig("structure factor plot")
pl.show()
