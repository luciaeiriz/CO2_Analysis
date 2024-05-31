import csv
import os
import matplotlib.pyplot as plt 
from matplotlib.ticker import MultipleLocator, FormatStrFormatter, AutoLocator, AutoMinorLocator
import pandas as pd
from scipy.optimize import curve_fit
import numpy as np
import math
import mplhep as hep
from extract_data import data
from datetime import datetime

""""
Before running the scripit: 
- change text according to gas mixture being used
- double check if TB from 2023/24 and change path directory accordingly
"""

#Fonts
font0 = {'family': 'DejaVu Sans',
            'weight': 'bold',
            'size': 34,
            } # for plot title
font1 = {'family': 'DejaVu Sans',
            'weight': 'normal',
            'size': 20,
            } # for text info
font2 = {'family': 'DejaVu Sans',
            'weight': 'bold',
            'size': 15,
            } # for legend

markers = ['o', '^', 's', '*', 'd']
colors = ['black', 'magenta', 'red', 'blue', 'green']
mixtures = {"30CO2" : "30% CO2 + 1% SF6",
            "30CO205SF6" : "30% CO2 + 0.5% SF6",
            "40CO2" : "40% CO2 + 1% SF6",
            "STDMX" : "Standard Mixture"
}

def find_index_of_value(HV, target_value):
    rounded_HV = math.ceil(target_value / 100) * 100
    try:
        return HV.index(rounded_HV)
    except ValueError:
        print(f"Value {rounded_HV} not found in HV column.")
        decremented_value = target_value - 100
        while decremented_value >= min(HV):
            rounded_decremented_value = math.ceil(decremented_value / 100) * 100
            try:
                return HV.index(rounded_decremented_value)
            except ValueError:
                print(f"Value {rounded_decremented_value} not found in HV column.")
                decremented_value -= 100
        print("No corresponding value found in HV column.")
        return None
        

def plot_scan(scan):
    HV = data['HV'][scan]
    eff = data['eff'][scan]
    err = data['err'][scan]
    noise_gamma = data['noise_gamma'][scan]
    gamma_cs = data['gamma_cs'][scan]


    def func(HV, E, L, H):
        return E / (1 + np.exp(L * (H - HV)))

    initial_guess = [0.98, 0.01, 7000]

    popt, pcov = curve_fit(func, HV, eff, p0=initial_guess, bounds=([0, -np.inf, -np.inf], [1.00, np.inf, np.inf]))
    E, L, H = popt

    knee = H - (math.log(1 / 0.95 - 1)) / L
    WP = knee + 120
    eff_WP = func(WP, E, L, H)
 
    a = round(E * 100)
    b = WP / 1000
    c = round(eff_WP * 100) 
    

    if scan not in plot_scan_colors:
        color = plot_scan_colors[scan] = colors[len(plot_scan_colors) % len(colors)]
    color = plot_scan_colors[scan]

    if scan not in plot_scan_markers:
        marker = plot_scan_markers[scan] = markers[len(plot_scan_markers) % len(markers)]
    marker = plot_scan_markers[scan]

    parts = scan.split("_")
    mixture = parts[0]
    d = mixtures.get(mixture, f"Gas mixture: {mixture}")

    label = f"{d}, plateau = {a} %, WP = {b:.2f} kV, Eff(WP) = {c} %"
    plt.errorbar(HV, eff, yerr=err, fmt=marker, markersize=9, color=color, label=label)

    x = np.linspace(min(HV), max(HV), 100)
    y = func(x, E, L, H)
    plt.plot(x, y, linewidth=3, color=color)

plot_scan_colors = {}
plot_scan_markers = {}

#Setting the plot
hep.style.use("CMS")
figure, ax = plt.subplots(figsize=(13, 13))
plt.xlabel(r'HV$_{\mathrm{eff}}$ [kV]', fontsize=36)
plt.ylabel('Muon efficiency [%]', fontsize=36)
plt.grid(ls='--')

ax.xaxis.set_major_locator(AutoLocator())
ax.yaxis.set_major_locator(AutoLocator())
ax.xaxis.set_minor_locator(AutoMinorLocator())
ax.yaxis.set_minor_locator(AutoMinorLocator())
ax.tick_params(which='major', direction='in', length=10, width=2.0, labelsize=12)
ax.tick_params(which='minor', direction='in', length=5, width=1.0)
plt.yticks(fontproperties='DejaVu Sans', size=20, weight='bold')  
plt.xticks(fontproperties='DejaVu Sans', size=20, weight='bold') 
plt.xlim(6050, 7550)
plt.ylim(0, 1.2)

# Input scan names from user
scans = input("Enter the names of the scans you want to plot, separated by commas: ").split(',')
#name =  datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
name = input("Enter a name for the plot: ")

for scan in scans:
    plot_scan(scan.strip())

if not os.path.exists('Plots_2024/'):
    os.makedirs('Plots_2024/')

# Plotting 
label = "ABS filter: 1" #change label accordingly ABS + TB date
ax.text(0.025, 0.75, label + "\nTest Beam April 2024 \nThreshold =  60 [fC] \n1.4 mm double gap RPC", transform=ax.transAxes, verticalalignment='top', horizontalalignment='left', fontsize=16)

plt.text(7300, 1.21+0.01, "GIF++", font0)
hep.cms.text("Preliminary", fontsize=32)
plt.legend(loc='upper left', prop=font2, frameon=False)
plt.axhline(y=1, color='black', linestyle='--')
plt.savefig(os.path.join('Plots_2024/' + name + ".png"))

#plt.show()

