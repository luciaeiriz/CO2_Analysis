import csv
from extract_data import data
import numpy as np
import math
import matplotlib.pyplot as plt 
from matplotlib.ticker import MultipleLocator, FormatStrFormatter, AutoLocator, AutoMinorLocator
import os
import mplhep as hep
from datetime import datetime
from scipy.optimize import curve_fit


font0 = {'family': 'DejaVu Sans',
            'weight': 'bold',
            'size': 36,
            } # for plot title
font1 = {'family': 'DejaVu Sans',
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

def read_background_rates(filename):
    background_rates = {}
    with open(filename, 'r') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # Skip the header row
        for row in reader:
            scan_name, wp_str, background_rate_str = row
            wp = float(wp_str) / 1000
            background_rate = float(background_rate_str)
            if scan_name not in background_rates:
                background_rates[scan_name] = []
            background_rates[scan_name].append((wp, background_rate))
    return background_rates

def plot_scan(scan):
    HV = data['HV'][scan]
    eff = data['eff'][scan]
    noise_gamma = np.array(data['noise_gamma'][scan])
    gamma_cs = np.array(data['gamma_cs'][scan])
    gamma_cs_err =  np.array(data['gamma_cs_err'][scan])  

    if scan not in plot_scan_colors:
        color = plot_scan_colors[scan] = colors[len(plot_scan_colors) % len(colors)]
    color = plot_scan_colors[scan]

    if scan not in plot_scan_markers:
        marker = plot_scan_markers[scan] = markers[len(plot_scan_markers) % len(markers)]
    marker = plot_scan_markers[scan]

    parts = scan.split("_")
    mixture = parts[0]
    d = mixtures.get(mixture, f"Gas mixture: {mixture}")
    labels = []

    for wp, background_rate in background_rates.get(scan, []):
        labels.append(f"Gas mixture = {d}, WP = {wp} [kV], Bckg rate = {background_rate:.2f} Hz/cm2")
    label = '\n'.join(labels)
    plt.errorbar(HV, gamma_cs, yerr=gamma_cs_err, fmt=marker, markersize=9, color=color, label=label)

background_rates = read_background_rates('background_rates.csv')
plot_scan_colors = {}
plot_scan_markers = {}

#Setting the plot
hep.style.use("CMS")
figure, ax = plt.subplots(figsize=(13, 13))
plt.xlabel(r'HV$_{\mathrm{eff}}$ [kV]', fontsize=36)
plt.ylabel('Gamma cluster', fontsize=36)
plt.grid(ls='--')

ax.xaxis.set_major_locator(AutoLocator())
ax.yaxis.set_major_locator(AutoLocator())
ax.xaxis.set_minor_locator(AutoMinorLocator())
ax.yaxis.set_minor_locator(AutoMinorLocator())
ax.tick_params(which='major', direction='in', length=10, width=2.0, labelsize=12)
ax.tick_params(which='minor', direction='in', length=5, width=1.0)
plt.yticks(fontproperties='DejaVu Sans', size=20, weight='bold')  
plt.xticks(fontproperties='DejaVu Sans', size=20, weight='bold') 
plt.xlim(5700, 7850)
plt.ylim(0.8, 3.5)

# Input scan names from user
scans = input("Enter the names of the scans you want to plot, separated by commas: ").split(',')
#name =  datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
name = input("Enter a name for the plot: ")

for scan in scans:
    plot_scan(scan.strip())

if not os.path.exists('Plots_2024/'):
    os.makedirs('Plots_2024/')

# Plotting 
label = "ABS 6.9" #change label accordingly to ABS filter
ax.text(0.025, 0.78, label + "\nTest Beam April 2024 \nThreshold: 60 fC \n1.4 mm double gap RPC", transform=ax.transAxes, verticalalignment='top', horizontalalignment='left', fontsize=16)

plt.text(7400, 3.5+0.01, "GIF++", font0)
hep.cms.text("Preliminary", fontsize=32)
plt.legend(loc='upper left', prop=font1, frameon=False)
plt.axhline(y=3, color='black', linestyle='--')
plt.savefig(os.path.join('Plots_2024/' + name + ".png"))

#plt.show()

