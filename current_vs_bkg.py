import csv
import os
import re
import matplotlib.pyplot as plt
import mplhep as hep
import matplotlib.lines as mlines
from matplotlib.ticker import AutoLocator, AutoMinorLocator

font0 = {'family': 'DejaVu Sans',
         'weight': 'bold',
         'size': 38,
         }  # "GIF++"
mixtures = {"30CO2": r"64% TFE + 30% CO$_2$ + 5% iC$_4$H$_{10}$ + 1.0% SF$_6$",
    "30CO205SF6": r"64.5% TFE + 30% CO$_2$ + 5% iC$_4$H$_{10}$ + 0.5% SF$_6$",
    "40CO2": r"54% TFE + 40% CO$_2$ + 5% iC$_4$H$_{10}$ + 1.0% SF$_6$",
    "STDMX": r"95.2% TFE + 4.5% iC$_4$H$_{10}$ + 0.3% SF$_6$"
            }
colors = {
    "30CO2": 'blue',
    "30CO205SF6": 'green',
    "40CO2": 'orange',
    "STDMX": 'purple'
}
markers = {
    "30CO2": 'o',
    "30CO205SF6": 's',
    "40CO2": 'D',
    "STDMX": '^'
}

def read_data(filename):
    background_rates = {}
    with open(filename, 'r') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # Skip the header row
        for row in reader:
            scan_name,hv_str,current_top_str,current_bot_str,muoncs_str,gammacs_str,muoncs_err_str,gammacs_err_str,eff_str,eff_err,noiseGammaRate,current_str,background_rate_str,gamma_charge_str=row
            current = float(current_str)
            background_rate = float(background_rate_str)
            if scan_name not in background_rates:
                background_rates[scan_name] = []
            background_rates[scan_name].append((current, background_rate))
    return background_rates

def filter_gas_mixtures(background_rates, gas):
    gas_mixtures = {}
    for scan_name, rates in background_rates.items():
        if gas + '_' in scan_name:
            gas_mixtures[scan_name] = rates
    return gas_mixtures

def plot_background_rates(all_gas_mixtures_2023, all_gas_mixtures_2024):
    hep.style.use("CMS")
    figure, ax = plt.subplots(figsize=(12, 12))

    plotted_labels = set()

    for gas, gas_mixtures_2023 in all_gas_mixtures_2023.items():
        for scan_name, rates in gas_mixtures_2023.items():
            currents, background_rates = zip(*rates)
            label = f'{mixtures[gas]}' if gas not in plotted_labels else '_nolegend_'
            ax.errorbar(background_rates, currents, fmt=markers[gas], color=colors[gas], label=label, markersize=12)
            plotted_labels.add(gas)

    for gas, gas_mixtures_2024 in all_gas_mixtures_2024.items():
        for scan_name, rates in gas_mixtures_2024.items():
            currents, background_rates = zip(*rates)
            label = f'{mixtures[gas]}' if gas not in plotted_labels else '_nolegend_'
            ax.errorbar(background_rates, currents, fmt=markers[gas], color=colors[gas], mfc='none', mec=colors[gas], label=label, markersize=12)
            plotted_labels.add(gas)

    # Set plot properties on ax object
    ax.set_ylabel('Current [µA]', fontproperties='DejaVu Sans', size=22, weight='bold')
    ax.set_xlabel('Background Gamma Rate [kHz/cm²]', fontproperties='DejaVu Sans', size=22, weight='bold')
    ax.grid(ls='--')

    ax.xaxis.set_major_locator(AutoLocator())
    ax.yaxis.set_major_locator(AutoLocator())
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.tick_params(which='major', direction='in', length=10, width=2.0, labelsize=12)
    ax.tick_params(which='minor', direction='in', length=5, width=1.0)

    plt.yticks(fontproperties='DejaVu Sans', size=20, weight='bold')
    plt.xticks(fontproperties='DejaVu Sans', size=20, weight='bold')
    plt.ylim(-5, 250) 
    plt.xlim(-0.1, 2.1)

    # Create custom legend for gas mixtures
    handles, labels = ax.get_legend_handles_labels()
    unique_labels = dict(zip(labels, handles))
    legend = ax.legend(unique_labels.values(), unique_labels.keys(), loc='upper left', fontsize=20)

    # Add custom markers for the text
    
    filled_circle = mlines.Line2D([], [], color='black', marker='o', linestyle='None', markersize=12)
    open_circle = mlines.Line2D([], [], color='black', marker='o', linestyle='None', markersize=12, markerfacecolor='none')
    label = f"Threshold: 60 fC \n1.4 mm double gap RPC" 
    ax.text(0.12, 0.60, label, transform=ax.transAxes, verticalalignment='top', horizontalalignment='left', fontsize=22, linespacing=1.5)

    # Add the custom markers to the plot with text
    ax.add_artist(ax.text(0.15, 0.7, 'Before irradiation', transform=ax.transAxes, fontsize=24, verticalalignment='top', horizontalalignment='left'))
    ax.add_artist(filled_circle)
    filled_circle.set_xdata([0.12])
    filled_circle.set_ydata([0.69])
    filled_circle.set_transform(ax.transAxes)

    ax.add_artist(ax.text(0.15, 0.65, 'After irradiation', transform=ax.transAxes, fontsize=24, verticalalignment='top', horizontalalignment='left'))
    ax.add_artist(open_circle)
    open_circle.set_xdata([0.12])
    open_circle.set_ydata([0.64])
    open_circle.set_transform(ax.transAxes)

    plt.text(1.7, 250+3.5, "GIF++", font0)
    hep.cms.text("Preliminary", fontsize=32)

    plt.savefig(os.path.join('Plots_23v24', "current_vs_bkg.png"))

if __name__ == "__main__":
    eff_wp_calc_2023 = read_data('2023.csv')
    eff_wp_calc_2024 = read_data('2024.csv')
    
    all_gas_mixtures_2023 = {gas: filter_gas_mixtures(eff_wp_calc_2023, gas) for gas in mixtures.keys()}
    all_gas_mixtures_2024 = {gas: filter_gas_mixtures(eff_wp_calc_2024, gas) for gas in mixtures.keys()}
    
    plot_background_rates(all_gas_mixtures_2023, all_gas_mixtures_2024)
