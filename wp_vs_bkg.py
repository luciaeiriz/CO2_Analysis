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
            wp = float(hv_str) / 1000
            background_rate = float(background_rate_str)
            if scan_name not in background_rates:
                background_rates[scan_name] = []
            background_rates[scan_name].append((wp, background_rate))
    return background_rates

def filter_gas_mixtures(background_rates, gas):
    gas_mixtures = {}
    pattern = re.compile(rf"{gas}_(\d+(\.\d+)?|OFF)")
    for scan_name, rates in background_rates.items():
        match = pattern.match(scan_name)
        if match:
            abs_value = match.group(1)
            abs_number = 0 if abs_value == 'OFF' else float(abs_value)
            gas_mixtures[abs_number] = rates
    return gas_mixtures

def plot_background_rates(all_gas_mixtures_2023, all_gas_mixtures_2024):
    hep.style.use("CMS")
    figure, ax = plt.subplots(figsize=(12, 12))

    plotted_labels = set()

    marker_size = 200  # Adjust this value to increase or decrease the marker size

    for gas, gas_mixtures_2023 in all_gas_mixtures_2023.items():
        for abs_number, rates in gas_mixtures_2023.items():
            wps, background_rates = zip(*rates)
            label = f'{mixtures[gas]}' if gas not in plotted_labels else '_nolegend_'
            ax.scatter(background_rates, wps, marker=markers[gas], color=colors[gas], label=label, s=marker_size)
            plotted_labels.add(gas)

    for gas, gas_mixtures_2024 in all_gas_mixtures_2024.items():
        for abs_number, rates in gas_mixtures_2024.items():
            wps, background_rates = zip(*rates)
            label = f'{mixtures[gas]}' if gas not in plotted_labels else '_nolegend_'
            ax.scatter(background_rates, wps, marker=markers[gas], edgecolor=colors[gas], facecolor='none', label=label, s=marker_size)
            plotted_labels.add(gas)

    # Set plot properties on ax object
    ax.set_ylabel('Working point [kV]', fontproperties='DejaVu Sans', size=22, weight='bold')
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
    plt.ylim(6.7, 7.5) 
    plt.xlim(-0.1, 2.1)

    # Create custom legend for gas mixtures
    handles, labels = ax.get_legend_handles_labels()
    unique_labels = dict(zip(labels, handles))
    legend = ax.legend(unique_labels.values(), unique_labels.keys(), loc='upper left', fontsize=20)

    # Add custom markers for the text
    
    filled_circle = mlines.Line2D([], [], color='black', marker='o', linestyle='None', markersize=12)
    open_circle = mlines.Line2D([], [], color='black', marker='o', linestyle='None', markersize=12, markerfacecolor='none')

    # Add the custom markers to the plot with text
    ax.add_artist(ax.text(0.72, 0.81, 'Before irradiation', transform=ax.transAxes, fontsize=22, verticalalignment='top', horizontalalignment='left'))
    ax.add_artist(filled_circle)
    filled_circle.set_xdata([0.7])
    filled_circle.set_ydata([0.8])
    filled_circle.set_transform(ax.transAxes)

    ax.add_artist(ax.text(0.72, 0.76, 'After irradiation', transform=ax.transAxes, fontsize=22, verticalalignment='top', horizontalalignment='left'))
    ax.add_artist(open_circle)
    open_circle.set_xdata([0.7])
    open_circle.set_ydata([0.75])
    open_circle.set_transform(ax.transAxes)
    
    plt.text(1.65, 7.5+0.01, "GIF++", font0)
    hep.cms.text("Preliminary", fontsize=32)

    plt.savefig(os.path.join('Plots_23v24', "wp_vs_bkg2.png"))

if __name__ == "__main__":
    eff_wp_calc_2023 = read_data('2023.csv')
    eff_wp_calc_2024 = read_data('2024.csv')
    
    all_gas_mixtures_2023 = {gas: filter_gas_mixtures(eff_wp_calc_2023, gas) for gas in mixtures.keys()}
    all_gas_mixtures_2024 = {gas: filter_gas_mixtures(eff_wp_calc_2024, gas) for gas in mixtures.keys()}
    
    plot_background_rates(all_gas_mixtures_2023, all_gas_mixtures_2024)
