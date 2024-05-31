# CO2 Analysis Codes

---

## Before Analysis

1. Make sure scans are inside a folder in the same directory as the scripts.
2. Change the directory paths in the code if needed to retrieve and store data/plots.
3. Update the scan names in the `extract_data.py` script with the appropriate ones that you are planning to analyze.
4. Run the `bckg_calculation.py` script; this will give you the gamma background rates at WP for each scan.

## For Plot Scripts

1. The `eff_plot` scripts will generate the efficiency plots (% efficiency at each HV point) for the selected scans.
2. The `gammacs_plot` script gives the gamma cluster size vs HV for the selected scans.
3. There are two types of scripts for each plot:
   - **ABS**: For plotting different gas mixtures but the same ABS filter.
   - **gas**: For plotting the same gas but different background rates.
   
   Use the corresponding one depending on the analysis and comparison you want.

4. Before running the above scripts, ensure that the labels are adjusted to the scans you are running (gas mixture, ABS filters, and TB dates).

Once you run the code, you will be prompted to enter the scan names you want to analyze. Note that the code allows a maximum of 5 scans.
