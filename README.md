CO2 Analysis codes:
--------------------------------------------------------------------------------------------------
Before analysis:
make sure scans are inside a folder in the same directory as the scripts
change the directory paths if need be on the codes (to retrive and store data/plots)
update the scan names in the 'extract_data.py' script with the apropiate ones that you are planing on analyzing
run the 'bckg_calculation.py', this will give you the gamma background rates at WP for each scan

For plot scripts:
'eff_plot' scripts will generate the efficiency plots (% efficincy at each HV point) for the selected scans
'gammacs_plot' script gives the gamma cluster size vs HV for the selected scans

there are two types of sript for each, one ending in 'ABS' the other ending in 'gas'. The one named 'ABS' is for plotting different gas mixtures but the same ABS filter. The script ending in 0gas' is for plotting same gas but different background rates. use the corresponding one depending on the analysis and comparison you want.

before running the above codes, make sure that the labels are adjusted to the scans you are running (gas mixtrue, ABS filters and TB dates)
once you run the code you can see it will ask for the scan names you want to analyis, this code allows a maximum of 5 scans
