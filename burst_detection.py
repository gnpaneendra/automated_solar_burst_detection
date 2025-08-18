# Author: G N Paneendra

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
import sys
import os
import csv
from ultralytics import YOLO
from rich.progress import track

t1 = datetime.now()
print(f"\nStarting time: {t1.strftime('%H:%M:%S')}\n")

# For mean subtraction
def meansub(ch_v):
	ch_mean = np.mean(ch_v, axis=0)
	ch_v_mean_sub = ch_v - ch_mean
	return ch_v_mean_sub

# fft(ch1 voltage, ch2 volatage, fft length)
def fft(ch1_v, ch2_v, fft_length):
	ch1_spec = []
	ch2_spec = []
	cross_spec = []
	
	for anv in range(len(ch1_v[:,0])):			
		# Perform FFT
		fft_ch1 = np.fft.fft(ch1_v[anv,:])
		fft_ch2 = np.fft.fft(ch2_v[anv,:])
		N = len(ch1_v[anv,:])
		
		# Calculating the power
		ch1_mag = (np.abs(fft_ch1))**2 / N**2
		ch2_mag = (np.abs(fft_ch2))**2 / N**2
		  
		# Taking only the positive values
		ch1_spec.append(ch1_mag[:fft_length //2])
		ch2_spec.append(ch2_mag[:fft_length //2])
		   
		# Correlation
		cross_power = fft_ch1 * np.conj(fft_ch2)
		   
		# Calculate correlation power
		cross_spectrum_magnitude = (np.abs(cross_power))**2 / N**2
		    
		# Taking only the positive values
		cross_spec.append(cross_spectrum_magnitude[:fft_length // 2])
			
	ch1_spec = np.array(ch1_spec)
	ch2_spec = np.array(ch2_spec)
	cross_spec = np.array(cross_spec)
		
	return ch1_spec, ch2_spec, cross_spec

# Bandpass isolation and satellite RFI masking
def bpirm(freq, spec1, spec2, both):

	# Pass band isolation
	freq_mask = (freq >= 179e6) & (freq <= 361e6)
	bp_freq = freq[freq_mask]
	bp_spec1 = spec1[:, freq_mask]
	bp_spec2 = spec2[:, freq_mask]
	bp_both = both[:, freq_mask]

	# RFI mitigation
	for spec in(bp_spec1, bp_spec2, bp_both):			
		
		rfi_mask_range = (bp_freq >= 320e6) & (bp_freq <= 340e6)
		rfim = spec[:, rfi_mask_range]
		
		rfi_replace = np.median(rfim, axis=1)

		rfi_bands = [(180e6, 181.5e6), (190e6, 190.2e6),
			(197.6e6, 198.2e6), (199.88e6, 200.1e6), (209.8e6, 210.25e6),
			(219e6, 225e6), (229.85e6, 230.05e6),
			(243.2e6, 244.6e6), (243e6, 271e6)
			]

		for low, high in rfi_bands:
			rfi_band_mask = (bp_freq >= low) & (bp_freq <= high)
			rfirows = spec[:,rfi_band_mask]
			replace_block = np.tile(rfi_replace, (rfirows.shape[1], 1))
			replace_block = np.array(replace_block)
			replace_block = replace_block.T
			spec[:,rfi_band_mask] = replace_block
			
	return bp_freq, bp_spec1, bp_spec2, bp_both

def mad_rfi_mitigation(spectrum, threshold_multiplier):
	median = np.median(spectrum)
	mad = np.median(np.abs(spectrum - median))
	upper_threshold = median + threshold_multiplier * mad
	lower_threshold = median - threshold_multiplier * mad
	rfi_mask = (spectrum < lower_threshold) | (spectrum > upper_threshold)
	return rfi_mask, median

def apply_mad_rfi_sliding(data, threshold_multiplier=5, window_size=512, step_size=1):
	data = np.asarray(data)
	mask = np.zeros(len(data), dtype=bool)

	for i in range(0, len(data) - window_size + 1, step_size):
		window = data[i:i + window_size]
		rfi_mask, rfi_replace = mad_rfi_mitigation(window, threshold_multiplier)

		if np.any(rfi_mask):
			mask[i:i + window_size] |= rfi_mask
			data[i:i + window_size][rfi_mask] = rfi_replace

	return data

# To isolate the data between the specified time
def select_time(start_time, end_time, time, spec1, spec2, spec3):
	
	# Convert time strings to datetime
	start_time = datetime.strptime(start_time, "%H:%M:%S")
	end_time = datetime.strptime(end_time, "%H:%M:%S")
	
	# Extract just the time part
	time_only = [t.time() for t in time]
	start_time = start_time.time()
	end_time = end_time.time()
	
	# Boolean mask for selecting time range
	mask = [(t >= start_time and t <= end_time) for t in time_only]
	indices = np.where(mask)[0]
	
	if len(indices) == 0:
		raise ValueError("No data in the selected time range.")
	
	begin = indices[0]
	end = indices[-1] + 1
	
	spec1_selected = np.array(spec1[begin:end])
	spec2_selected = np.array(spec2[begin:end])
	spec3_selected = np.array(spec3[begin:end])
	selected_time = time[begin:end]
	
	return spec1_selected, spec2_selected, spec3_selected, selected_time

# Function to plot
def plot(spec, freq, time, vl, vh, name, name2, model, csv_writer=None):
	
	# TO PRODUCE SPECTROGRAPH
	plt.figure(figsize=(12, 7))
	plt.imshow(spec.T, aspect='auto',
		cmap='plasma', vmin=vl, vmax=vh,
		extent=[time[0], time[-1], freq[0], freq[-1]], origin='lower'
		)
	#plt.colorbar(label='Power')
	plt.title(f"{date}, {start_time_str} - {end_time_str}, Gauribidanur Two-Element Interferometer Spectrograph (RRI)",
		fontsize=14
		)
	plt.xlabel('Time (UTC)', fontsize=16)
	plt.ylabel('Frequency (MHz)', fontsize=16)
	plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x * 1e-6:.0f}'))
	plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
	plt.xticks(fontsize=16)
	plt.yticks(fontsize=16)
	plt.tight_layout()
	
	filename = f"{date}_{start_time_str}_{end_time_str}_{name2}.png"
	plt.savefig(filename, dpi=200)
	print(f"\nDisplaying {date}_{start_time_str}_{end_time_str}_{name2}")
	plt.show()
	plt.close()
	
	image = [filename]
	
	results = model(image, conf=0.05)
	
	# Detection process timing information
	speed = results[0].speed
	preprocess_time = speed['preprocess']
	inference_time = speed['inference']
	postprocess_time = speed['postprocess']
	total_time = preprocess_time + inference_time + postprocess_time
	if csv_writer:
		csv_writer.writerow(
			{'Image name': f"{date}_{start_time_str}_{end_time_str}_{name2}.png",
			'Preprocess Time (ms)': round(preprocess_time, 2),
			'Inference Time (ms)': round(inference_time, 2),
			'Postprocess Time (ms)': round(postprocess_time, 2),
			'Total Time (ms)': round(total_time, 2)
		})

	results[0].save(filename.replace('.png', '_out.png'))
	
	plt.figure(figsize=(12, 7))
	plt.imshow(results[0].plot())
	plt.axis("off")
	print(f"\nDisplaying solar burst detection in {date}_{start_time_str}_{end_time_str}_{name2}")
	plt.show()
	plt.close()

path = r'burst_detection_using_yolo/2024_02_12.csv' # Set the location to the data
#path = str(sys.argv[1])

# Load the trained model
model = YOLO('burst_detection_using_yolo/runs/detect/train/weights/best.pt') # Set the location to the weights

file_name = os.path.basename(os.path.normpath(path))
print(f"\nCurrent file: {file_name}")

print("\nReading the file...", end=" ")
df = pd.read_csv(path)
ch1_v = df.iloc[:, 1:9953]
ch2_v = df.iloc[:, 9953:df.shape[1]]
date_time = df.iloc[:, 0]

ch1_v = np.array(ch1_v)
ch2_v = np.array(ch2_v)
utc_time = np.array([datetime.strptime(t, "%Y-%m-%d %H:%M:%S") for t in date_time])
print("Done")

date = utc_time[0].strftime('%Y-%m-%d')
time_1 = utc_time[0].strftime('%H:%M:%S')
time_n = utc_time[-1].strftime('%H:%M:%S')
print(f"\nObservation date: {date}")
print(f"Observation time(UT): {time_1} - {time_n}")

print("\nPerforming mean subtraction...", end=" ")
# Mean subtraction
ch1_v_mean_sub = meansub(ch1_v)
ch2_v_mean_sub = meansub(ch2_v)
print("Done")

print("\nPerforming RFI mitigation...\n")
ch1_v_rfi_mitigated = []
for i in track(ch1_v_mean_sub, description="CH1 RFI Mitigation"):
	rfi_mitigated = apply_mad_rfi_sliding(i)
	ch1_v_rfi_mitigated.append(rfi_mitigated)
ch1_v_rfi_mitigated = np.array(ch1_v_rfi_mitigated)

ch2_v_rfi_mitigated = []
for i in track(ch2_v_mean_sub, description="CH2 RFI Mitigation"):
	rfi_mitigated = apply_mad_rfi_sliding(i)
	ch2_v_rfi_mitigated.append(rfi_mitigated)

ch2_v_rfi_mitigated = np.array(ch2_v_rfi_mitigated)
print("\nDone")

print("\nPerforming FFT... ", end=" ")
sampling_rate = 1.25e9
fft_length = len(ch1_v_rfi_mitigated[0])
    
ch1_spec, ch2_spec, cross_spec = fft(
 	ch1_v_rfi_mitigated, ch2_v_rfi_mitigated, fft_length
   	)

frequencies = np.fft.fftfreq(fft_length, 1/sampling_rate)[:fft_length//2]
print("Done")

print("\nBand pass isolation... ", end=" ")
bp_frequencies, ch1_bp_spec, ch2_bp_spec, cross_bp_spec = bpirm(
	frequencies, ch1_spec, ch2_spec, cross_spec
	)
bpcn = np.arange(len(bp_frequencies))
print("Done")

utc_start = datetime.strptime(date_time[0], "%Y-%m-%d %H:%M:%S")
utc_end = datetime.strptime(date_time[len(date_time)-1], "%Y-%m-%d %H:%M:%S")

interval = timedelta(minutes=60)

# Create the .csv files required for logging detetcion info
csv_file_1 = f"{date}_channel_1_detection_log.csv"
csv_file_2 = f"{date}_channel_2_detection_log.csv"
csv_file_3 = f"{date}_correlated_detection_log.csv"
fieldnames = ['Image name',
	'Preprocess Time (ms)',
	'Inference Time (ms)',
	'Postprocess Time (ms)',
	'Total Time (ms)'
	]

log_writers = {}
for fname in [csv_file_1, csv_file_2, csv_file_3]:
    f = open(fname, mode='w', newline='')
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    log_writers[fname] = (f, writer)

interval = timedelta(minutes=60)

# Loop through the observation time range in 60 min intervals
current_start = utc_start

while current_start < utc_end:
    current_end = current_start + interval

    # Format time strings as needed by select_time function
    start_time_str = current_start.strftime("%H:%M:%S")
    end_time_str = current_end.strftime("%H:%M:%S")
    
    print(f"\nProcessing data between {start_time_str} - {end_time_str}")
    
    try:
    	ch1_spec_selected, ch2_spec_selected, crc_spec_selected, selected_time = select_time(
        	start_time_str, end_time_str, utc_time,
        	ch1_bp_spec, ch2_bp_spec, cross_bp_spec
        	)
    except ValueError:
        print(f"\nSkipping interval {start_time_str} to {end_time_str} (No data)")
        current_start = current_end
        continue

    plot(ch1_spec_selected, bp_frequencies, selected_time, 0, 0.02,
    	'Channel 1', 'channel_1', model,
    	csv_writer=log_writers[csv_file_1][1]
    	)
    	
    plot(ch2_spec_selected, bp_frequencies, selected_time, 0, 0.018,
    	'Channel 2', 'channel_2', model,
    	csv_writer=log_writers[csv_file_2][1]
    	)
    	
    plot(crc_spec_selected, bp_frequencies, selected_time, 0, 8e3, 
    	'Correlated', 'correlated', model,
    	csv_writer=log_writers[csv_file_3][1]
    	)

    current_start = current_end

# Close the .csv files
for f, writer in log_writers.values():
    f.close()

t2 = datetime.now()
print(f"\nEnding time: {t2.strftime('%H:%M:%S')}")

tf = t2 - t1
print(f"\nTotal time: {tf}\n")
