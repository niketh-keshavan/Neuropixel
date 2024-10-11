import readSGLX
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as sig

start_t = 0  # Time range for data imported
duration = 10
plot_start_t = 0  # Time range for plotting the data
plot_duration = 10

# parameters for peak finding
peak_threshold = 400 * 1e-6
trough_threshold = 2*10**-6   # population rate must be lower than this value at some point between two bursts to avoid finding multiple peaks in one burst
min_separation = 0.5  # min separation (s) between 2 consecutive bursts


def find_peaks(arr):
    def find_chunks_indices_above_threshold(array, threshold):
        """
        Identifies the start and end indices of continuous chunks in the array
        where values are greater than the specified threshold.

        Parameters:
        - array: A list or 1D numpy array of numerical values.
        - threshold: A numerical value representing the threshold.

        Returns:
        - chunks_indices: A list of tuples where each tuple represents the start and
          end indices of a chunk (inclusive start, exclusive end) where values
          in the array are greater than the threshold.
        """

        chunks_indices = []  # To store the start and end indices of each chunk
        start = None  # To keep track of the starting index of the current chunk

        # Loop through the array, with i as the index and value as the element in the array
        for i, value in enumerate(array):
            if value > threshold:
                if start is None:
                    # Start a new chunk when we first find a value greater than the threshold
                    start = i  # Mark the starting index of the chunk
            else:
                if start is not None:
                    # When the value is not greater than the threshold, close the current chunk
                    chunks_indices.append((start, i))  # Record the start and end index of the chunk
                    start = None  # Reset start to None to detect the next chunk

        # After looping, check if the array ended while a chunk was still open
        if start is not None:
            # If start is not None, the chunk extended to the end of the array
            chunks_indices.append((start, len(array)))  # Record the final chunk, ending at the last index

        return chunks_indices

    def index_of_max_value(array, indices):
        """
        Find the index of the maximum value in 'array' limited to the specified 'indices'.

        Parameters:
        array (numpy.ndarray): The array to search in.
        indices (numpy.ndarray): The indices to consider in the search.

        Returns:
        int: The index of the maximum value within the specified indices of the array.
        """
        # Access values at the given indices
        values_at_indices = array[indices]

        # Find the index of the maximum value among these values
        max_value_index_among_indices = np.argmax(values_at_indices)
        max_value = indices[max_value_index_among_indices]

        return max_value

    rms = np.sqrt(np.mean(arr ** 2))
    print('rms: ', round(rms*10**6, 2), 'uV')

    # find_peaks returns list[array()], so zeroth element is the result
    raw_peaks = np.array(sig.find_peaks(arr, height=(peak_threshold, None), distance=min_separation * 1000)[0])

    # Make sure there must be troughs between two peaks to avoid multiple peaks within one burst
    highlands = find_chunks_indices_above_threshold(arr, trough_threshold)
    peaks = []
    for start, end in highlands:  # Loop over highlands and find the max value
        peaks_temp = raw_peaks[(raw_peaks >= start) & (raw_peaks <= end)]
        if len(peaks_temp) != 0:
            max_peak = index_of_max_value(arr, peaks_temp)
            if any(arr[max_peak:max_peak+round(0.2*sample_rate)] < arr[max_peak]-200*10**-6):
                peaks.append(max_peak)
    peaks = np.array(peaks)
    # print("Number of peaks: " + str(peaks.shape[0]))

    return peaks


# Main code to process the binary file and convert to voltages
bin_path = Path('C:\\Users\\niket\\Documents\\Neuropixel\\Heart_1\\240610_Heart1\\240610_Heart1_start0_end10.exported.imec0.ap.bin')
meta = readSGLX.readMeta(bin_path)
sample_rate = int(readSGLX.SampRate(meta))
print("Meta: ", meta)

# Create memory map of the raw data
data = readSGLX.makeMemMapRaw(bin_path, meta, start_t=start_t, duration=duration)
print("Raw data shape: ", data.shape)

# Get the list of channels
channels = readSGLX.OriginalChans(meta)
print('# of Channels: ', len(channels))

# Convert raw data to voltage values
data = readSGLX.GainCorrectIM(data, channels, meta)
print("Voltage data shape: ", data.shape, " (should match with raw data shape)")

# Average over channels, and subtract the average from channels (Glb all method from SpikeGLX)
data = data - np.mean(data, axis=0)


# Loop over each electrode to find bursts
peaks = [[] for _ in range(data.shape[0])]
for i, data_unit in enumerate(data):
    data_unit = data_unit - np.mean(data_unit)
    peaks_temp = find_peaks(data_unit)

    if len(peaks_temp) == 0:
        continue

    print('Threshold: ', round(peak_threshold * 1e6, 2), 'uV')
    print('Peaks: ', peaks_temp)

    # Plotting one channel for sanity check
    x = np.linspace(start_t, start_t+plot_duration, round(plot_duration * sample_rate))
    y = data_unit[plot_start_t * sample_rate: (plot_start_t+plot_duration) * sample_rate] * 10**6
    plt.plot(x, y, linewidth=0.5)
    try:
        plt.scatter(x[peaks_temp], y[peaks_temp], marker='x', color='orange')  # mark the peak of bursts
    except:
        pass
    plt.axhline(y=peak_threshold*10**6, color='r', linestyle='--')  # mark the threshold of detecting bursts
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.1)

    plt.xlabel('time(s)', fontsize=14)
    plt.ylabel('Population rate (kHz)', fontsize=14)
    plt.title('Channel' + str(i), fontsize=14)
    plt.tick_params(axis='x', labelsize=12)
    plt.tick_params(axis='y', labelsize=12)

    # plt.yticks(np.arange(-0.0006, 0.000601, 0.0002), np.arange(-600, 601, 200))
    plt.ylabel('Potential (uV)')
    plt.xlim(start_t, start_t + plot_duration)
    plt.ylim(-1000, 2500)
    plt.show()
    plt.close()

    # Extract the slices before and after peak time
    t_bef = 0.1
    t_aft = 0.5
    extracted_bursts = [data_unit[x-round(t_bef*sample_rate):x+round(t_aft*sample_rate)] for x in peaks_temp
                        if x-t_bef*sample_rate > 0 and x+t_aft*sample_rate<len(data_unit)]

    num_arrays = len(extracted_bursts)
    sigma = num_arrays / 4  # Approximation to cover most of the values within the range
    x = np.linspace(-2, 2, num_arrays)  # Creating 20 points from -3 to 3
    gaussian_weights = np.exp(-x ** 2 / 2) / (sigma * np.sqrt(2 * np.pi))
    gaussian_weights /= gaussian_weights.sum()  # Normalize the weights

    gaussian_sum_bursts = np.sum([w*arr for w, arr in zip(gaussian_weights, extracted_bursts)], axis=0)

    plt.plot(np.linspace(-t_bef, t_aft, round((t_bef + t_aft) * sample_rate))/1000, gaussian_sum_bursts*10**6, linestyle='-', color='black')
    for arr in extracted_bursts:
        plt.plot(np.linspace(-t_bef, t_aft, round((t_bef + t_aft) * sample_rate))/1000, arr*10**6, linestyle='--', color='black', alpha=0.1)
    plt.ylim(-1000, 1000)
    plt.show()
    plt.close()
