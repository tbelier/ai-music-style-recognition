import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':
    # Assuming au_feature_extractor has been executed
    file_path = '../build/blues-features.txt'

    up_to = 3000

    # Read data from the file
    try:
        fft = np.loadtxt(file_path)
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        exit()
    except ValueError:
        print(f"Error: Unable to parse data from '{file_path}'. Make sure the file contains valid numeric data.")
        exit()

    # fft is a list of 11026 coefficients corresponding to frequencies between 0 and 11025 Hz.
    frequency = np.arange(fft.size)
    normed_fft = fft / np.max(fft)

    # Plot the fft
    plt.plot(frequency[:up_to], normed_fft[:up_to], linewidth=0.2)
    # plt.scatter(frequency[:up_to], normed_fft[:up_to], s=0.1, alpha=0.5)
    plt.title("FFT blues signal")
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Magnitude')
    plt.show()
