
# get signal from scipy for the filter
from scipy import signal
# get numpy libs
import numpy as np
# get plot libs
import matplotlib.pyplot as plt

# The Chebyshev type I filter maximizes the rate of cutoff between the frequency response’s passband and stopband,
# at the expense of ripple in the passband and increased ringing in the step response.
# Type I filters roll off faster than Type II (cheby2), but Type II filters do not have any ripple in the passband.

# Chebyshev type I digital and analog filter design

def cheby1_filter_design():

    n=10        # order of the filter
    rp=1        # maximum ripple allowed below unity gain in the bandpass. Specified in dB as positive number
    wc=25       # critical frequency
    type='low'  # lowpass’, ‘highpass’, ‘bandpass’, ‘bandstop’
    FS=1000     # The sampling frequency of the digital system.

    # Design an analog filter (analog = True)
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.cheby1.html
    # Output types: # ba’ = numerator/denominator, ‘zpk’=pole-zero, ‘sos’= second-order sections
    b, a = signal.cheby1(n, rp, wc, type, analog=True, output='ba')

    sos = signal.cheby1(n, rp, wc, type, fs=FS, output='sos')

    # Compute frequency response of analog filter.
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.freqs.html
    w, h = signal.freqs(b, a)

    # Compute group delay of the filter
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.group_delay.html
    wgd, gd = signal.group_delay((b, a), fs=FS)

    return w, h, wgd, gd, sos, wc, rp


def cheby1_filter_test():

    # Generate a signal made up of 3 Hz and 50 Hz, sampled at 1 kHz
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.cheby1.html
    t = np.linspace(0, 1, 1000, False)  # 1 second
    sig = np.sin(2*np.pi*3*t) + np.sin(2*np.pi*50*t)

    # Get the filter parameters
    w, h, wgd, gd, sos, wc, rp = cheby1_filter_design()

    # Filter it with sos
    filtered = signal.sosfilt(sos, sig)

    fig1, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    # plot the filter amplitude (frequency response)
    ax1.semilogx(w, 20 * np.log10(abs(h)))
    ax1.set_title('Chebyshev Type I frequency response')
    ax1.set_xlabel('Frequency [radians / second]')
    ax1.set_ylabel('Amplitude [dB]')
    ax1.margins(0, 0.1)
    ax1.grid(which='both', axis='both')
    ax1.axvline(wc, color='green') # cutoff frequency
    ax1.axhline(-rp, color='green') # rp
    # plot the group delay
    ax2.set_title('Digital filter group delay')
    ax2.plot(wgd, gd)
    ax2.set_ylabel('Group delay [samples]')
    ax2.set_xlabel('Frequency [rad/sample]')

    plt.show()

    # plot the original signal and the filtered signal
    fig2, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    ax1.plot(t, sig)
    ax1.set_title('3 Hz and 50 Hz sinusoids')
    ax1.axis([0, 1, -2, 2])
    ax2.plot(t, filtered)
    ax2.set_title('After 25 Hz Chebyshev Type I LP filter')
    ax2.axis([0, 1, -2, 2])
    ax2.set_xlabel('Time [seconds]')
    plt.tight_layout()
    plt.show()

