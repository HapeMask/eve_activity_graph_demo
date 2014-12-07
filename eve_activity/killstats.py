import numpy as np

from .vmfmm import VMFMM

def compute_peak_times(hours, n_peaks):
    """Compute n_peaks peak activity times for a list of kill hours.
    
    Parameters
    ----------
    hours : list, values in floating point range [0,24).
        List of kill times (hours of the day, 24H clock).

    n_peaks : int
        Number of activity peaks to compute.

    Returns
    -------
        peaks : list
            A list of n_peaks peak activity time estimates for the given times.
    """

    # Find the peak times by converting times to angles and fitting a vMF
    # mixture to them, then converting the vMF means back to times.
    hours = np.asarray(hours, np.float)
    time_angles = np.pi*((hours / 12.) - 1)

    mm = VMFMM(n_components=n_peaks, n_init=5).fit(time_angles[:,np.newaxis])
    peak_means = mm.means_[np.argsort(mm.weights_)[::-1]]

    peak_angles = np.arctan2(peak_means[:,1], peak_means[:,0])
    return list(12*((peak_angles / np.pi) + 1))

def compute_avg_kills_by_hour(timestamps, hours):
    """Computes the average number of kills made in each hour of the day.

    Parameters
    ----------
    timestamps : list of ints
        List of kill timestamps in seconds-since-epoch format.

    hours : list, values in floating point range [0,24).
        List of kill times (hours of the day, 24H clock).

    Returns
    -------
    kills_per_hr : list, length=24
        List of average kills within each 1hr period from 0:00 - 23:59.
    """
        
    timestamps = np.asarray(timestamps, np.int)
    hours = np.asarray(hours, np.float)

    hists = np.array([np.histogram(hours[timestamps == ut],
                     bins=24, range=(0,23))[0]
             for ut in np.unique(timestamps)])

    return list(hists.mean(axis=0))
