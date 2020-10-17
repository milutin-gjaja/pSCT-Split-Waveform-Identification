import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import seaborn as sns # makes prettier plots than standard matplotlib
from tqdm import tqdm # will provide a progress bar
import target_io # one of the basic tools that we installed, used to read data
sns.set_style("whitegrid")
import utils
import os
import time
from numba import njit

grid_ind = utils.get_grid_ind()




def graph_split(event, pix, runID, graph=False, correct=False):
    """
    A function that takes a number of events/pixels in a run and returns those which are splits. Can also graph and/or
    correct the split waveforms. Calls idef_split to identify the split waveforms as well as Target to get waveform/
    phase information.

    :param event: array of events
    :param pix: array of pixels using the pixel ID number (goes from 0 to 1599)
    :param runID: int, run number
    :param graph: if set to True, graphs each split waveform identified
    :param correct: if set to True, "corrects" the split waveform by inversing its first and third blocks
    :return: split_pix, an array with size (2, len(event), # of splits in that event)
    """
    datadir = "/data/wipac/CTA/target5and7data/runs_320000_through_329999"
    cal = f"{datadir}/cal{runID}.r1"
    calreader = target_io.WaveformArrayReader(cal)
    n_pixels = calreader.fNPixels
    n_events = calreader.fNEvents
    n_samples = calreader.fNSamples

    split_pix = [[], []]

    id_pix = [grid_ind[i] for i in range(1536)]
    translated = [id_pix.index(i) for i in pix]

    col1 = fits.Column(name = "Events", format = "F", array = np.zeros((len(event),)))
    t = fits.BinTableHDU.from_columns([col1])
    t.writeto("run_{}_split_corrected.fits".format(runID))
    
    for idef, ev in enumerate(event):
        cal_waveforms = np.zeros((n_pixels, n_samples), dtype=np.float32)
        calreader.GetR1Event(ev, cal_waveforms)
        calreader.GetBlock(ev)
        phase_points = [i for i in range(128) if (i + calreader.fPhase) % 32 == 0]
        split_pix[0].append(ev)

        ev_splits = []

        for i in range(len(pix)):
            if idef_split(cal_waveforms[translated[i]], calreader.fPhase) == True:
                ev_splits.append(pix[i])
                if correct == True:
                    cal_waveforms[translated[i]] = correct_split(cal_waveforms[translated[i]], phase_points)
                if graph == True:
                    fig = plt.figure()
                    ax = plt.subplot(1, 1, 1)
                    ax.set_ylabel("Sample value (ADC units)")
                    ax.set_xlabel("Sample (time in ns)")
                    plt.title("Run: " + str(runID) + " Event: " + str(ev) + " Pixel ID: " + str(pix[i]))

                    ax.plot(np.arange(len(cal_waveforms[translated[i]])), cal_waveforms[translated[i]], 'r',
                            label="Waveform")
                    ax1 = ax.twinx()
                    ax1.plot(np.arange(len(cal_waveforms[translated[i]])),
                             np.abs(np.gradient(cal_waveforms[translated[i]])), 'bo', label="Derivative")
                    for item in phase_points:
                        plt.axvline(x=item)
                    # ax.legend()
                    # ax1.legend()
                    plt.show()
                    
        fits.append("run_{}_split_corrected.fits".format(runID), cal_waveforms, hdr = "Events")       
        split_pix[1].append(ev_splits)
        
    return split_pix





@njit
def idef_split(pixel: np.ndarray, phase: int) -> bool:
    """
    A function that identifies whether a waveform is a split or not. The main points:
    - Checks if there are groups of consecutive samples in the calibrated waveform above a certain threshold ("peaks"),
    then check if those peaks are contained within two blocks
    - Checks for spikes in the derivative of 1 - 2 samples, then checks if those spikes are on or right before the
    beginning/end of blocks, and if those blocks are separated by 64 ns (i.e. if the blocks are consecutive)

    :param pixel: a (128,) array which has the waveform information for a single pixel in a single event
    :param phase: the phase information for the event in question
    :return: returns True if the waveform is a split, False if not
    """
    deriv = np.abs(deriv_1d(pixel))
    phase_jumps = []
    peaks = []
    threshold = 100

    i = 0
    while i < len(pixel):
        if pixel[i] > threshold:
            temp = []
            for j in range(i, len(pixel)):
                if pixel[j] > threshold:
                    temp.append(j)
                else:
                    peaks.append(temp)
                    i = j
                    break
        i += 1

    if len(peaks) != 2 or len([i for i in peaks if len(i) > 2]) != len(peaks):
        return False
        
    for i in range(1, len(deriv)):
        if (deriv[i] + deriv[i - 1])/4 > 90:
            if (i + phase) % 32 == 0:
                phase_jumps.append(i)
            else:
                return False

    if len(phase_jumps) != 2:
        return False
    elif phase_jumps[1] - phase_jumps[0] != 64:
        return False
    else:
        for item in peaks:
            for samp in item:
                if samp in range(phase_jumps[0] - 1, phase_jumps[-1] + 2):
                    continue
                else:
                    return False

    return True



@njit
def deriv_1d(f: np.ndarray) -> np.ndarray:
    """
    A quick function that calculates the derivative for a given array, to be used when plotting waveforms. This was
    created due to a lack of compatibility between numba and np.gradient
    
    :param f: A single-dimensional array
    :return: grad, an array of the same size of f that contains the values of the derivative at each point
    """
    grad = np.zeros(len(f))
    grad[0] = f[1] - f[0]
    grad[-1] = f[-1] - f[-2]
    for i in range(1, len(f)-1):
        grad[i] = ((f[i] - f[i-1]) + (f[i+1] - f[i]) / 2.)
    return grad



def correct_split(pix_arr, phase_points):
    """
    A basic correction for splits. Breaks the waveform into three complete blocks and two incomplete blocks ("block1",
    "block2", "block3", "beg", and "end), then exchanges the first complete block with the third.

    :param pix_arr: A (128,) array which has the waveform information for a single pixel in a single event
    :param phase_points:
    :return: pix_arr, modified so that the first and third complete blocks are inversed
    """
    block1 = pix_arr[phase_points[0]:phase_points[1]]
    block2 = pix_arr[phase_points[1]:phase_points[2]]
    block3 = pix_arr[phase_points[2]:phase_points[3]]
    beg = pix_arr[:phase_points[0]]
    end = pix_arr[phase_points[3]:]
    pix_arr = np.concatenate((beg, block3, block2, block1, end))
    return pix_arr



def flasher_pulse(event, verb=False):
    """
    A basic flasher event check based on the percentage of pixels that peak above a certain threshold. Not fantastically
    accurate but filters out a lot if needed.

    :param event: A (1536, 128) array with the waveforms for each pixel in an event
    :param verb: If True, will print text while functioning to tell the user how many pixels are above the threshold
    :return: True if it identifies a flasher event, False if not
    """
    max_pix = []
    for pixel in event:
        max_pix.append(max(pixel))
    above_thresh = [i for i in max_pix if i > 300]
    if len(above_thresh) > len(max_pix) * 0.2:
        if verb == True:
            print("This is a flasher event")
            print(len(above_thresh), "out of", len(max_pix), "are above the threshold")
            print("The brightest pixel is", max_pix.index(np.max(max_pix)), "with", np.max(max_pix), "units")
        return True
    if verb == True:
        print("This is not a flasher event")
        print(len(above_thresh), "out of", len(max_pix), "are above the threshold")
    return False


def graph_wf(event, pix, runID, calcheck=True, uncalcheck=True, ex_flash=False):
    """
    A catch-all graphing function that produces graphs based on the number of events/pixels input. Unless there are
    multiple events and pixels, the waveforms will be superimposed on the graph for easy comparison. Note: this function
    takes a pixel array with pixel numbers ranging from 0-1535, not the pixel ID. This needs to be changed in the future

    :param event: An array with the events to graph
    :param pix: An array with the pixels to graph. These are NOT pixel IDs
    :param runID: An integer designating the run used
    :param calcheck: Graphs the calibrated waveform. Set by default to True
    :param uncalcheck: Graphs the uncalibrated waveform. Set by default to True
    :param ex_flash: Checks if the event is a flasher event before graphing. Set by default to false
    :return: Function doesn't turn anything, only prints the graph. Needs to be changed in the future
    """
    datadir = "/data/wipac/CTA/target5and7data/runs_320000_through_329999"
    uncal = f"{datadir}/run{runID}.fits"
    cal = f"{datadir}/cal{runID}.r1"
    uncalreader = target_io.WaveformArrayReader(uncal)
    calreader = target_io.WaveformArrayReader(cal)
    n_pixels = calreader.fNPixels
    n_events = calreader.fNEvents
    n_samples = calreader.fNSamples
    cal_waveforms = np.zeros((n_pixels, n_samples), dtype=np.float32)
    uncal_waveforms = np.zeros((n_pixels, n_samples), dtype=np.ushort)

    trans = 1
    if len(event) in [i for i in range(3, 15)] or len(pix) in [i for i in range(3, 15)]:
        trans = 0.3
    elif len(event) >= 15 or len(pix) >= 15:
        trans = 0.1

    if not (len(event) > 1 and len(pix) > 1):
        fig = plt.figure(figsize=(11, 11))
        ax = plt.subplot(1, 1, 1)
        ax.set_ylabel("Sample value (ADC units)")
        ax.set_xlabel("Sample (time in ns)")
        if len(event) > 1:
            plt.title(
                "Run: " + str(runID) + " Events: " + str(event[0]) + "-" + str(event[-1]) + " Pixel: " + str(pix[0]))
        elif len(pix) > 1:
            plt.title(
                "Run: " + str(runID) + " Event: " + str(event[0]) + " Pixels: " + str(pix[0]) + "-" + str(pix[-1]))
        else:
            plt.title("Run: " + str(runID) + " Event: " + str(event[0]) + " Pixel: " + str(pix[0]))

    for ev in event:
        calreader.GetR1Event(ev, cal_waveforms)
        uncalreader.GetR0Event(ev, uncal_waveforms)
        calreader.GetBlock(ev)
        phase_points = [i for i in range(128) if (i + calreader.fPhase) % 32 == 0]
        if (flasher_pulse(cal_waveforms) == False and ex_flash == True) or ex_flash == False:
            if len(event) > 1 and len(pix) > 1:
                fig = plt.figure()
                ax = plt.subplot(1, 1, 1)
                ax.set_ylabel("Sample value (ADC units)")
                ax.set_xlabel("Sample (time in ns)")
                plt.title("Run: " + str(runID) + " Event: " + str(ev) + " Pixels: " + str(pix[0]) + "-" + str(pix[-1]))
            if calcheck == True:
                for item in pix:
                    ax.plot(np.arange(len(cal_waveforms[item])), cal_waveforms[item], 'r', alpha=trans)
            if uncalcheck == True:
                for item in pix:
                    ax.plot(np.arange(len(uncal_waveforms[item])), uncal_waveforms[item], 'b', alpha=trans)
    plt.show()

    
    
   
def camera_splits(results):
    """
    A useful visualization function that graphs the 40 by 40 camera image and highlights the areas with a large number
    of splits. vmax, the highest value shown, is hardcoded to 10 for use with a logarithmic scale, but this can be
    changed.
    
    :param results: A (1600, 2) array with an ordered list of pixels and the corresponding number of splits
    """
    image = np.zeros((40, 40))
    for i, val in enumerate(results):
        image[i//40, i%40] = val
    
#     fig = plt.figure(figsize = (10, 10))
#     im = plt.imshow(image, origin = "lower", vmin = 0, vmax = 10.4, cmap = "viridis")
#     fig.colorbar(im)

    x_list = []
    y_list = []
    x_list_pos = []
    y_list_pos = []
    for m in range(5):
        for k in range(5):
            x_list.append(np.asarray([i - 20 + 0.31 * ((i // 9) - 2) -1*(i//9) for i in range(9*k,9*(k+1))]).T)
            y_list.append(np.asarray([j - 20 + 0.31 * ((j // 9) - 2) -1*(j//9) for j in range(9*m,9*(m+1))]).T)
            x_list_pos.append(np.asarray([i for i in range(8*k,8*(k+1))]).T)
            y_list_pos.append(np.asarray([j for j in range(8*m,8*(m+1))]).T)

    fig = plt.figure(figsize=(10,10))
    ax = plt.subplot(1,1,1)
    ax.set_xlim(-25,25)
    ax.set_ylim(-25,25)
    for k in range(25):
        xx, yy = np.meshgrid(x_list[k],y_list[k])
        xx_pos, yy_pos = np.meshgrid(x_list_pos[k],y_list_pos[k])
        ax.pcolor(xx,yy,image[yy_pos,xx_pos],vmin=0,vmax=10, cmap="viridis")
        x_B = y_B = np.asarray([i - 19.5 + 0.31 * ((i // 8) - 2) for i in range(40)]).T
        x_B_mg, y_B_mg = np.meshgrid(x_B,y_B)
        ax.plot(x_B_mg, y_B_mg, 'o',color='white', markersize=2)
        ax.plot(0.,0.,'r+',markersize=1000)
    plt.show()
    
    
    
"""
Main
"""

run = 328719
event = [i for i in range(100, 200)] #change the range for different number of events
pix = [grid_ind[i] for i in range(1536)]
results = graph_split(event, pix, run, correct = True)
