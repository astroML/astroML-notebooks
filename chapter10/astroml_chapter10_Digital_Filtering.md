---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.11.1
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

+++ {"id": "RY3UTJn5xS61"}

[![Open in Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/astroML/astroML-notebooks/main?filepath=chapter10/astroml_chapter10_Digital_Filtering.ipynb)

# Digital Filtering

+++ {"id": "rUtO_Om0xS61"}

## Introduction
**Digital filtering** aims to reduce noise in time series data, or to compress data. Common examples
include low-pass filtering, where high frequencies are suppressed, high-pass filtering, where low
frequencies are suppressed, passband filtering, where only a finite range of frequencies is admitted,
and a notch filter, where a finite range of frequencies is blocked. We will use a few examples to illustrate the most common
applications of filtering.  
Fourier analysis is one of the most useful tools for performing filtering. Numerous other techniques can be found in signal processing literature, including approaches based on the wavelets discussed in the modeling toolkit notebook.   
  
We emphasize that filtering always decreases the information content of data (despite making
it appear less noisy). As we have already learned throughout previous chapters, when model
parameters are estimated from data, raw (unfiltered) data should be used. In some sense, this is
an analogous situation to binning data to produce a histogram-while very useful for visualization, estimates of model parameters can become biased if one is not careful.

+++ {"id": "X56Sp8FkxS61"}

### Import packages and data
In this notebook, we are going to explore the astroML.filters. We mainly use the Savitzky-Golay and the Wiener filters. The spectrum of a white dwarf data imported for this notebook is from Sloan Digital Sky
Survey (SDSS).

```{code-cell} ipython3
:id: Tudp07WPxS61

import numpy as np
from matplotlib import pyplot as plt

from astroML.fourier import PSD_continuous
from astroML.datasets import fetch_sdss_spectrum

from scipy import optimize, fftpack, interpolate
from scipy.signal import savgol_filter
from astroML.fourier import IFT_continuous
from astroML.filters import wiener_filter
from astroML.filters import min_component_filter
```

+++ {"id": "OCNftVWHxS62"}

## 1. Low-pass filters
The power spectrum for common Gaussian noise is 
at and will extend to frequencies as high as
the Nyquist limit, $f_N = 1=(2\Delta t)$. If the data are band limited to a lower frequency, $f_c < f_N$, then
they can be smoothed without much impact by suppressing frequencies $|f| > f_c$. Given a filter
in frequency space, $\Phi (f)$, we can obtain a smoothed version of data by taking the inverse Fourier
transform of

$$\hat{Y}(f)=Y(f) \Phi(f)$$

where $Y(f)$ is the discrete Fourier transform of data. At least in principle, we could simply set
$\Phi(f)$ to zero for $|f| > f_c$, but this approach would result in ringing (i.e., unwanted oscillations)
in the signal. Instead, the optimal filter for this purpose is constructed by minimizing the MISE
between $\hat{Y}(f)$ and $Y(f)$ (for detailed derivation see NumRec) and is called the **Wiener filter**:

$$\Phi(f) = \frac{P_S(f)}{P_S(f)+P_N(f)}$$

Here $P_S(f)$ and $P_N(f)$ represent components of a two-component (signal and noise) fit to the
PSD of input data, $PSD_Y (f) = P_S(f) + P_N(f)$, which holds as long as the signal and noise are
uncorrelated. We will see how the filtering works in the example below.

+++ {"id": "Rh2qPRLhxS62"}

### Create the noisy data
We are going to generate a set of noisy data as the input signal, on which we apply filters. The figure below shows the input data (200 evenly spaced points) with a narrow Gaussian peak centered at x = 20.

```{code-cell} ipython3
:id: Yod0vWt_xS62
:outputId: ce2a8ca0-e165-482a-b306-2dae0d9b2e52

# Create the noisy data
np.random.seed(5)
N = 2000
dt = 0.05

t = dt * np.arange(N)
h = np.exp(-0.5 * ((t - 20.) / 1.0) ** 2)
hN = h + np.random.normal(0, 0.5, size=h.shape)

# Plot the results
N = len(t)
Df = 1. / N / (t[1] - t[0])
f = fftpack.ifftshift(Df * (np.arange(N) - N / 2))
HN = fftpack.fft(hN)

fig = plt.figure(figsize=(5, 3.75))
fig.subplots_adjust(wspace=0.05, hspace=0.35,
                    bottom=0.1, top=0.95,
                    left=0.12, right=0.95)

# First plot: noisy signal
ax = fig.add_subplot(111)
ax.plot(t, hN, '-', c='gray')
ax.plot(t, np.zeros_like(t), ':k')
ax.text(0.98, 0.95, "Input Signal", ha='right', va='top',
        transform=ax.transAxes, bbox=dict(fc='w', ec='none'))

ax.set_xlim(0, 90)
ax.set_ylim(-0.5, 1.5)

ax.xaxis.set_major_locator(plt.MultipleLocator(20))
ax.set_xlabel(r'$\lambda$')
ax.set_ylabel('flux')
```

+++ {"id": "vV9tc7gkxS64"}

### Set up the Wiener filter
We fit a model to the PSD consisting of the sum of a gaussian and white noise using **Wiener filter**. We will see this method in the later sections in this notebook.  

```{code-cell} ipython3
:id: iGIXtsKtxS64

N = 2000

Df = 1. / N / dt
f = fftpack.ifftshift(Df * (np.arange(N) - N / 2))
HN = fftpack.fft(hN)

# apply the Wiener filter
h_smooth, PSD, P_S, P_N, Phi = wiener_filter(t, hN, return_PSDs=True)
```

+++ {"id": "RSdbDG5dxS64"}

### Set up the Savitzky-Golay filter
We use a fourth-order **Savitzky-Golay** filter with a window size of $\Delta \lambda = 10$ to filter the vales. 
The Savitzky-Golay filter is a very simple but powerful method as a low-pass filter. It fits low-order polynomials
to data (in the time domain) using sliding windows (it is also known as the least-squares filter).
For a detailed discussion, see NumRec.

```{code-cell} ipython3
:id: Htv8fcpcxS64

# apply the Savitzky-Golay filter
h_sg = savgol_filter(hN, window_length=201, polyorder=4, mode='mirror')
```

+++ {"id": "v0acUi29xS65"}

### Show filtered signal
Plot below shows noisy signal after filtering. 
* Result from Wiener filter is shown in black.
* Result from Savitzky-Golay filter is shown in gray.  

The Gaussian peak at x=20 is clearly seen in both curves.

```{code-cell} ipython3
:id: NtV9u_PixS65
:outputId: 170a42af-8047-4378-e68e-b23cb15afe12

# Second plot: filtered signal
ax = plt.subplot(111)
ax.plot(t, np.zeros_like(t), ':k', lw=1)
ax.plot(t, h_smooth, '-k', lw=1.5, label='Wiener')
ax.plot(t, h_sg, '-', c='gray', lw=1, label='Savitzky-Golay')

ax.text(0.98, 0.95, "Filtered Signal", ha='right', va='top',
        transform=ax.transAxes)
ax.legend(loc='upper right', bbox_to_anchor=(0.98, 0.9), frameon=False)

ax.set_xlim(0, 90)
ax.set_ylim(-0.5, 1.5)

ax.xaxis.set_major_locator(plt.MultipleLocator(20))
ax.set_xlabel(r'$\lambda$')
ax.set_ylabel('flux')
```

+++ {"id": "v4S1w9XdxS65"}

### Show filtered PSD
* The upper panel shows the input power spectral density (PSD) distribution.  
* The lower panel shows the Wiener-filtered power spectral density (PSD) distributions.  

The two curves in the upper panel represent two-component fit to PSD given by equation

$$\Phi(f) = \frac{P_S(f)}{P_S(f)+P_N(f)}$$

```{code-cell} ipython3
:id: WMuHlbmxxS65
:outputId: 6e335310-ce32-47fd-cee0-a30b8c038649

# Plot the results
N = len(t)
Df = 1. / N / (t[1] - t[0])
f = fftpack.ifftshift(Df * (np.arange(N) - N / 2))
HN = fftpack.fft(hN)

fig = plt.figure(figsize=(5, 3.75))
fig.subplots_adjust(wspace=0.05, hspace=0.35,
                    bottom=0.1, top=0.95,
                    left=0.12, right=0.95)

# Third plot: Input PSD
ax = fig.add_subplot(211)
ax.scatter(f[:N // 2], PSD[:N // 2], s=9, c='k', lw=0)
ax.plot(f[:N // 2], P_S[:N // 2], '-k')
ax.plot(f[:N // 2], P_N[:N // 2], '-k')

ax.text(0.98, 0.95, "Input PSD", ha='right', va='top',
        transform=ax.transAxes)

ax.set_ylim(-100, 3500)
ax.set_xlim(0, 0.9)

ax.yaxis.set_major_locator(plt.MultipleLocator(1000))
ax.xaxis.set_major_locator(plt.MultipleLocator(0.2))
ax.set_xlabel('$f$')
ax.set_ylabel('$PSD(f)$')

# Fourth plot: Filtered PSD
ax = fig.add_subplot(212)
filtered_PSD = (Phi * abs(HN)) ** 2
ax.scatter(f[:N // 2], filtered_PSD[:N // 2], s=9, c='k', lw=0)

ax.text(0.98, 0.95, "Filtered PSD", ha='right', va='top',
        transform=ax.transAxes)

ax.set_ylim(-100, 3500)
ax.set_xlim(0, 0.9)

ax.yaxis.set_major_locator(plt.MultipleLocator(1000))
ax.xaxis.set_major_locator(plt.MultipleLocator(0.2))
ax.set_xlabel('$f$')
ax.set_ylabel('$PSD(f)$')
```

+++ {"id": "o4jfFOcTxS66"}

### Wiener Filter and kernel smoothing Connection
There is an interesting connection between the kernel density estimation method discussed in the KDE notebook
and Wiener filtering. By the convolution theorem, the Wiener-filtered result is equivalent to the
convolution of the unfiltered signal with the inverse Fourier transform of $\Phi(f)$.  

This convolution is equivalent to kernel density estimation. When Wiener filtering is viewed in this way, it effectively says that we believe the signal is as wide as the central
peak, and the statistics of the noise are such that the minor peaks in the
wings work to cancel out noise in the major peak.  

Hence, the modeling of the PSD in the frequency
domain via 

$$\Phi(f) = \frac{P_S(f)}{P_S(f)+P_N(f)}$$ 

corresponds to choosing the optimal kernel width. Just as detailed modeling
of the Wiener filter is not of paramount importance, the choice of kernel is not either.

We will use the same data as the previous Wiener filter figure as an example to explore this connection.

+++ {"id": "F7xOtMVCxS66"}

### Find effective kernel

```{code-cell} ipython3
:id: z2Ku2w21xS66

# inverse fourier transform Phi to find the effective kernel
t_plot, kernel = IFT_continuous(f, Phi)
```

+++ {"id": "OoRvHj9ixS66"}

### perform kernel smoothing
This is faster in frequency
space (i.e. using the standard Wiener filter above) but we will do it in the slow & simple way here to demonstrate the equivalence explicitly.

```{code-cell} ipython3
:id: XxHPo9gExS66

kernel_func = interpolate.interp1d(t_plot, kernel.real)

t_eval = np.linspace(0, 90, 1000)
t_KDE = t_eval[:, np.newaxis] - t
t_KDE[t_KDE < t_plot[0]] = t_plot[0]
t_KDE[t_KDE > t_plot[-1]] = t_plot[-1]
F = kernel_func(t_KDE)

h_smooth = np.dot(F, hN) / np.sum(F, 1)
```

+++ {"id": "3X8gmvTrxS66"}

### Show kernel and smoothing results
* The left panel shows the inverse Fourier transform of the Wiener filter $\Phi(f)$ applied in the sample data we used previously.  
* The right panel shows the data smoothed by this kernel, which is equivalent to the Wiener filter smoothing in the previous figure.  

By the convolution theorem, the Wiener-filtered result is equivalent to the convolution of the unfiltered signal with
the kernel shown above, and thus Wiener filtering and kernel density estimation (KDE) are directly related. 

```{code-cell} ipython3
:id: 0PiN2ZBExS66
:outputId: c0d7be9a-7290-4ac2-bfbb-98963289ad33

# Plot the results
fig = plt.figure(figsize=(10, 4.4))
fig.subplots_adjust(left=0.1, right=0.95, wspace=0.25,
                    bottom=0.15, top=0.9)

# First plot: the equivalent Kernel to the WF
ax = fig.add_subplot(121)
ax.plot(t_plot, kernel.real, '-k')
ax.text(0.95, 0.95, "Effective Wiener\nFilter Kernel",
        ha='right', va='top', transform=ax.transAxes)

ax.set_xlim(-10, 10)
ax.set_ylim(-0.05, 0.45)
ax.set_xlabel(r'$\lambda$')
ax.set_ylabel(r'$K(\lambda)$')

# Second axes: Kernel smoothed results
ax = fig.add_subplot(122)
ax.plot(t_eval, h_smooth, '-k', lw=1)
ax.plot(t_eval, 0 * t_eval, '-k', lw=1)
ax.text(0.95, 0.95, "Kernel smoothing\nresult",
        ha='right', va='top', transform=ax.transAxes)

ax.set_xlim(0, 90)
ax.set_ylim(-0.5, 1.5)

ax.set_xlabel('$\lambda$')
ax.set_ylabel('flux')

plt.show()
```

+++ {"id": "9CbCW5s_xS66"}

## 2. High-pass filters
The most common example of high-pass filtering in astronomy is baseline estimation in spectral
data. Unlike the case of low-pass filtering, here there is no universal filter recipe. Baseline estimation
is usually the first step toward the estimation of model parameters (e.g. location, width, and strength of spectral lines). In such cases, the best approach might be full modeling and
marginalization of baseline parameters as nuisance parameters at the end of analysis.  
  
A simple iterative technique for high-pass filtering, called **minimum component filtering**, is discussed
in detail in WJ03. These are the **main steps**:
1. Determine baseline: exclude or mask regions where signal is clearly evident and fit a baseline
model (e.g., a low-order polynomial) to the unmasked regions.
2. Get FT for the signal: after subtracting the baseline fit in the unmasked regions (i.e., a linear
regression fit), apply the discrete Fourier transform.
3. Filter the signal: remove high frequencies using a low-pass filter (e.g., Wiener filter), and
inverse Fourier transform the result.
4. Recombine the baseline and the filtered signal: add the baseline fit subtracted in step 2 to
the result from step 3. This is the minimum component filtering estimate of baseline.  
  
In the next two examples, we will see the application of a minimum component filter to the spectrum of a white dwarf.

+++ {"id": "Ts8lXF3hxS66"}

## Example 1 (Show steps)
### Fetch the data
We first fetch the spectrum data from SDSS database for use. The intermediate steps of the minimum component filter procedure applied to the spectrum of a white dwarf from the SDSS data set (mjd= 52199, plate=659, fiber=381).

```{code-cell} ipython3
:id: WX3tvLb7xS66
:outputId: 184871ff-d340-4d1c-d39c-f4ecfec41177

# Fetch the spectrum from SDSS database
plate = 659
mjd = 52199
fiber = 381

data = fetch_sdss_spectrum(plate, mjd, fiber)

lam = data.wavelength()
spec = data.spectrum
```

+++ {"id": "uzcLa8RvxS67"}

### Pre-process the data
Wavelengths we get are logorithmically spaced: we will work in log(lam).

```{code-cell} ipython3
:id: KjdsQTRJxS67

def preprocess(lam, spec):
    loglam = np.log10(lam)

    flag = (lam > 4000) & (lam < 5000)
    lam = lam[flag]
    loglam = loglam[flag]
    spec = spec[flag]

    lam = lam[:-1]
    loglam = loglam[:-1]
    spec = spec[:-1]
    
    return [lam, loglam, spec]

[lam, loglam, spec] = preprocess(lam, spec)
```

+++ {"id": "8J04jlS3xS67"}

### Apply minimum component filtering steps

+++ {"id": "FkglHwn3xS67"}

First step: mask-out significant features

```{code-cell} ipython3
:id: beRD9woQxS67

feature_mask = (((lam > 4080) & (lam < 4130)) |
                ((lam > 4315) & (lam < 4370)) |
                ((lam > 4830) & (lam < 4900)))
```

+++ {"id": "zEoHf1TFxS67"}

Second step: fit a line to the unmasked portion of the spectrum

```{code-cell} ipython3
:id: iGTGscg4xS67
:outputId: 7d8064b0-7076-4a32-85d0-c3d750411fd2

XX = loglam[:, None] ** np.arange(2)
beta = np.linalg.lstsq(XX[~feature_mask], spec[~feature_mask], rcond=None)[0]

spec_fit = np.dot(XX, beta)
spec_patched = spec - spec_fit
spec_patched[feature_mask] = 0
```

+++ {"id": "SfSdQzOAxS67"}

Third step: Fourier transform the patched spectrum

```{code-cell} ipython3
:id: J1-cXKPTxS68

N = len(loglam)
df = 1. / N / (loglam[1] - loglam[0])
f = fftpack.ifftshift(df * (np.arange(N) - N / 2.))
spec_patched_FT = fftpack.fft(spec_patched)
```

+++ {"id": "0VlkhtydxS68"}

Fourth step: Low-pass filter on the transform

```{code-cell} ipython3
:id: T6PPvAOvxS68

filt = np.exp(- (0.01 * (abs(f) - 100.)) ** 2)
filt[abs(f) < 100] = 1

spec_filt_FT = spec_patched_FT * filt
```

+++ {"id": "D8qhl6V9xS68"}

Fifth step: inverse Fourier transform, and add back the fit

```{code-cell} ipython3
:id: tzK1151FxS68

spec_filt = fftpack.ifft(spec_filt_FT)
spec_filt += spec_fit
```

+++ {"id": "ipNe0YhKxS68"}

### Show filter result
* The top panel shows the input spectrum; the masked sections of the input spectrum are shown by thin lines (i.e., step 1 of the process). 
* The bottom panel shows the PSD of the masked spectrum, after the linear fit has been subtracted (gray line). 
* A simple low-pass filter (dashed line) is applied, and the resulting filtered spectrum (dark line) is used to construct the result shown in the next figure.

```{code-cell} ipython3
:id: m_SNHlE2xS68
:outputId: 8f496cd1-6cd7-4974-c3a1-d8044b89749a

# plot results
fig = plt.figure(figsize=(5, 3.75))
fig.subplots_adjust(hspace=0.45)

ax = fig.add_subplot(211)
ax.plot(lam, spec, '-', c='gray')
ax.plot(lam, spec_patched + spec_fit, '-k')

ax.set_ylim(25, 110)
ax.set_xlim(4000, 5000)

ax.set_xlabel(r'$\lambda\ {\rm(\AA)}$')
ax.set_ylabel('flux')

ax = fig.add_subplot(212)
factor = 15 * (loglam[1] - loglam[0])
ax.plot(fftpack.fftshift(f),
        factor * fftpack.fftshift(abs(spec_patched_FT) ** 1),
        '-', c='gray', label='masked/shifted spectrum')
ax.plot(fftpack.fftshift(f),
        factor * fftpack.fftshift(abs(spec_filt_FT) ** 1),
        '-k', label='filtered spectrum')
ax.plot(fftpack.fftshift(f),
        fftpack.fftshift(filt), '--k', label='filter')

ax.set_xlim(0, 2000)
ax.set_ylim(0, 1.1)

ax.set_xlabel('$f$')
ax.set_ylabel('scaled $PSD(f)$')
```

+++ {"id": "fX9AKH9XxS69"}

## Example 2 (use package)
### Fetch the data
We follow the same process to analyze the same white dwarf example. Here instead of explicitly applying steps, we call function *min_component_filter* to achieve the same goal.

```{code-cell} ipython3
:id: _877Orm7xS69

plate = 659
mjd = 52199
fiber = 381

data = fetch_sdss_spectrum(plate, mjd, fiber)

lam = data.wavelength()
spec = data.spectrum
```

+++ {"id": "IU9mYPyexS69"}

### Pre-process the data

```{code-cell} ipython3
:id: SRil6iHMxS69

[lam, loglam, spec] = preprocess(lam,spec)
```

+++ {"id": "YWBCxlnaxS69"}

### Apply minimum component filtering using function
The function used here is *min_component_filter* in *astroML.filters* package.

```{code-cell} ipython3
:id: U5fxYgtWxS69

feature_mask = (((lam > 4080) & (lam < 4130)) |
                ((lam > 4315) & (lam < 4370)) |
                ((lam > 4830) & (lam < 4900)))

spec_filtered = min_component_filter(loglam, spec, feature_mask, fcut=100)
```

+++ {"id": "OfgWgHbwxS69"}

### Compute PSD of filtered and unfiltered versions

```{code-cell} ipython3
:id: e86wfwZExS69

f, spec_filt_PSD = PSD_continuous(loglam, spec_filtered)
f, spec_PSD = PSD_continuous(loglam, spec)
```

+++ {"id": "D_xm9FppxS69"}

### Show filter result
* The upper panel shows a portion of the input spectrum, along with the continuum computed via the minimum component filtering procedure described above (See the previous figure). 
* The lower panel shows the PSD for both the input spectrum and the filtered result.

```{code-cell} ipython3
:id: BCvqIFTSxS69

# Plot the results
fig = plt.figure(figsize=(5, 3.75))
fig.subplots_adjust(hspace=0.45)

# Top panel: plot noisy and smoothed spectrum
ax = fig.add_subplot(211)
ax.plot(lam, spec, '-', c='gray', lw=1)
ax.plot(lam, spec_filtered, '-k')

ax.text(0.97, 0.93, "SDSS white dwarf\n %i-%i-%i" % (mjd, plate, fiber),
        ha='right', va='top', transform=ax.transAxes)

ax.set_ylim(25, 110)
ax.set_xlim(4000, 5000)

ax.set_xlabel(r'$\lambda\ {\rm (\AA)}$')
ax.set_ylabel('flux')

# Bottom panel: plot noisy and smoothed PSD
ax = fig.add_subplot(212, yscale='log')
ax.plot(f, spec_PSD, '-', c='gray', lw=1)
ax.plot(f, spec_filt_PSD, '-k')

ax.set_xlabel(r'$f$')
ax.set_ylabel('$PSD(f)$')
ax.set_xlim(0, 2000)
```

```{code-cell} ipython3
:id: 4Rq-shUYxS6-


```
