---
title: "De-Noising Quasar Spectra"
layout: post
date: 2019-07-13 11:10
tag: [machinelearning, regression, supervisedlearning, artificialintelligence]
#image: https://koppl.in/indigo/assets/images/jekyll-logo-light-solid.png
headerImage: false
projects: true
hidden: true # don't count this post in blog pagination
description: "Apply a supervised learning technique to estimate the light spectrum of quasars."
category: project
author: kaushkay
externalLink: false
---

<div style="text-align:center">![Quasar](/assets/images/projects/reg_for_quasar_spectra/quasar_2.jpg)</div>

A quasar can be defined as an extremely Active Galactic Nucleus (AGN). An AGN is nothing more than a supermassive black hole that is active and feeding at the center of a galaxy. They are extremely bright and sometimes mistaken for stars. However, the energy output of a star is nowhere near the amount of energy pumped out by a quasar. The word “quasar” originates from the contraction of “quasi-stellar”, which references a star-like radio source. 

---

### Properties of Quasars : 
The electromagnetic spectrum gives us the range of frequencies of different electromagnetic waves and their respective wavelengths. There are different electromagnetic wave regions, based on their frequency.

<div style="text-align:center">![Different Spectra](/assets/images/projects/reg_for_quasar_spectra/Electromagnetic-Spectrum-1.jpg)</div>

Quasars are known to emit electromagnetic radiation, which lies between the visible and X-ray regions. They also emit large amounts of ultraviolet waves.

Quasar Spectra for "Messier 31" Galaxy is shown below :

<div style="text-align:center">![Different Spectra](/assets/images/projects/reg_for_quasar_spectra/quasar_spec_messier.jpg)</div>

---

Understanding the properties of the spectrum of the light emitted by a quasar is useful for a number of tasks :
- A number of quasar properties can be estimated from the spectra.
- Properties of the regions of the universe through which the light passes can also be evaluated.
For example, we can estimate the density of neutral and ionized particles in the universe, which helps cosmologists understand the evolution and fundamental laws governing its structure.

The light spectrum is a curve that relates the light’s intensity (formally, lumens per square meter), or luminous flux, to its wavelength. The wavelengths are measured in Angstroms (°A), where 1°A= 10^(−10) meters.

<div style="text-align:center">![quasar_1](/assets/images/projects/reg_for_quasar_spectra/quasar_1.jpg)</div>

The blue line shows the intrinsic (i.e. original) flux spectrum emitted by the quasar. The red line denotes the observed spectrum here on Earth. To the left of the Lyman-α line, the observed flux is damped and the intrinsic (unabsorbed) flux continuum is not clearly recognizable (red line). To the right of the Lyman-α line, the observed flux approximates the intrinsic spectrum.

The Lyman-α wavelength is a wavelength beyond which intervening particles at most negligibly interfere with light emitted from the quasar. (Interference generally occurs when a photon is absorbed by a neutral hydrogen atom, which only occurs for certain wavelengths of light.) For wavelengths greater than this Lyman-α wavelength, the observed light spectrum f_obs can be modeled as a smooth spectrum f plus noise:

<div style="text-align:center">f_obs(λ) = f(λ) + noise(λ)</div>

For wavelengths below the Lyman-α wavelength, a region of the spectrum known as the Lyman-α forest, intervening matter causes attenuation of the observed signal. As light emitted by the quasar travels through regions of the universe richer in neutral hydrogen, some of it is absorbed, which we model as

<div style="text-align:center">f_obs(λ) = absorption(λ) · f(λ) + noise(λ)</div>

Astrophysicists and cosmologists wish to understand the absorption function, which gives infor- mation about the Lyman-α forest, and hence the distribution of neutral hydrogen in otherwise unreachable regions of the universe. This gives clues toward the formation and evolution of the universe. Thus, it is our goal to estimate the spectrum f of an observed quasar.

---

### Getting the data : 
Used data generated from the Hubble Space Telescope Faint Object Spectrograph (HST-FOS), Spectra of Active Galactic Nuclei and Quasars. [Dataset](https://github.com/kaushkay/denoising-quasar-spectra/tree/master/Data)

---

We wish to predict an entire part of a spectrum—a curve—from noisy observed data. We begin by supposing that we observe a random sample of m absorption-free spectra, which is possible for quasars very close (in a sense relative to the size of the universe!) to Earth. For a given spectrum f, define f_right to be the spectrum to the right of the Lyman-α line. Let f_left be the spectrum within the Lyman-α forest region, that is, for lower wavelengths. To make the results cleaner, we define:

<div style="text-align:center">f(λ) = ( f_left(λ) ; if λ < 1200  and f_right(λ) ; if λ ≥ 1300 )</div>

We will learn a function r (for regression) that maps an observed f_right to an unobserved target f_left (note that f_left and f_right don’t cover the entire spectrum). This is useful in practice because we observe f_right with only random noise: there is no systematic absorp- tion, which we cannot observe directly, because hydrogen does not absorb photons with higher wavelengths. By predicting f_left from a noisy version of f_right, we can estimate the unobservable spectrum of a quasar as well as the absorption function. Imaging systems collect data of the form 

<div style="text-align:center">f_obs(λ) = absorption(λ) · f(λ) + noise(λ)</div>

for λ ∈ {λ1, . . . , λn}, a finite number of points λ, because they must quantize the information. That is, even in the quasars-close-to-Earth training data, our observations of f_left and f_right consist of noisy evaluations of the true spectrum f at multiple wavelengths. In our case, we have n = 450 and λ1 = 1150, . . . , λn = 1599.

We formulate the functional regression task as the goal of learning the function r mapping f_right to f_left:

<div style="text-align:center">r(f_right)(λ) = E(f_left | f_right)(λ)</div>

To estimate the unobserved spectrum f_left of a quasar from its (noisy) observed spectrum f_right. To do so, we perform a weighted regression of the locally weighted regressions. In particular, given a new noisy spectrum observation:

<div style="text-align:center">f_obs(λ) = f(λ) + noise(λ)</div>

for λ ∈ {1300, . . . , 1599}.

We define a metric d which takes as input, two spectra f1 and f2, and outputs a scalar:

<div style="text-align:center">d(f_1, f_2) = \sum_{i} (f_1(λ_i) − f_2(λ_i))^2</div>

The metric d computes squared distance between the new datapoint and previous datapoints. If f1 and f2 are right spectra, then we take the preceding sum only over λ ∈ {1300, . . . , 1599}, rather than the entire spectrum.

Based on this distance function, we may define the nonparametric functional regression estimator, which is a locally weighted sum of functions f_left from the training data (this is like locally weighted linear regression, except that instead of predicting y ∈ R we predict a function f_left). Specifically, let f_right denote the right side of a spectrum, which we have smoothed using locally weighted linear regression (as you were told to do in the previous part of the problem). We wish to estimate the associated left spectrum f_left. Define the function ker(t) = max{1 − t, 0} and let neighb_k(f_right) denote the k indices i ∈ {1, 2, . . . ,m} of the training set that are closest to f_right.

---

###  Python Notebook: 

---

### References : 

- Ciollaro, Mattia, et al. “Functional regression for quasar spectra.” arXiv:1404.3168 (2014)
- <https://www.scienceabc.com/pure-sciences/what-are-quasars.html>
- <http://www.astrosurf.com/buil/galaxies/spectra.html>




---