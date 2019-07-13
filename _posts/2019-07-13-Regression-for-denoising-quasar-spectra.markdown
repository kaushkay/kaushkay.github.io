---
title: "De-Noising Quasar Spectra"
layout: post
date: 2019-07-13 11:10
tag: [machine learning, regression, supervised learning, artificial intelligence]
#image: https://koppl.in/indigo/assets/images/jekyll-logo-light-solid.png
headerImage: false
projects: true
hidden: true # don't count this post in blog pagination
description: "Apply a supervised learning technique to estimate the light spectrum of quasars."
category: project
author: kaushkay
externalLink: false
---

![Quasar](/assets/images/projects/reg_for_quasar_spectra/quasar_2.jpg)

A quasar can be defined as an extremely Active Galactic Nucleus (AGN). An AGN is nothing more than a supermassive black hole that is active and feeding at the center of a galaxy. They are extremely bright and sometimes mistaken for stars. However, the energy output of a star is nowhere near the amount of energy pumped out by a quasar. The word “quasar” originates from the contraction of “quasi-stellar”, which references a star-like radio source. 

---

### Properties of Quasars : 
The electromagnetic spectrum gives us the range of frequencies of different electromagnetic waves and their respective wavelengths. There are different electromagnetic wave regions, based on their frequency.

![Different Spectra](/assets/images/projects/reg_for_quasar_spectra/Electromagnetic-Spectrum-1.jpg)

Quasars are known to emit electromagnetic radiation, which lies between the visible and X-ray regions. They also emit large amounts of ultraviolet waves.

Quasar Spectra for "Messier 31" Galaxy is shown below :

![Different Spectra](/assets/images/projects/reg_for_quasar_spectra/quasar_spec_messier.jpg)

---

Understanding the properties of the spectrum of the light emitted by a quasar is useful for a number of tasks :
- A number of quasar properties can be estimated from the spectra.
- Properties of the regions of the universe through which the light passes can also be evaluated.
For example, we can estimate the density of neutral and ionized particles in the universe, which helps cosmologists understand the evolution and fundamental laws governing its structure.

The light spectrum is a curve that relates the light’s intensity (formally, lumens per square meter), or luminous flux, to its wavelength. The wavelengths are measured in Angstroms (°A), where 1°A= 10^(−10) meters.

![quasar_1](/assets/images/projects/reg_for_quasar_spectra/quasar_1.jpg)

The blue line shows the intrinsic (i.e. original) flux spectrum emitted by the quasar. The red line denotes the observed spectrum here on Earth. To the left of the Lyman-α line, the observed flux is damped and the intrinsic (unabsorbed) flux continuum is not clearly recognizable (red line). To the right of the Lyman-α line, the observed flux approximates the intrinsic spectrum.

The Lyman-α wavelength is a wavelength beyond which intervening particles at most negligibly interfere with light emitted from the quasar. (Interference generally occurs when a photon is absorbed by a neutral hydrogen atom, which only occurs for certain wavelengths of light.) For wavelengths greater than this Lyman-α wavelength, the observed light spectrum fobs can be modeled as a smooth spectrum f plus noise:

f_obs(λ) = f(λ) + noise(λ)

For wavelengths below the Lyman-α wavelength, a region of the spectrum known as the Lyman-α forest, intervening matter causes attenuation of the observed signal. As light emitted by the quasar travels through regions of the universe richer in neutral hydrogen, some of it is absorbed, which we model as

f_obs(λ) = absorption(λ) · f(λ) + noise(λ)

Astrophysicists and cosmologists wish to understand the absorption function, which gives infor- mation about the Lyman-α forest, and hence the distribution of neutral hydrogen in otherwise unreachable regions of the universe. This gives clues toward the formation and evolution of the universe. Thus, it is our goal to estimate the spectrum f of an observed quasar.

---

### Getting the data : 
Used data generated from the Hubble Space Telescope Faint Object Spectrograph (HST-FOS), Spectra of Active Galactic Nuclei and Quasars. [Link]()

---

### References : 

- Ciollaro, Mattia, et al. “Functional regression for quasar spectra.” arXiv:1404.3168 (2014)
- <https://www.scienceabc.com/pure-sciences/what-are-quasars.html>
- <http://www.astrosurf.com/buil/galaxies/spectra.html>




---

[Check it out](http://sergiokopplin.github.io/indigo/) here.
If you need some help, just [tell me](http://github.com/sergiokopplin/indigo/issues).
