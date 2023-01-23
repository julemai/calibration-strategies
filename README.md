# Ten strategies towards successful calibration of environmental models

This repository contains the data and scripts associated with the publication on "Ten strategies towards successful calibration of environmental models". 

## Requirements
The codes are mostly in Python. The Python environment can be build using [requirements.txt](). Some examples require an excecutable of the hydrologic modeling framework [Raven]() to be available. Please download a Raven exectuable from here or compile the code yourself. Then name the executable "Raven.exe" and place it under ``. All examples contain the prederived outputs used in the publications (no need to have Raven etc available). They are all called `figure_XX.json` where XX denotes the figure that will be produced using these data.

## Generate figures
The script to produce the figures is `plot.sh`. Please select the figure you want to produce in the first lines of that scipts by setting `dofigXX=1` to 1. If `dofigXX=0` the figure will not be produced. 

## Citation

### Journal Publication
Mai, J. (2023).<br>
Ten strategies towards successful calibration of environmental models. <br>
*Journal of Hydrology*, XX(XX), XX-XX.<br>
https://doi.org/XX/XX


### Code and Data Publication
Mai, J (2023).<br>
Ten strategies towards successful calibration of environmental models. <br>
*Zenodo*<br>
[![DOI](https://zenodo.org/badge/XX.svg)](https://zenodo.org/badge/latestdoi/XX)

