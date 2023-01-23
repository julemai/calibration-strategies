# Ten strategies towards successful calibration of environmental models

This repository contains the data and scripts associated with <i>"Ten strategies towards successful calibration of environmental models"</i> by Mai et al. (2023).

<p align="center">
   <img alt="The calibration life cycle" src="https://github.com/julemai/calibration-strategies/blob/main/figures/figure_1.png" width="65%" />
</p>

## Requirements
Figures 1, 5, and 11 are flowcharts and schematic overviews. They are
derived using LaTeX. It requires LaTeX to be installed including
several packages such as TikZ.

Figures 2-4, and 6-10 are produced through calibration experiments. The
setups and results used for the publication are provided as
`scripts/example_XX.zip`. Please unzip them and run the experiments again in
case you want to test. Be aware that some examples require an
excecutable of the hydrologic modeling framework
[Raven](http://raven.uwaterloo.ca/) to be available. Please download a
Raven exectuable from [here](http://raven.uwaterloo.ca/Downloads.html)
or compile the code yourself. Then replace all Raven executables in
the example folders with your executable.

However, all outputs of the examples 2-4 and 6-10 have been saved and
stored in a way that the figures can be created without running the
experiments, i.e. no need to run Raven. These **pre-derived outputs**
were used in the publication and are called `scripts/figure_XX.json` where XX
denotes the figure that will be produced using these data.

The codes to produce these figures are in Python. The Python
environment can be build using
[requirements.txt](https://github.com/julemai/calibration-strategies/requirements.txt) 
or setup a Python environment manually using:

```
pyenv virtualenv 3.8.5 env-3.8.5-calib-pub
pyenv activate env-3.8.5-calib-pub

pip install argparse
pip install numpy
pip install scipy
pip install matplotlib
pip install lhsmdu
```

## Generate figures
The script to produce the figures is `plot.sh`. Please select the
figure you want to produce in the first lines of that scipts by
setting `dofigXX=1` to 1. If `dofigXX=0` the figure will not be
produced. The figues will be placed in `figures/` after creation.

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
