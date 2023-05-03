#!/bin/bash

# Copyright 2023 Juliane Mai - contact(at)juliane-mai(dot)com
#
# License
# This file is part of Juliane Mai's personal code library.
#
# Juliane Mai's personal code library is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Juliane Mai's personal code library is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.

# You should have received a copy of the GNU Lesser General Public License v2.1
# along with Juliane Mai's personal code library.  If not, see <http://www.gnu.org/licenses/>.
#
# run with:
#     ./plot.sh

set -ex
#
# Produces plots for Calibration Practices paper
#

# Switches
dotex=1       	   #   LaTeX fonts in plots

dofig1=0           #   schematic for checklist and cycle of calibration experiments
dofig2=0           #   results of hydrologic calibration experiments w/ and w/o screening of parameters
dofig3=0           #   sampling of tied parameters and parameters with constraints
dofig4=0           #   differences when data span orders of magnitude
dofig5=0           #   results of calibrating with old or new data (cal/val/testing)
dofig6=0           #   random sampling vs stratified sampling vs automatic calibration
dofig7=0           #   results of calibration experiments using different ranges for parameters
dofig8=0           #   results of hydrologic calibration experiments using different metrics
dofig9=0           #   results of hydrologic calibration experiments using different calibration algorithms
dofig10=0          #   results of hydrologic calibration experiments using SO and MO algorithms
dofig11=0          #   schematic sub-figures that will get combined to figure 9 via LaTeX

dofig8sub=1           #   results of hydrologic calibration experiments using different metrics


verbose=2 # 0: pipe stdout and stderr to /dev/null
          # 1: pipe stdout to /dev/null
          # 2: print everything

# pathes
inputpath='../data'
outpath='scripts/'

# pdf margins
pdfmargins=3

# Treat switches
if [[ ${dotex} -eq 1 ]] ; then
    texit='-u'
else
    texit=''
fi

pipeit=''
if [[ ${verbose} -eq 0 ]] ; then pipeit=' > /dev/null 2>&1' ; fi
if [[ ${verbose} -eq 1 ]] ; then pipeit=' > /dev/null' ; fi

# Figures

if [[ ${dofig1} -eq 1 ]] ; then
    echo ''
    echo 'figure 1 in progress...'

    # make LaTeX schematic
    cd ${outpath}/figure_1/
    pdflatex figure_1.tex  ${pipeit}
    mv figure_1.pdf ../.
    cd -

    # crop
    pdfcrop ${outpath}/figure_1.pdf
    mv ${outpath}/figure_1-crop.pdf figures/figure_1.pdf

    # cleanup
    rm ${outpath}/figure_1.pdf
    rm ${outpath}/figure_1/figure_1.log
    rm ${outpath}/figure_1/figure_1.aux
fi

if [[ ${dofig2} -eq 1 ]] ; then
    echo ''
    echo 'figure 2 in progress...'
    python ${outpath}/figure_2.py -t pdf -p "${outpath}/figure_2" ${texit}
    pdfcrop --margins ${pdfmargins} ${outpath}/figure_2.pdf ${pipeit}
    mv ${outpath}/figure_2-crop.pdf ${outpath}/../figures/figure_2.pdf
    rm ${outpath}/figure_2.pdf
fi

if [[ ${dofig3} -eq 1 ]] ; then
    echo ''
    echo 'figure 3 in progress...'
    python ${outpath}/figure_3.py -t pdf -p "${outpath}/figure_3" ${texit}
    pdfcrop --margins ${pdfmargins} ${outpath}/figure_3.pdf ${pipeit}
    mv ${outpath}/figure_3-crop.pdf ${outpath}/../figures/figure_3.pdf
    rm ${outpath}/figure_3.pdf
fi

if [[ ${dofig4} -eq 1 ]] ; then
    echo ''
    echo 'figure 4 in progress...'
    python ${outpath}/figure_4.py -t pdf -p "${outpath}/figure_4" ${texit}
    pdfcrop --margins ${pdfmargins} ${outpath}/figure_4.pdf ${pipeit}
    mv ${outpath}/figure_4-crop.pdf ${outpath}/../figures/figure_4.pdf
    rm ${outpath}/figure_4.pdf
fi

if [[ ${dofig5} -eq 1 ]] ; then
    echo ''
    echo 'figure 5 in progress...'

    # make LaTeX schematic
    cd ${outpath}/figure_5/
    pdflatex figure_5.tex  ${pipeit}
    mv figure_5.pdf ../.
    cd -

    # crop
    pdfcrop ${outpath}/figure_5.pdf
    mv ${outpath}/figure_5-crop.pdf figures/figure_5.pdf

    # cleanup
    rm ${outpath}/figure_5.pdf
    rm ${outpath}/figure_5/figure_5.log
    rm ${outpath}/figure_5/figure_5.aux
fi

if [[ ${dofig6} -eq 1 ]] ; then
    echo ''
    echo 'figure 6 in progress...'
    python ${outpath}/figure_6.py -t pdf -p "${outpath}/figure_6" ${texit}
    pdfcrop --margins ${pdfmargins} ${outpath}/figure_6.pdf ${pipeit}
    mv ${outpath}/figure_6-crop.pdf ${outpath}/../figures/figure_6.pdf
    rm ${outpath}/figure_6.pdf
fi

if [[ ${dofig7} -eq 1 ]] ; then
    echo ''
    echo 'figure 7 in progress...'
    python ${outpath}/figure_7.py -t pdf -p "${outpath}/figure_7" ${texit}
    pdfcrop --margins ${pdfmargins} ${outpath}/figure_7.pdf ${pipeit}
    mv ${outpath}/figure_7-crop.pdf ${outpath}/../figures/figure_7.pdf
    rm ${outpath}/figure_7.pdf
fi

if [[ ${dofig8} -eq 1 ]] ; then
    echo ''
    echo 'figure 8 in progress...'
    python ${outpath}/figure_8.py -t pdf -p "${outpath}/figure_8" ${texit}
    pdfcrop --margins ${pdfmargins} ${outpath}/figure_8.pdf ${pipeit}
    mv ${outpath}/figure_8-crop.pdf ${outpath}/../figures/figure_8.pdf
    rm ${outpath}/figure_8.pdf
fi

if [[ ${dofig8sub} -eq 1 ]] ; then
    echo ''
    echo 'figure 8 in progress...'
    python ${outpath}/figure_8_subset.py -t pdf -p "${outpath}/figure_8_subset" ${texit}
    pdfcrop --margins ${pdfmargins} ${outpath}/figure_8_subset.pdf ${pipeit}
    mv ${outpath}/figure_8_subset-crop.pdf ${outpath}/../figures/figure_8_subset.pdf
    rm ${outpath}/figure_8_subset.pdf
fi

if [[ ${dofig9} -eq 1 ]] ; then
    echo ''
    echo 'figure 9 in progress...'
    python ${outpath}/figure_9.py -t pdf -p "${outpath}/figure_9" ${texit}
    pdfcrop --margins ${pdfmargins} ${outpath}/figure_9.pdf ${pipeit}
    mv ${outpath}/figure_9-crop.pdf ${outpath}/../figures/figure_9.pdf
    rm ${outpath}/figure_9.pdf
fi

if [[ ${dofig10} -eq 1 ]] ; then
    echo ''
    echo 'figure 10 in progress...'
    python ${outpath}/figure_10.py -t pdf -p "${outpath}/figure_10" ${texit}
    pdfcrop --margins ${pdfmargins} ${outpath}/figure_10.pdf ${pipeit}
    mv ${outpath}/figure_10-crop.pdf ${outpath}/../figures/figure_10.pdf
    rm ${outpath}/figure_10.pdf
fi

if [[ ${dofig11} -eq 1 ]] ; then
    echo ''
    echo 'figure 11 in progress...'
    python ${outpath}/figure_11.py -t pdf -p "${outpath}/figure_11" ${texit}
    pdfcrop --margins ${pdfmargins} ${outpath}/figure_11.pdf ${pipeit}
    mv ${outpath}/figure_11-crop.pdf ${outpath}/figure_11/figure_11.pdf
    rm ${outpath}/figure_11.pdf

    # split
    pdfsplit ${outpath}/figure_11/figure_11.pdf ${outpath}/figure_11/figure_11-

    # make LaTeX schematic
    cd ${outpath}/figure_11/
    pdflatex figure_11.tex  ${pipeit}
    mv figure_11.pdf ../.
    cd -

    # crop
    pdfcrop ${outpath}/figure_11.pdf
    mv ${outpath}/figure_11-crop.pdf figures/figure_11.pdf

    # cleanup
    rm ${outpath}/figure_11.pdf
    rm ${outpath}/figure_11/figure_11.log
    rm ${outpath}/figure_11/figure_11.aux
fi
