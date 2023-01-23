#!/usr/bin/env python

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
#     run figure_9.py -t pdf -p figure_9

#!/usr/bin/env python
from __future__ import print_function

"""

Plots results of hydrologic calibration experiments using different calibration algorithms

History
-------
Written,  JM, July 2022
"""

# -------------------------------------------------------------------------
# Command line arguments - if script
#

# Comment|Uncomment - Begin
if __name__ == '__main__':

    import argparse
    import numpy as np

    plotname    = ''
    outtype     = ''
    usetex      = False
    serif       = False

    parser   = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                      description='''Plots results of hydrologic calibration experiments using different calibration algorithms.''')
    parser.add_argument('-p', '--plotname', action='store',
                        default=plotname, dest='plotname', metavar='plotname',
                        help='Name of plot output file for types pdf, html or d3, '
                        'and name basis for type png (default: '+__file__[0:__file__.rfind(".")]+').')
    parser.add_argument('-s', '--serif', action='store_true', default=serif, dest="serif",
                    help="Use serif font; default sans serif.")
    parser.add_argument('-t', '--type', action='store',
                        default=outtype, dest='outtype', metavar='outtype',
                        help='Output type is pdf, png, html, or d3 (default: open screen windows).')
    parser.add_argument('-u', '--usetex', action='store_true', default=usetex, dest="usetex",
                        help="Use LaTeX to render text in pdf, png and html.")

    args     = parser.parse_args()
    plotname = args.plotname
    outtype  = args.outtype
    serif    = args.serif
    usetex   = args.usetex

    del parser, args
    # Comment|Uncomment - End


    # -----------------------
    # add subolder scripts/lib to search path
    # -----------------------
    import sys
    import os
    dir_path = os.path.dirname(os.path.realpath(__file__))
    sys.path.append(dir_path+'/lib')


    # -------------------------------------------------------------------------
    # Function definition - if function
    #

    # Check input
    outtype = outtype.lower()
    outtypes = ['', 'pdf', 'png', 'html', 'd3']
    if outtype not in outtypes:
        print('\nError: output type must be in ', outtypes)
        import sys
        sys.exit()

    import numpy as np
    import time
    import json as json
    from json import JSONEncoder
    import copy
    import datetime as datetime

    import color                                          # in lib/
    from position          import position                # in lib/
    from str2tex           import str2tex                 # in lib/
    from autostring        import astr                    # in lib/
    from abc2plot          import abc2plot                # in lib/
    from read_raven_output import read_raven_hydrograph   # in lib/

    t1 = time.time()

    if (outtype == 'd3'):
        try:
            import mpld3
        except:
            print("No mpld3 found. Use output type html instead of d3.")
            outtype = 'html'

    class NumpyArrayEncoder(JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return JSONEncoder.default(self, obj)


    # -------------------------------------------------------------------------
    # Setup
    #
    dowhite    = False  # True: black background, False: white background
    title      = False   # True: title on plots, False: no plot titles
    textbox    = False  # if true: additional information is set as text box within plot
    textbox_x  = 0.95
    textbox_y  = 0.85

    # -------------------------------------------------------------------------
    # Setup Calculations
    #
    if dowhite:
        fgcolor = 'white'
        bgcolor = 'black'
    else:
        fgcolor = 'black'
        bgcolor = 'white'


    # fix seed for reproducible results
    np.random.seed(seed=123)

    jsonfile = 'figure_9.json'

    # -------------------------------------------------------------------------
    # general
    # -------------------------------------------------------------------------

    ntrials = 10

    # name of examples
    example_names_str = [ 'Log-transformed data fit with\nlogistic function ($n_P = 4$)', 'Hydrologic model GR4J for\nSalmon River ($n_P = 9$)', 'Ackley benchmark test\nfunction ($n_P = 20$)' ]  # for plots
    example_names_str = [ 'Logistic function\n($n_P = 4$)', 'Hydrologic model\n($n_P = 9$)', 'Ackley function\n($n_P = 10$)', 'Ackley function\n($n_P = 20$)' ]  # for plots
    example_names     = [ 'ostrich-dispersivity', 'ostrich-gr4j-salmon', 'ostrich-ackley10', 'ostrich-ackley20' ]  # in folder names
    npara             = [              4,                9,                10,                20  ]
    budgets           = [ [100, 200, 400], [ 225, 450, 900 ], [250, 500, 1000], [500, 1000, 2000] ]

    # names of metrics
    metric_names_str   = [ 'RMSE', 'KGE$_\mathrm{Q}$', '$f_{Ackley}^{(10)}(x)$', '$f_{Ackley}^{(20)}(x)$' ]  # for plots
    metric_names       = [ 'rmse', 'kge', 'fx', 'fx' ]               # in folder names
    metric_names_raven = [ 'rmse', 'DIAG_KLING_GUPTA', 'fx', 'fx' ]  # in Raven diagnostics

    # names of algorithms
    algorithm_names_str = [ 'DDS', 'SCE', 'PSO' ]  # for plots
    algorithm_names     = [ 'dds', 'sce', 'pso' ]  # in folder names


    if not(os.path.exists(jsonfile)):

        dict_results = {}

        # read calibration results per algorithm
        for ialgo in algorithm_names:

            # read results for each example
            tmp_example = {}
            for iiexample,iexample in enumerate(example_names):

                # read calibration results per budget
                tmp_budget = {}
                for ibudget in budgets[iiexample]:

                    # read calibration results per metric
                    tmp_metric = {}
                    for imetric in metric_names[iiexample:iiexample+1]:

                        para = {}
                        obfv = {}
                        hist = {}
                        Qsim = {}
                        for itrial in range(ntrials):

                            # get para values over course of calibration
                            f = open("example_9/"+iexample+"_"+ialgo+"_"+imetric+"_budget-"+str(ibudget)+"/trial_"+str(itrial+1)+"/OstModel0.txt", "r")
                            content = f.readlines()
                            content = [ cc.strip() for cc in content if not(cc.startswith('Run')) ]
                            content = [ cc for cc in content if not(cc.strip() == '') ]
                            f.close()

                            history = {}

                            history['iter'] = [ int(cc.split()[0]) for cc in content ]

                            # record only best paraset found so far (less noisy)
                            oobfv = [ float(content[0].split()[1]) ]
                            ppara = [ [ float(icc) for icc in content[0].split()[2:] ] ]
                            for cc in content[1:]:
                                ioobfv = float( cc.split()[1] )
                                if ioobfv < oobfv[-1]:
                                    ippara = [ float(icc) for icc in cc.split()[2:] ]
                                else:
                                    ippara = ppara[-1]
                                    ioobfv = oobfv[-1]
                                oobfv.append( ioobfv )
                                ppara.append( ippara )

                            history['iter'] = [ int(cc.split()[0]) for cc in content ]
                            if iexample == 'ostrich-gr4j-salmon':
                                history['obfv'] = np.array(oobfv) * -1.0   # was maximized; ostrich multipied and reports vals with -1.0
                            else:
                                history['obfv'] = np.array(oobfv) *  1.0   # was minimized
                            history['para'] = np.array(ppara)
                            hist['trial_'+str(itrial+1)] = history

                            # get calibrated values (last parameter set in history)
                            para['trial_'+str(itrial+1)] = history['para'][-1]

                            # get objective function values (last objective function value in history)
                            obfv['trial_'+str(itrial+1)] = history['obfv'][-1]

                        tmp_metric[imetric] = { 'para': para, 'obfv': obfv, 'hist': hist }

                    tmp_budget[str(ibudget)] = tmp_metric

                tmp_example[iexample] = tmp_budget

            dict_results[ialgo] = tmp_example

        # save results to file such that it can be used again later
        # create json object from dictionary
        json_dict = json.dumps(dict_results,cls=NumpyArrayEncoder)

        # open file for writing, "w"
        ff = open(jsonfile,"w")

        # write json object to file
        ff.write(json_dict)

        # close file
        ff.close()


    # read from json file
    with open(jsonfile) as ff:
        dict_results = json.load(ff)


    # -------------------------------------------------------------------------
    # Plotting of results
    # -------------------------------------------------------------------------
    # Main plot
    ncol        = 3           # number columns
    nrow        = 4           # number of rows
    textsize    = 10          # standard text size
    dxabc       = 0.95          # % of (max-min) shift to the right from left y-axis for a,b,c,... labels
    dyabc       = 0.92          # % of (max-min) shift up from lower x-axis for a,b,c,... labels
    dxsig       = 1.23        # % of (max-min) shift to the right from left y-axis for signature
    dysig       = -0.075      # % of (max-min) shift up from lower x-axis for signature
    dxtit       = 0           # % of (max-min) shift to the right from left y-axis for title
    dytit       = 1.2         # % of (max-min) shift up from lower x-axis for title
    hspace      = 0.07        # x-space between subplots
    vspace      = 0.05        # y-space between subplots

    lwidth      = 1.0         # linewidth
    elwidth     = 0.5         # errorbar line width
    alwidth     = 1.0         # axis line width
    glwidth     = 0.5         # grid line width
    msize       = 3.0         # marker size
    mwidth      = 0.0         # marker edge width
    mcol1       = '0.7'       # primary marker colour
    mcol2       = '0.0'       # secondary
    mcol3       = '0.0'       # third
    mcols       = color.colours(['blue','green','yellow','orange','red','darkgray','darkblue','black','darkgreen','gray'])
    lcol0       = color.colours('black')    # line colour
    lcol1       = color.colours('blue')     # line colour
    lcol2       = color.colours('red')    # line colour
    lcol3       = color.colours('darkgreen')   # line colour
    lcols       = color.colours(['blue','lightblue','darkblue'])
    markers     = ['o','v','s','^']

    # Legend
    llxbbox     = 0.5        # x-anchor legend bounding box
    llybbox     = -0.6        # y-anchor legend bounding box
    llrspace    = 0.          # spacing between rows in legend
    llcspace    = 1.0         # spacing between columns in legend
    llhtextpad  = 0.4         # the pad between the legend handle and text
    llhlength   = 1.5         # the length of the legend handles
    frameon     = False       # if True, draw a frame around the legend. If None, use rc

    import matplotlib as mpl
    import matplotlib.patches as patches
    from matplotlib.lines import Line2D
    import matplotlib.colors as mcolors
    import matplotlib.dates as dates
    from matplotlib.ticker import FormatStrFormatter
    mpl.use('TkAgg')

    if (outtype == 'pdf'):
        mpl.use('PDF') # set directly after import matplotlib
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_pdf import PdfPages
        # Customize: http://matplotlib.sourceforge.net/users/customizing.html
        mpl.rc('ps', papersize='a4', usedistiller='xpdf') # ps2pdf
        # mpl.rc('figure', figsize=(8.27,11.69)) # a4 portrait
        mpl.rc('figure', figsize=(7.48,9.06)) # WRR maximal figure size
        if usetex:
            mpl.rc('text', usetex=True)
            if not serif:
                #   r'\usepackage{helvet}',                             # use Helvetica
                mpl.rcParams['text.latex.preamble'] = [
                    r'\usepackage[math,lf,mathtabular,footnotefigures]{MyriadPro}', # use MyriadPro font
                    r'\renewcommand{\familydefault}{\sfdefault}',       # normal text font is sans serif
                    r'\figureversion{lining,tabular}',
                    r'\usepackage{wasysym}',                            # for permil symbol (load after MyriadPro)
                    ]
            else:
                mpl.rcParams['text.latex.preamble'] = [
                    r'\usepackage{wasysym}'                     # for permil symbol
                    ]
        else:
            if serif:
                mpl.rcParams['font.family']     = 'serif'
                mpl.rcParams['font.sans-serif'] = 'Times'
            else:
                mpl.rcParams['font.family']     = 'sans-serif'
                mpl.rcParams['font.sans-serif'] = 'Arial'       # Arial, Verdana
    elif (outtype == 'png') or (outtype == 'html') or (outtype == 'd3'):
        mpl.use('Agg') # set directly after import matplotlib
        import matplotlib.pyplot as plt
        # mpl.rc('figure', figsize=(8.27,11.69)) # a4 portrait
        mpl.rc('figure', figsize=(7.48,9.06)) # WRR maximal figure size
        if usetex:
            mpl.rc('text', usetex=True)
            if not serif:
                #   r'\usepackage{helvet}',                             # use Helvetica
                mpl.rcParams['text.latex.preamble'] = [
                    r'\usepackage[math,lf,mathtabular,footnotefigures]{MyriadPro}', # use MyriadPro font
                    r'\renewcommand{\familydefault}{\sfdefault}',       # normal text font is sans serif
                    r'\figureversion{lining,tabular}',
                    r'\usepackage{wasysym}',                            # for permil symbol (load after MyriadPro)
                    ]
            else:
                mpl.rcParams['text.latex.preamble'] = [
                    r'\usepackage{wasysym}'                     # for permil symbol
                    ]
        else:
            if serif:
                mpl.rcParams['font.family']     = 'serif'
                mpl.rcParams['font.sans-serif'] = 'Times'
            else:
                mpl.rcParams['font.family']     = 'sans-serif'
                mpl.rcParams['font.sans-serif'] = 'Arial'       # Arial, Verdana
        mpl.rc('savefig', dpi=dpi, format='png')
    else:
        import matplotlib.pyplot as plt
        # mpl.rc('figure', figsize=(4./5.*8.27,4./5.*11.69)) # a4 portrait
        mpl.rc('figure', figsize=(7.48,9.06)) # WRR maximal figure size
    mpl.rc('text.latex') #, unicode=True)
    mpl.rc('font', size=textsize)
    mpl.rc('path', simplify=False) # do not remove
    # print(mpl.rcParams)
    mpl.rc('axes', linewidth=alwidth, edgecolor=fgcolor, facecolor=bgcolor, labelcolor=fgcolor)
    mpl.rc('figure', edgecolor=bgcolor, facecolor='grey')
    mpl.rc('grid', color=fgcolor)
    mpl.rc('lines', linewidth=lwidth, color=fgcolor)
    mpl.rc('patch', edgecolor=fgcolor)
    mpl.rc('savefig', edgecolor=bgcolor, facecolor=bgcolor)
    mpl.rc('patch', edgecolor=fgcolor)
    mpl.rc('text', color=fgcolor)
    mpl.rc('xtick', color=fgcolor)
    mpl.rc('ytick', color=fgcolor)

    if (outtype == 'pdf'):
        pdffile = plotname+'.pdf'
        print('Plot PDF ', pdffile)
        pdf_pages = PdfPages(pdffile)
    elif (outtype == 'png'):
        print('Plot PNG ', plotname)
    else:
        print('Plot X')

    t1  = time.time()
    ifig = 0

    figsize = mpl.rcParams['figure.figsize']
    mpl.rcParams['axes.linewidth'] = lwidth


    # green-pink colors
    # cc = color.get_brewer('piyg10', rgb=True)
    # low_cc = tuple([1.0,1.0,1.0])
    # del cc[0]  # drop darkest two pink color
    # del cc[0]  # drop darkest two pink color

    # blue-white colors
    cc = color.get_brewer('blues7', rgb=True)
    low_cc = tuple([1.0,1.0,1.0])
    #del cc[0]  # drop darkest two pink color
    #del cc[0]  # drop darkest two pink color
    cc = list([low_cc])+cc      # prepend "white"
    cc = [ icc+(1.0,) if iicc > 0 else icc+(0.0,) for iicc,icc in enumerate(cc) ]  # add alpha
    cmap = mpl.colors.ListedColormap(cc)

    # gray-white colors
    cc_gray = color.get_brewer('greys7',rgb=True)
    low_cc_gray = tuple([1.0,1.0,1.0])
    #del cc[0]  # drop darkest two pink color
    #del cc[0]  # drop darkest two pink color
    cc_gray = list([low_cc_gray])+cc_gray      # prepend "white"
    cc_gray = [ icc+(1.0,) if iicc > 0 else icc+(0.0,) for iicc,icc in enumerate(cc_gray) ]  # add alpha
    cmap_gray = mpl.colors.ListedColormap(cc_gray)

    inorm = 'linear'

    if inorm == 'log':
        min_sti = 0.01
        max_sti = 1.0
        norm = mcolors.LogNorm(min_sti,max_sti)
    elif inorm == 'pow':
        # n=2 --> samples=200/trial
        pow_lambda1 = 0.2
        max_pow1    = nsets*1./20.
        norm1 = mcolors.PowerNorm(pow_lambda1,vmin=0,vmax=max_pow1)
        # n=10 --> samples=1000/trial
        pow_lambda2 = 0.2
        max_pow2    = 1000.
        norm2 = mcolors.PowerNorm(pow_lambda2,vmin=0,vmax=max_pow2)
    elif inorm == 'linear':
        norm = None
        norm1 = None
    else:
        raise ValueError('Norm for colormap not known.')


    ifig = 0

    ifig += 1
    iplot = 0
    print('Plot - Fig ', ifig)
    fig = plt.figure(ifig)

    # -----------------------
    # plot
    # -----------------------
    #
    xpoints = 1 # plots only every nth point

    # -------------------------
    # plot - hydrographs for each algorithm used for calibration
    # -------------------------
    for iiexample,iexample in enumerate(example_names):
        #print("")
        #print("Example: ",iexample)

        imetric = metric_names[iiexample]

        for iialgo,ialgo in enumerate(algorithm_names):

            iplot += 1
            sub = fig.add_axes(position(nrow, ncol, iplot, hspace=hspace, vspace=vspace))

            # label indicating algorithm (top)
            if iiexample == 0:
                sub.text(0.5,1.07, str2tex('Calibration using '+algorithm_names_str[iialgo], usetex=usetex),
                         fontsize=textsize+2,
                         transform=sub.transAxes, horizontalalignment='center', verticalalignment='bottom')

            # label indicating the example name (right)
            if iplot%ncol == 0:  # last col
                sub.text(1.3,0.5, str2tex(example_names_str[iiexample], usetex=usetex),
                         fontsize=textsize+2, rotation=90,
                         transform=sub.transAxes, horizontalalignment='center', verticalalignment='center')

            for iibudget,ibudget in enumerate(budgets[iiexample]):

                # which one is best trial
                all_final_obfv = [ dict_results[ialgo][iexample][str(ibudget)][imetric]['hist']['trial_'+str(itrial+1)]['obfv'][-1] for itrial in range(ntrials) ]

                if imetric in ['rmse','fx']:
                    idx_best_trial = np.argmin(all_final_obfv)
                    best_obfv = np.min(all_final_obfv)
                    #print("algorithm: {} with budget: {} --> trials: {} (best: {})".format(ialgo,ibudget,all_final_obfv,best_obfv))
                elif imetric in ['pbias']:
                    idx_best_trial = np.argmin(np.abs(all_final_obfv))
                    best_obfv = np.min(np.abs(all_final_obfv))
                    #print("algorithm: {} with budget: {} --> trials: {} (best: {})".format(ialgo,ibudget,all_final_obfv,best_obfv))
                elif imetric in ['nse', 'lnse', 'kge', 'r2', ]:
                    idx_best_trial = np.argmax(all_final_obfv)
                    best_obfv = np.max(all_final_obfv)
                    #print("algorithm: {} with budget: {} --> trials: {} (best: {})".format(ialgo,ibudget,all_final_obfv,best_obfv))
                else:
                    raise ValueError('Metric {} not known'.formt(imetric))

                for itrial in range(ntrials):

                    xvals = dict_results[ialgo][iexample][str(ibudget)][imetric]['hist']['trial_'+str(itrial+1)]['iter']
                    yvals = dict_results[ialgo][iexample][str(ibudget)][imetric]['hist']['trial_'+str(itrial+1)]['obfv']

                    if itrial == 0:
                        label = str2tex('budget = '+str(ibudget)+" = $"+str(int(ibudget/npara[iiexample]))+" n_p$", usetex=usetex)
                        label = str2tex('budget = '+str(ibudget), usetex=usetex)
                    else:
                        label = ''
                    if ibudget == budgets[iiexample][-1]:    # largest budget
                        color = '0.7'
                        alpha = 0.7
                        zorder = 10
                    elif ibudget == budgets[iiexample][-2]:  # medium budget
                        color = lcol1
                        alpha = 0.7
                        zorder = 20
                    elif ibudget == budgets[iiexample][-3]:  # lowest budget
                        color = lcol2
                        alpha = 0.7
                        zorder = 30
                    else:
                        color = 'red'
                        alpha = 0.3
                    sub.plot( xvals, yvals,
                                  color=color,
                                  alpha=alpha,
                                  label=label,
                                  zorder=zorder)

                # add best obfv to plot as label
                if imetric in ['rmse','fx','pbias']:
                    valign = 'top'
                    delta = -0.05
                else:
                    valign = 'bottom'
                    delta = 0.0
                sub.text( 0.98*ibudget, best_obfv+delta, str2tex(astr(best_obfv,prec=2), usetex=usetex),
                            #transform=sub.transAxes,
                            fontsize=textsize-2,
                            color=color,
                            alpha=alpha,
                            horizontalalignment='right',
                            verticalalignment=valign)



            # mark best to achieve objective function value
            extend_x_range = 1.07   # factor to make extend x-axis to the right (to better see that trials actually stopped at budget)
            if iexample == 'ostrich-gr4j-salmon':
                sub.plot( [0,np.max(budgets[iiexample])*extend_x_range], [1.0,1.0], linestyle='--', linewidth=lwidth/2, color='0.7', alpha=0.7)
            else:
                sub.plot( [0,np.max(budgets[iiexample])*extend_x_range], [0.0,0.0], linestyle='--', linewidth=lwidth/2, color='0.7', alpha=0.7)
            if iplot%ncol == 2:
                if iexample == 'ostrich-gr4j-salmon':
                    sub.text( 0.02*np.max(budgets[iiexample])*extend_x_range, 1.02, str2tex('Global maximum (theoretical)', usetex=usetex),
                        #transform=sub.transAxes,
                        fontsize=textsize-2,
                        color='0.7',
                        horizontalalignment='left', verticalalignment='bottom')
                elif iexample == 'ostrich-dispersivity':
                    sub.text( 0.02*np.max(budgets[iiexample])*extend_x_range, -0.1, str2tex('Global minimum (theoretical)', usetex=usetex),
                        #transform=sub.transAxes,
                        fontsize=textsize-2,
                        color='0.7',
                        horizontalalignment='left', verticalalignment='top')
                elif iexample == 'ostrich-ackley10' or iexample == 'ostrich-ackley20':
                    sub.text( 0.02*np.max(budgets[iiexample])*extend_x_range, -0.1, str2tex('Global minimum', usetex=usetex),
                        #transform=sub.transAxes,
                        fontsize=textsize-2,
                        color='0.7',
                        horizontalalignment='left', verticalalignment='top')
                else:
                    raise ValueError("Example {} not implemented.".format(iexample))

            # axis labels
            if (iplot-1)//ncol == len(example_names)-1:   # last row
                sub.set_xlabel(str2tex('Iteration', usetex=usetex))
            if iplot%ncol == 1:
                sub.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
                sub.set_ylabel(str2tex('Objective Function Value\n'+metric_names_str[iiexample], usetex=usetex))
            else:
                sub.set_yticklabels([])

            # limits
            sub.set_xlim([ 0.0,np.max(budgets[iiexample])*extend_x_range])
            if iexample == 'ostrich-gr4j-salmon':
                sub.set_ylim([-0.3,1.2])
            elif iexample == 'ostrich-dispersivity':
                sub.set_ylim([-0.5,3.5])
            elif iexample == 'ostrich-ackley10' or iexample == 'ostrich-ackley20':
                sub.set_ylim([-0.7,5.0])
            else:
                raise ValueError("Example {} not implemented.".format(iexample))

            # ticks on x-axis are budget values
            sub.set_xticks(budgets[iiexample])


            # legend
            if iplot%ncol == 2:
                if iexample == 'ostrich-gr4j-salmon':
                    sub.legend(frameon=frameon, ncol=1,
                                    labelspacing=llrspace, handletextpad=llhtextpad, handlelength=llhlength,
                                    loc='lower right', bbox_to_anchor=(1.0,0.0), scatterpoints=1, numpoints=1,
                                    fontsize = textsize-2)
                elif iexample == 'ostrich-dispersivity':
                    sub.legend(frameon=frameon, ncol=1,
                                    labelspacing=llrspace, handletextpad=llhtextpad, handlelength=llhlength,
                                    loc='upper right', bbox_to_anchor=(1.0,1.0), scatterpoints=1, numpoints=1,
                                    fontsize = textsize-2)
                elif iexample == 'ostrich-ackley10' or iexample == 'ostrich-ackley20':
                    sub.legend(frameon=frameon, ncol=1,
                                    labelspacing=llrspace, handletextpad=llhtextpad, handlelength=llhlength,
                                    loc='upper right', bbox_to_anchor=(1.0,1.0), scatterpoints=1, numpoints=1,
                                    fontsize = textsize-2)
                else:
                    raise ValueError("Example {} not implemented.".format(iexample))

            # abc
            sub.text( 1.05, 1.0, str2tex(chr(96+iplot),usetex=usetex),
                                        ha = 'left', va = 'top',
                                        fontweight='bold',
                                        transform=sub.transAxes,
                                        fontsize=textsize+3 )



    if (outtype == 'pdf'):
        pdf_pages.savefig(fig)
        plt.close(fig)
    elif (outtype == 'png'):
        pngfile = pngbase+"{0:04d}".format(ifig)+".png"
        fig.savefig(pngfile, transparent=transparent, bbox_inches=bbox_inches, pad_inches=pad_inches)
        plt.close(fig)



    # --------------------------------------
    # Finish
    # --------------------------------------
    if (outtype == 'pdf'):
        pdf_pages.close()
    elif (outtype == 'png'):
        pass
    else:
        plt.show()


    t2  = time.time()
    istr = '  Time plot [m]: '+astr((t2-t1)/60.,1) if (t2-t1)>60. else '  Time plot [s]: '+astr(t2-t1,0)
    print(istr)
