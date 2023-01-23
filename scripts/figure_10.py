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
#     run figure_10.py -t pdf -p figure_10

#!/usr/bin/env python
from __future__ import print_function

"""

Plots results of hydrologic calibration experiments using SO and MO algorithms

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
                                      description='''Plots results of hydrologic calibration experiments using SO and MO algorithms.''')
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

    jsonfile = dir_path+'/figure_10.json'

    # -------------------------------------------------------------------------
    # general
    # -------------------------------------------------------------------------

    ntrials = 10

    # name of examples
    example_names_str = [ [ 'KGE$_{logQ_{low}}$ and KGE$_{logQ_{high}}$', 'KGE$_{logQ_{low}}$', 'KGE$_{logQ_{high}}$', '0.5 (KGE$_{logQ_{low}} + KGE$_{logQ_{high}}$)'],
                          #[ 'KGE$_\alpha$ and KGE$_\beta$', 'KGE$_{\alpha}$', 'KGE$_{\beta}$'],
                          #[ 'KGE$_\alpha$ and KGE$_r$',     'KGE$_{\alpha}$', 'KGE$_{r}$'],
                          [ 'KGE$_{b}$ and KGE$_r$',      'KGE$_{b}$',  'KGE$_{r}$', r'0.5 (KGE$_{\beta}+KGE$_{r}$)'],
                          [ 'KGE$_{b}$ and KGE$_r$',      'KGE$_{b}$',  'KGE$_{r}$', r'0.5 (KGE$_{\beta}+KGE$_{r}$)']
                        ]  # for plots
    example_names     = [ [ 'ostrich-gr4j-salmon', 'ostrich-gr4j-salmon', 'ostrich-gr4j-salmon', 'ostrich-gr4j-salmon'],
                          #[ 'ostrich-gr4j-salmon', 'ostrich-gr4j-salmon', 'ostrich-gr4j-salmon'],
                          #[ 'ostrich-gr4j-salmon', 'ostrich-gr4j-salmon', 'ostrich-gr4j-salmon'],
                          [ 'ostrich-gr4j-salmon', 'ostrich-gr4j-salmon', 'ostrich-gr4j-salmon', 'ostrich-gr4j-salmon'],
                          [ 'ostrich-gr4j-salmon', 'ostrich-gr4j-salmon', 'ostrich-gr4j-salmon', 'ostrich-gr4j-salmon']
                        ]  # in folder names
    npara             = [  9,
                           #9,
                           #9,
                           9,
                           9  ]
    budgets           = [ [2700,900,900,900],
                          #[2700,900,900],
                          #[2700,900,900],
                          [2700,900,900,900],
                          [2700,900,900,900] ]

    # names of metrics
    metric_names_str   = [ [ ['KGE$_{logQ_{low}}$','KGE$_{logQ_{high}}$'], 'KGE$_{logQ_{low}}$', 'KGE$_{logQ_{high}}$', '0.5 (KGE$_{logQ_{low}} + KGE$_{logQ_{high}}$)'],
                           #[ ['KGE$_{\u03B1}$','KGE$_{\u03B2}$'],         'KGE$_{\u03B1}$',     'KGE$_{\u03B2}$'],
                           #[ ['KGE$_{\u03B1}$','KGE$_r$'],                'KGE$_{\u03B1}$',     'KGE$_{r}$'],
                           [ [r'KGE$_{\beta}$','KGE$_r$'],        r'KGE$_{\beta}$',     'KGE$_{r}$', r'0.5 (KGE$_{\beta}+KGE$_{r}$)'],
                           [ [r'KGE$_{\beta}$','KGE$_r$'],        r'KGE$_{\beta}$',     'KGE$_{r}$', r'0.5 (KGE$_{\beta}+KGE$_{r}$)']
                         ]  # for plots
    metric_names       = [ [ ['lkgelow05','lkgehgh95'],                    'lkgelow05',          'lkgehgh95', 'mean_lkgelow05_lkgehgh95'],
                           #[ ['kgea','kgeb'],                              'kgea',               'kgeb'],
                           #[ ['kgea','kger'],                              'kgea',               'kger'],
                           [ ['kgeb','kger'],                              'kgeb',               'kger', 'mean_kgeb_kger'],
                           [ ['kgeb','kger'],                              'kgeb',               'kger', 'mean_kgeb_kger']
                         ] # in folder names
    metric_names_raven = [ [ ['lkgeqlow05', 'lkgeqhgh95'], 'lkgeqlow05', 'lkgeqhgh95', ['lkgeqlow05', 'lkgeqhgh95']  ],
                           #[ ['kge_a', 'kge_b'],           'kge_a',      'kge_b' ],
                           #[ ['kge_a', 'kge_r'],           'kge_a',      'kge_r' ],
                           [ ['kge_b', 'kge_r'],           'kge_b',      'kge_r', ['kge_b', 'kge_r'] ],
                           [ ['kge_b', 'kge_r'],           'kge_b',      'kge_r', ['kge_b', 'kge_r'] ]
                         ] # in (Raven) diagnostics_extended

    # names of algorithms
    algorithm_names_str = [ [ 'PADDS', 'DDS', 'DDS' , 'DDS' ],
                            #[ 'PADDS', 'DDS', 'DDS' ],
                            #[ 'PADDS', 'DDS', 'DDS' ],
                            [ 'PADDS', 'DDS', 'DDS', 'DDS'  ],
                            [ 'PADDS', 'DDS', 'DDS', 'DDS'  ]    ]  # for plots
    algorithm_names     = [ [ 'padds', 'dds', 'dds', 'dds' ],
                            #[ 'padds', 'dds', 'dds' ],
                            #[ 'padds', 'dds', 'dds' ],
                            [ 'padds', 'dds', 'dds', 'dds' ],
                            [ 'padds', 'dds', 'dds', 'dds' ]    ]  # in folder names




    if not(os.path.exists(jsonfile)):   # True: #

        print(" ")
        print(">>>>>>>>> READING ORIGINAL DATA AGAIN INTO JSON")
        print(" ")

        dict_results = {}

        # read calibration results per algorithm
        for iiexample_group,iexample_group in enumerate(example_names):

            tmp_example = {}
            for iiexample,iexample in enumerate(iexample_group):

                ialgo = algorithm_names[iiexample_group][iiexample]
                imetric = metric_names[iiexample_group][iiexample]

                # print('imetric = ',imetric)

                if ialgo == 'dds':

                    para = {}
                    obfv = {}
                    hist = {}    # is correct but just not saved because not needed
                    for itrial in range(ntrials):

                        # get para values over course of calibration
                        filename = "example_10/"+iexample+"_"+ialgo+"_"+imetric+"/trial_"+str(itrial+1)+"/OstModel0.txt"
                        f = open(filename, "r")
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
                        history['obfv'] = np.array(oobfv) * -1.0   # was maximized; ostrich multipied and reports vals with -1.0
                        history['para'] = np.array(ppara)
                        hist['trial_'+str(itrial+1)] = history

                        # get calibrated values (last parameter set in history)
                        para['trial_'+str(itrial+1)] = history['para'][-1]

                        # get objective function values (last objective function value in history)
                        obfv['trial_'+str(itrial+1)] = history['obfv'][-1]

                        # this must be the run saved in "best" --> check and then get the other metric as well (for plotting)
                        filename_diag = "example_10/"+iexample+"_"+ialgo+"_"+imetric+"/trial_"+str(itrial+1)+"/best/Diagnostics_extended.csv"
                        f = open(filename_diag, "r")
                        content = f.readlines()
                        f.close()
                        header = [ cc.strip() for cc in content[0].strip().split(',') ]
                        vals   = [ float(cc.strip()) for cc in content[1].strip().split(',') ]

                        imetric_diag = metric_names_raven[iiexample_group][iiexample]
                        if ( type(imetric_diag) == list ) :

                            #print('list of objectives: {}. means they need to get merged somehow. taking MEAN({}). implement something else if needed.'.format(imetric_diag,imetric_diag))
                            idx_diag = [ header.index(iimetric_diag) for iimetric_diag in imetric_diag ]
                            if np.abs( np.mean(np.array(vals)[idx_diag]) - obfv['trial_'+str(itrial+1)] ) > 0.000001:  # here's where "mean" is assumed; will fail if "mean" was not actually calibrated
                                raise ValueError("Best function value in history (val: {}, file: {})\n does not seem to be the one saved as 'best' (val: {}, file: {}).".format(filename,obfv['trial_'+str(itrial+1)],filename_diag,vals[idx_diag]))
                            else:
                                # save now all metrics instead of only the single objective calibrated
                                metrics_mo = metric_names_raven[iiexample_group][0]
                                idx_diag = [  header.index(ii) for ii in metrics_mo ]
                                tmp_diag = [ vals[ii] for ii in idx_diag ]
                                obfv['trial_'+str(itrial+1)] = tmp_diag

                        else:
                            idx_diag = header.index(imetric_diag)

                            if np.abs( vals[idx_diag] - obfv['trial_'+str(itrial+1)] ) > 0.000001:
                                raise ValueError("Best function value in history (val: {}, file: {})\n does not seem to be the one saved as 'best' (val: {}, file: {}).".format(filename,obfv['trial_'+str(itrial+1)],filename_diag,vals[idx_diag]))
                            else:
                                # save now all metrics instead of only the single objective calibrated
                                metrics_mo = metric_names_raven[iiexample_group][0]
                                idx_diag = [  header.index(ii) for ii in metrics_mo ]
                                tmp_diag = [ vals[ii] for ii in idx_diag ]
                                obfv['trial_'+str(itrial+1)] = tmp_diag

                        tmp_example[ialgo+"_"+imetric] = { 'para': para, 'obfv': obfv } #, 'hist': hist }

                elif ialgo == 'padds':

                    para = {}
                    obfv = {}
                    for itrial in range(ntrials):

                        # get para values and obfv or non-dominated solutions
                        # "OstModel0.txt" does not contain OBFV's
                        f = open("example_10/"+iexample+"_"+ialgo+"_"+'_'.join(imetric)+"/trial_"+str(itrial+1)+"/OstNonDomSolutions0.txt", "r")
                        content = f.readlines()
                        content = [ cc.strip() for cc in content if not(cc.startswith('Ostrich') or cc.startswith('gen'))  ]
                        content = [ cc for cc in content if not(cc.strip() == '') ]
                        f.close()

                        # record only best paraset found so far (less noisy)
                        oobfv = [ [ float(icc) for icc in cc.split()[1:1+len(imetric)] ] for cc in content ]
                        ppara = [ [ float(icc) for icc in cc.split()[1+len(imetric):1+len(imetric)+npara[iiexample_group]] ] for cc in content ]

                        oobfv = np.array(oobfv) * -1.0
                        ppara = np.array(ppara)

                        # some sorting (along first objective) to make plotting easier
                        col = 0
                        idx = np.argsort(oobfv[:,col])
                        oobfv = oobfv[idx]
                        ppara = ppara[idx]

                        obfv['trial_'+str(itrial+1)] = oobfv
                        para['trial_'+str(itrial+1)] = ppara

                    tmp_example[ialgo+"_"+'_'.join(imetric)] = { 'para': para, 'obfv': obfv } #, 'hist': hist }

                else:
                     raise ValueError("Don't know what to do with algorithm {}.".format(algorithm_names[iiexample_group][iiexample]))

            dict_results['experiment_'+str(iiexample_group+1)] = tmp_example

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
    markers     = ['s','<','v','s','^']

    # Legend
    llxbbox     = 0.5         # x-anchor legend bounding box
    llybbox     = -0.6        # y-anchor legend bounding box
    llrspace    = 0.4         # spacing between rows in legend
    llcspace    = 1.0         # spacing between columns in legend
    llhtextpad  = 0.4         # the pad between the legend handle and text
    llhlength   = 1.0         # the length of the legend handles
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
        mpl.rc('figure', figsize=(10.97,11.69)) # a4 portrait
        if usetex:
            mpl.rc('text', usetex=True)
        else:
            #mpl.rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
            mpl.rc('font',**{'family':'serif','serif':['times']})
        mpl.rc('text.latex') #, unicode=True)
    elif (outtype == 'png'):
        mpl.use('Agg') # set directly after import matplotlib
        import matplotlib.pyplot as plt
        mpl.rc('figure', figsize=(10.97,11.69)) # a4 portrait
        if usetex:
            mpl.rc('text', usetex=True)
        else:
            #mpl.rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
            mpl.rc('font',**{'family':'serif','serif':['times']})
        mpl.rc('text.latex') #, unicode=True)
        mpl.rc('savefig', dpi=dpi, format='png')
    else:
        import matplotlib.pyplot as plt
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

    # -------------------------
    # plot - hydrographs for each algorithm used for calibration
    # -------------------------
    for iiexample_group,iexample_group in enumerate(example_names):
        #print("")
        #print("Example group: ",iiexample_group)

        iplot += 1
        sub = fig.add_axes(position(nrow, ncol, iplot, hspace=hspace, vspace=vspace)+[0.05*(iplot-1)-0.035,0.0,0.0,0.0])


        for iiexample in [1,2,3,0]: #order such that Pareto is plotted last such that plotrange is maximal and allows to properly extend front until plot range

            ialgo = algorithm_names[iiexample_group][iiexample]
            imetric = metric_names[iiexample_group][iiexample]
            imetric_str = metric_names_str[iiexample_group][iiexample]
            imetric_rvn = metric_names_raven[iiexample_group][iiexample]

            if ialgo == 'dds':

                # just plot some points
                xvals = [ dict_results['experiment_'+str(iiexample_group+1)][ialgo+"_"+imetric]['obfv']['trial_'+str(itrial+1)][0] for itrial in range(ntrials) ]
                yvals = [ dict_results['experiment_'+str(iiexample_group+1)][ialgo+"_"+imetric]['obfv']['trial_'+str(itrial+1)][1] for itrial in range(ntrials) ]

                if ( type(imetric_rvn) == list ) :
                    label = str2tex('all calibration trials',usetex=usetex)  # LAZY  # SO: Calibrating 0.5$\cdot$(Obj. #1 + Obj. #2)\n
                else:
                    label = str2tex('all calibration trials',usetex=usetex)   # SO: Calibrating Obj. #'+str(iiexample)+'\n

                sub.plot( xvals, yvals,
                              linewidth=0.0, marker=markers[iiexample],
                              markersize=msize, markeredgewidth=msize/3,markerfacecolor='none',
                              color='0.7',
                              alpha=0.7,
                              label=label,
                              zorder=10)

                # find best and plot in blue
                if ( type(imetric_rvn) == list ) :
                    all_final_obfv = [ dict_results['experiment_'+str(iiexample_group+1)][ialgo+"_"+imetric]['obfv']['trial_'+str(itrial+1)] for itrial in range(ntrials) ]
                    all_final_obfv = [ np.mean(all_final_obfv[itrial]) for itrial in range(ntrials) ] # "mean" assumed here
                    idx_best_trial = np.argmax(all_final_obfv)
                    label = str2tex('best calibration trial',usetex=usetex)   # LAZY   # SO: Calibrating 0.5$\cdot$(Obj. #1 + Obj. #2)\n
                else:
                    idx_diag     = metric_names[iiexample_group][0].index(imetric)
                    # print("iexample = '",imetric,"'  find in ",metric_names[iiexample_group][0],"   --> idx = ",idx_diag)
                    all_final_obfv = [ dict_results['experiment_'+str(iiexample_group+1)][ialgo+"_"+imetric]['obfv']['trial_'+str(itrial+1)][idx_diag] for itrial in range(ntrials) ]
                    idx_best_trial = np.argmax(all_final_obfv)
                    label = str2tex('best calibration trial',usetex=usetex)   # SO: Calibrating Obj. #'+str(iiexample)+'\n

                sub.plot( xvals[idx_best_trial], yvals[idx_best_trial],
                              linewidth=0.0, marker=markers[iiexample],
                              markersize=msize, markeredgewidth=msize/3,markerfacecolor='none',
                              color=lcol1,
                              alpha=0.7,
                              label=label,
                              zorder=40)

            elif ialgo == 'padds':

                # plot pareto fronts
                for itrial in range(ntrials):
                    xvals = np.array(dict_results['experiment_'+str(iiexample_group+1)][ialgo+"_"+"_".join(imetric)]['obfv']['trial_'+str(itrial+1)])[:,0]
                    yvals = np.array(dict_results['experiment_'+str(iiexample_group+1)][ialgo+"_"+"_".join(imetric)]['obfv']['trial_'+str(itrial+1)])[:,1]

                    if itrial == 0:
                        label = str2tex('all calibration trials',usetex=usetex)  # MO: Calibrating Obj. #1 and Obj. #2\n
                    else:
                        label = ''
                    sub.plot( xvals, yvals,
                              linewidth=lwidth,
                              color='0.85',
                              alpha=0.7,
                              label=label,
                              zorder=10)

                # merge all pareto fronts and plot in blue
                all_final_obfv = [ dict_results['experiment_'+str(iiexample_group+1)][ialgo+"_"+"_".join(imetric)]['obfv']['trial_'+str(itrial+1)] for itrial in range(ntrials) ]
                all_final_obfv = [ x for xs in all_final_obfv for x in xs ]

                pareto_final    = []
                c_non_dominated = 0
                c_dominated     = 0
                for iisolution,isolution in enumerate(all_final_obfv):
                    non_dominated = True

                    for iisolutioncomp,isolutioncomp in enumerate(all_final_obfv):
                        if iisolution != iisolutioncomp:
                            if np.all(np.array(isolutioncomp) > np.array(isolution)): # expects maximization of all objectives
                                non_dominated = False

                    if non_dominated:
                        c_non_dominated += 1
                        pareto_final.append(isolution)
                    else:
                        c_dominated += 1
                pareto_final = np.array(pareto_final)

                # just sort again
                col = 0
                idx = np.argsort(pareto_final[:,col])
                pareto_final = pareto_final[idx]

                #print("Number of non-dominated solutions in merged Pareto fronts: ",c_non_dominated)
                #print("Number of     dominated solutions in merged Pareto fronts: ",c_dominated)

                xlim=sub.get_xlim()
                ylim=sub.get_ylim()

                xvals = pareto_final[:,0]
                xvals = np.append(np.array(xlim[0]),xvals)
                xvals = np.append(xvals,xvals[-1])
                yvals = pareto_final[:,1]
                yvals = np.append(np.array(yvals[0]),yvals)
                yvals = np.append(yvals,np.array(ylim[0]))

                sub.plot( xvals, yvals,
                              linewidth=lwidth,
                              color=lcol1,
                              alpha=0.7,
                              label=str2tex('merge of all calibration trials',usetex=usetex),  # MO: Calibrating Obj. #1 and Obj. #2\n
                              zorder=20)

            else:
                raise ValueError("Don't know what to do with algorithm {}.".format(algorithm_names[iiexample_group][iiexample]))

            # axis labels
            if ialgo == 'padds':
                if usetex:
                    sub.set_xlabel(str2tex('Obj. #1: '+imetric_str[0], usetex=usetex))
                    sub.set_ylabel(str2tex('Obj. #2: '+imetric_str[1], usetex=usetex))
                else:
                    sub.set_xlabel(str2tex('Obj. #1: '+imetric_str[0], usetex=usetex))
                    sub.set_ylabel(str2tex('Obj. #2: '+imetric_str[1], usetex=usetex))


        # last plot: just say its a zoom of the previous panel
        if iplot == 3:
            sub.text(0.5,0.95, str2tex('[This is a zoom of panel '+chr(96+iplot-1)+']', usetex=usetex),
                          fontsize=textsize-2,
                          color= '0.7',
                          transform=sub.transAxes, horizontalalignment='center', verticalalignment='top')

        # limits
        if iplot == 3:
            sub.set_xlim([ 0.95,1.03])
            sub.set_ylim([ 0.87,0.94])

        # captions above legend
        if iplot == 2:
            ylegcap = -0.4
            xlegcap = -1.2   # touch this number inly with -u activated
            dxlegcap = 0.845   # touch this number inly with -u activated

            sub.text( xlegcap+0*dxlegcap, ylegcap, str2tex('Single-Objective Calibration:\n[Obj. #1]',usetex=usetex),
                                    ha = 'left', va = 'top',
                                    transform=sub.transAxes,
                                    fontsize=textsize-2 )
            sub.text( xlegcap+1*dxlegcap, ylegcap, str2tex('Single-Objective Calibration:\n[Obj. #2]',usetex=usetex),
                                    ha = 'left', va = 'top',
                                    transform=sub.transAxes,
                                    fontsize=textsize-2 )
            sub.text( xlegcap+2*dxlegcap, ylegcap, str2tex('Single-Objective Calibration:\n[0.5$\cdot$(Obj. #1 + Obj. #2)]',usetex=usetex),
                                    ha = 'left', va = 'top',
                                    transform=sub.transAxes,
                                    fontsize=textsize-2 )
            sub.text( xlegcap+3*dxlegcap-0.02, ylegcap, str2tex('Multi-Objective Calibration:\n[Obj. #1, Obj. #2]',usetex=usetex),
                                    ha = 'left', va = 'top',
                                    transform=sub.transAxes,
                                    fontsize=textsize-2 )

        # legend
        if iplot == 2:
            ll = sub.legend(frameon=frameon, ncol=4,
                                labelspacing=llrspace, handletextpad=llhtextpad, handlelength=llhlength,
                                columnspacing=6.8,    # touch this number inly with -u activated
                                loc='upper center', bbox_to_anchor=(xlegcap+1.61,ylegcap-0.12), scatterpoints=1, numpoints=1,
                                fontsize = textsize-2)

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
