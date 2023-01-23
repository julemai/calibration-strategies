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
#     run figure_4.py -t pdf -p figure_4

#!/usr/bin/env python
from __future__ import print_function

"""

Plots differences when data span orders of magnitude

History
-------
Written,  JM, June 2022
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
                                      description='''Plots differences when data span orders of magnitude.''')
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

    import color                      # in lib/
    from position   import position   # in lib/
    from str2tex    import str2tex    # in lib/
    from autostring import astr       # in lib/
    from abc2plot   import abc2plot   # in lib/

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

    jsonfile = dir_path+'/figure_4.json'

    # -------------------------------------------------------------------------
    # general
    # -------------------------------------------------------------------------

    nsets = 10000
    npara = 4
    ntrials = 10


    if not(os.path.exists(jsonfile)):

        dict_results = {}

        # -------------------------------------------------------------------------
        # random data (from Katharia Ross: time vs. conductivity)
        # from: Documents/MMA/Katharina_Dispersivity_Fit.nb
        # -------------------------------------------------------------------------

        xvals = np.array([20000, 23, 3500, 4000, 2000, 122, 91, 15, 600, 91, 6, 800,
                          800, 17, 20000, 55, 50000, 14000, 4.4, 4.4, 4.4, 10.4,
                          10.4, 10.4, 100, 110, 500, 8, 8, 25, 13000, 18000, 4, 5,
                          16.4, 290, 8, 3, 1000, 32000, 11, 20, 40, 16, 43, 20000,
                          6.4, 10000, 3200, 5.3, 10.7, 25, 50, 490, 700, 43400, 30,
                          538, 700, 37, 105, 200, 79.2, 4.6, 100000])
        yvals = np.array([30.5, 5.2, 6, 460, 170, 15, 20, 3, 45, 11.6, 11, 15, 12, 2,
                          91, 38.1, 140, 30.5, 0.1, 0.01, 0.2, 0.3, 0.04, 0.7, 6.7,
                          10, 58, 3.1, 1, 1.6, 30.5, 30.5, 0.06, 0.01, 2.74, 41, 0.5,
                          0.03, 21.3, 21.5, 5, 2, 8, 4, 11, 910, 15.2, 61, 61, 0.3,
                          0.46, 11, 25, 6.7, 7.6, 91.4, 12.5, 134, 182, 131, 208, 234,
                          15.2, 0.55, 22800])

        dict_results['x'] = xvals
        dict_results['y'] = yvals
        dict_results['logx'] = np.log10(xvals)
        dict_results['logy'] = np.log10(yvals)

        # read SCE calibration results :: logx-logy
        para = {}
        obfv = {}
        for itrial in range(ntrials):

            # get calibrated values
            f = open("example_4/calibrate_with_sce_logx_logy/trial_"+str(itrial+1)+"/best/parameters.py", "r")
            content = f.readlines()
            content = [ cc.strip() for cc in content if not(cc.startswith('#')) ]
            content = [ cc for cc in content if not(cc.strip() == '') ]
            f.close()
            para['trial_'+str(itrial+1)] = [ float(cc.split('=')[1]) for cc in content ]

            # get objective function values
            f = open("example_4/calibrate_with_sce_logx_logy/trial_"+str(itrial+1)+"/best/objective_function.out", "r")
            content = f.readlines()
            content = [ cc.strip() for cc in content if not(cc.startswith('#')) ]
            content = [ cc for cc in content if not(cc.strip() == '') ]
            f.close()
            obfv['trial'+str(itrial+1)] = [ float(cc.split(',')[1]) for cc in content ][0]

        dict_results['para_logx_logy'] = para
        dict_results['obfv_logx_logy'] = obfv

        # read SCE calibration results :: x-y
        para = {}
        obfv = {}
        for itrial in range(ntrials):

            # get calibrated values
            f = open("example_4/calibrate_with_sce_x_y/trial_"+str(itrial+1)+"/best/parameters.py", "r")
            content = f.readlines()
            content = [ cc.strip() for cc in content if not(cc.startswith('#')) ]
            content = [ cc for cc in content if not(cc.strip() == '') ]
            f.close()
            para['trial_'+str(itrial+1)] = [ float(cc.split('=')[1]) for cc in content ]

            # get objective function values
            f = open("example_4/calibrate_with_sce_x_y/trial_"+str(itrial+1)+"/best/objective_function.out", "r")
            content = f.readlines()
            content = [ cc.strip() for cc in content if not(cc.startswith('#')) ]
            content = [ cc for cc in content if not(cc.strip() == '') ]
            f.close()
            obfv['trial'+str(itrial+1)] = [ float(cc.split(',')[1]) for cc in content ][0]

        dict_results['para_x_y'] = para
        dict_results['obfv_x_y'] = obfv

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
    ncol        = 2           # number columns
    nrow        = 4           # number of rows
    textsize    = 8          # standard text size
    dxabc       = 0.95          # % of (max-min) shift to the right from left y-axis for a,b,c,... labels
    dyabc       = 0.92          # % of (max-min) shift up from lower x-axis for a,b,c,... labels
    dxsig       = 1.23        # % of (max-min) shift to the right from left y-axis for signature
    dysig       = -0.075      # % of (max-min) shift up from lower x-axis for signature
    dxtit       = 0           # % of (max-min) shift to the right from left y-axis for title
    dytit       = 1.2         # % of (max-min) shift up from lower x-axis for title
    hspace      = 0.23        # x-space between subplots
    vspace      = 0.06        # y-space between subplots

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
    lcols       = color.colours(['black','blue','green','yellow'])
    markers     = ['o','v','s','^']

    # Legend
    llxbbox     = 0.01        # x-anchor legend bounding box
    llybbox     = 1.0        # y-anchor legend bounding box
    llrspace    = 0.          # spacing between rows in legend
    llcspace    = 1.0         # spacing between columns in legend
    llhtextpad  = 0.4         # the pad between the legend handle and text
    llhlength   = 1.5         # the length of the legend handles
    frameon     = False       # if True, draw a frame around the legend. If None, use rc

    import matplotlib as mpl
    import matplotlib.patches as patches
    from matplotlib.lines import Line2D
    import matplotlib.colors as mcolors
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

    # -------------
    # Samples points
    # -------------


    # -------------------------
    # plot - values with only log x/y axis
    # -------------------------
    iplot += 1
    sub = fig.add_axes(position(nrow, ncol, iplot, hspace=hspace/2, vspace=vspace))

    for itrial in range(ntrials):

        para = dict_results['para_x_y']['trial_'+str(itrial+1)]

        # transform with log10
        xgrid = np.arange( np.min(dict_results['logx']),  # logx == just that functions look better with a log-log axis
                              np.max(dict_results['logx']),
                              (np.max(dict_results['logx'])-np.min(dict_results['logx'])) / 1000. )

        # print("fit x/y: trial: {} para: {}".format(itrial+1,para))

        # f(x) = 10.0 ^ ( L/(1 + Exp[-k*(Log10[x] - x0)]) - s )
        # ymod = 10. ** ( p[0] / ( 1.0 + np.exp(-p[1]*(np.log10(xvals) - p[2])) ) - p[3] )
        ymod = 10. ** ( para[0] / ( 1.0 + np.exp(-para[1]*(np.log10(10**xgrid) - para[2])) ) - para[3] )  # 10**xgrid just because xgrid is log-vals

        if itrial == 0:
            label = str2tex('Calibration using native scale (x,y)',usetex=usetex)
        else:
            label = ''

        sub.plot( 10**xgrid,ymod,   # 10**xgrid just because xgrid is log-vals
                  linewidth=lwidth, linestyle='-', color=lcol1, alpha=0.7, zorder=100, label=label)

    # # just in background the logx-logy calibrated functions
    # for itrial in range(ntrials):

    #     para = dict_results['para_logx_logy']['trial_'+str(itrial+1)]

    #     # transform with log10
    #     xgridlog = np.arange( np.min(dict_results['logx']),
    #                           np.max(dict_results['logx']),
    #                           (np.max(dict_results['logx'])-np.min(dict_results['logx'])) / 1000. )

    #     print("fit logx/logy: trial: {} para: {}".format(itrial+1,para))

    #     # f(x) = L/(1 + Exp[-k*(x - x0)]) - s
    #     # ymod  = p[0]    / ( 1.0 + np.exp( -p[1]    * ( logxvals - p[2]    ))) - p[3]
    #     ymodlog = para[0] / ( 1.0 + np.exp( -para[1] * ( xgridlog - para[2] ))) - para[3]

    #     if itrial == 0:
    #         label = str2tex('Calibration using log-scale (log$_{10}$x,log$_{10}$y)',usetex=usetex)
    #     else:
    #         label = ''

    #     sub.plot( 10**xgridlog,10**ymodlog,
    #               linewidth=lwidth, linestyle='--', color='0.8', alpha=0.9, zorder=50, label=label)

    label = str2tex('Data points (x,y)',usetex=usetex)
    sub.plot( dict_results['x'],dict_results['y'],
                  linewidth=0.0, marker='o', color=lcol1,
                  markersize=msize/1, markeredgewidth=msize/4,markerfacecolor='w',
                  alpha=0.7, zorder=400, label=label)

    # sub.set_xlabel(str2tex('Time [$t/$\u03C4$_u$]', usetex=usetex))
    # sub.set_ylabel(str2tex('Dispersion Coeffi. [$m^2/d$]', usetex=usetex))
    sub.set_xlabel(str2tex('native x', usetex=usetex))
    sub.set_ylabel(str2tex('native y', usetex=usetex))

    # limits
    xmin=np.min(dict_results['logx'])
    xmax=np.max(dict_results['logx'])
    ymin=np.min(dict_results['logy'])
    ymax=np.max(dict_results['logy'])
    delta=0.02
    sub.set_xlim([10**(xmin-delta*(xmax-xmin)),10**(xmax+delta*(xmax-xmin))])
    sub.set_ylim([10**(ymin-delta*(ymax-ymin)),10**(ymax+delta*4*(ymax-ymin))])

    sub.set_xscale('log')
    sub.set_yscale('log')

    # legend
    sub.legend(frameon=frameon, ncol=1,
                            labelspacing=llrspace, handletextpad=llhtextpad, handlelength=llhlength,
                            loc='upper left', bbox_to_anchor=(llxbbox,llybbox), scatterpoints=1, numpoints=1,
                            fontsize = textsize-2)

    # abc
    sub.text( 1.05, 1.0, str2tex(chr(96+iplot),usetex=usetex),
                                ha = 'left', va = 'top',
                                fontweight='bold',
                                transform=sub.transAxes,
                                fontsize=textsize+3 )

    # fill row
    #iplot += 1


    # -------------------------
    # plot - log x/y values and fit function (in logx/logy scale)
    # -------------------------
    iplot += 1
    sub = fig.add_axes(position(nrow, ncol, iplot, hspace=hspace/2, vspace=vspace))


    for itrial in range(ntrials):

        para = dict_results['para_logx_logy']['trial_'+str(itrial+1)]
        # transfform with loge
        xgridlog = np.arange( np.min(np.log(dict_results['x'])),
                              np.max(np.log(dict_results['x'])),
                              (np.max(np.log(dict_results['x']))-np.min(np.log(dict_results['x']))) / 1000. )
        # transform with log10
        xgridlog = np.arange( np.min(dict_results['logx']),
                              np.max(dict_results['logx']),
                              (np.max(dict_results['logx'])-np.min(dict_results['logx'])) / 1000. )

        # print("fit logx/logy: trial: {} para: {}".format(itrial+1,para))

        # f(x) = L/(1 + Exp[-k*(x - x0)]) - s
        # ymod  = p[0]    / ( 1.0 + np.exp( -p[1]    * ( logxvals - p[2]    ))) - p[3]
        ymodlog = para[0] / ( 1.0 + np.exp( -para[1] * ( xgridlog - para[2] ))) - para[3]

        if itrial == 0:
            label = str2tex('Calibration using log-scale (log$_{10}$x,log$_{10}$y)',usetex=usetex)
        else:
            label = ''
        sub.plot( xgridlog,ymodlog,
                  linewidth=lwidth, linestyle='-', color=lcol1, alpha=0.7, zorder=100, label=label)

    # transform with log10
    label = str2tex('Data points (log$_{10}$x,log$_{10}$y)',usetex=usetex)
    sub.plot( dict_results['logx'],dict_results['logy'],
                  linewidth=0.0, marker='o', color=lcol1,
                  markersize=msize/1, markeredgewidth=msize/4,markerfacecolor='w',
                  alpha=0.7, zorder=400, label=label)

    sub.set_xlabel(str2tex('Log Time  [$t/$\u03C4$_u$]', usetex=usetex))
    sub.set_ylabel(str2tex('Log Dispersion Coeffi. [$m^2/d$]', usetex=usetex))
    sub.set_xlabel(str2tex('log-transformed x', usetex=usetex))
    sub.set_ylabel(str2tex('log-transformed y', usetex=usetex))

    # limits
    xmin=np.min(dict_results['logx'])
    xmax=np.max(dict_results['logx'])
    ymin=np.min(dict_results['logy'])
    ymax=np.max(dict_results['logy'])
    delta=0.02
    sub.set_xlim([xmin-delta*(xmax-xmin),xmax+delta*(xmax-xmin)])
    sub.set_ylim([ymin-delta*(ymax-ymin),ymax+delta*4*(ymax-ymin)])

    # set ticks same as in previous plot
    sub.set_yticks([-1,1,3])

    # legend
    sub.legend(frameon=frameon, ncol=1,
                            labelspacing=llrspace, handletextpad=llhtextpad, handlelength=llhlength,
                            loc='upper left', bbox_to_anchor=(llxbbox,llybbox), scatterpoints=1, numpoints=1,
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
    str = '  Time plot [m]: '+astr((t2-t1)/60.,1) if (t2-t1)>60. else '  Time plot [s]: '+astr(t2-t1,0)
    print(str)
