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

Plots schematic sub-figures that will get combined to figure 9 via LaTeX

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
                                      description='''Plots schematic sub-figures that will get combined to figure 9 via LaTeX .''')
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
        fgcolor = '0.5'  # black
        bgcolor = 'white'


    # fix seed for reproducible results
    np.random.seed(seed=123)

    jsonfile7 = dir_path+'/figure_7.json'
    jsonfile8 = dir_path+'/figure_8.json'
    jsonfile9 = dir_path+'/figure_9.json'
    jsonfile10 = dir_path+'/figure_10.json'

    if not(os.path.exists(jsonfile7)):
        raise ValueError('Needs {}. Please run figure_4.py first.'.format(jsonfile7))
    else:
        # read from json file
        with open(jsonfile7) as ff:
            dict_results7 = json.load(ff)

    if not(os.path.exists(jsonfile8)):
        raise ValueError('Needs {}. Please run figure_6.py first.'.format(jsonfile8))
    else:
        # read from json file
        with open(jsonfile8) as ff:
            dict_results8 = json.load(ff)

    if not(os.path.exists(jsonfile9)):
        raise ValueError('Needs {}. Please run figure_7.py first.'.format(jsonfile9))
    else:
        # read from json file
        with open(jsonfile9) as ff:
            dict_results9 = json.load(ff)

    if not(os.path.exists(jsonfile10)):
        raise ValueError('Needs {}. Please run figure_8.py first.'.format(jsonfile10))
    else:
        # read from json file
        with open(jsonfile10) as ff:
            dict_results10 = json.load(ff)


    ntrials = 10

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
    markers     = ['o','<','v','s','^']  #['o','v','s','^']
    transparent = True

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

    import matplotlib as mpl
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
    mpl.rc('figure', figsize=(4./5.*10.97,4./5.*11.69)) # a4 portrait
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
    iiplot = 0

    # -----------------------
    # plot
    # -----------------------

    # -------------------------
    # plot (A) - parameter values: converge to different values (Fig. 4l)
    # -------------------------
    ifig += 1
    iplot = 1
    iiplot += 1
    print('Plot - Fig ', chr(96+iiplot))
    fig = plt.figure(ifig)

    sub = fig.add_axes(position(nrow, ncol, iplot, hspace=hspace, vspace=vspace))

    # plot
    ipara = 2
    ranges_narrow = [ [10,20],
                      [2.5,5.0],
                      [1,3],
                      [0,5]]
    for itrial in range(ntrials):

            xvals = dict_results7['hist_logx_logy_narrow']['trial'+str(itrial+1)]['iter']
            yvals = np.array(dict_results7['hist_logx_logy_narrow']['trial'+str(itrial+1)]['para'])[:,ipara]
            sub.plot( xvals,yvals,
                  linewidth=lwidth, linestyle='-', color=lcol1, alpha=0.7, zorder=100)

    sub.plot( [0,500], [ranges_narrow[ipara][0],ranges_narrow[ipara][0]], linestyle='--', linewidth=lwidth/2, color='0.5', alpha=0.7)
    sub.plot( [0,500], [ranges_narrow[ipara][1],ranges_narrow[ipara][1]], linestyle='--', linewidth=lwidth/2, color='0.5', alpha=0.7)

    sub.text( 0.95*500, ranges_narrow[ipara][0]-(ranges_narrow[ipara][1]-ranges_narrow[ipara][0])*0.03, str2tex('Lower limit', usetex=usetex),
                      #transform=sub.transAxes,
                      fontsize=textsize-2,
                      color='0.5',
                      horizontalalignment='right', verticalalignment='top')
    sub.text( 0.95*500, ranges_narrow[ipara][1]+(ranges_narrow[ipara][1]-ranges_narrow[ipara][0])*0.03, str2tex('Upper limit', usetex=usetex),
                      #transform=sub.transAxes,
                      fontsize=textsize-2,
                      color='0.5',
                      horizontalalignment='right', verticalalignment='bottom')

    # limits
    sub.set_xlim([-10,510])
    sub.set_ylim(ranges_narrow[ipara][0]-(ranges_narrow[ipara][1]-ranges_narrow[ipara][0])*0.2,ranges_narrow[ipara][1]+(ranges_narrow[ipara][1]-ranges_narrow[ipara][0])*0.2)

    # no ticks, no ticklabels
    sub.set_xticks([])
    sub.set_yticks([])
    sub.set_xlabel(str2tex('Iteration',usetex=usetex))
    sub.set_ylabel(str2tex('Parameter value',usetex=usetex))

    # abc
    sub.text( 1.05, 1.0, str2tex(chr(96+iiplot),usetex=usetex),
                                        ha = 'left', va = 'top',
                                        fontweight='bold',
                                        transform=sub.transAxes,
                                        fontsize=textsize+3 )

    if (outtype == 'pdf'):
        pdf_pages.savefig(fig, transparent=transparent)
        plt.close(fig)
    elif (outtype == 'png'):
        pngfile = pngbase+"{0:04d}".format(ifig)+".png"
        fig.savefig(pngfile, transparent=transparent, bbox_inches=bbox_inches, pad_inches=pad_inches)
        plt.close(fig)

    # -------------------------
    # plot (B) - parameter values: converge against range boundary (Fig. 4f)
    # -------------------------
    ifig += 1
    iplot = 1
    iiplot += 1
    print('Plot - Fig ', chr(96+iiplot))
    fig = plt.figure(ifig)

    sub = fig.add_axes(position(nrow, ncol, iplot, hspace=hspace, vspace=vspace))

    # plot
    ipara = 0
    ranges_narrow = [ [10,20],
                      [2.5,5.0],
                      [1,3],
                      [0,5]]
    for itrial in range(ntrials):

            xvals = dict_results7['hist_logx_logy_narrow']['trial'+str(itrial+1)]['iter']
            yvals = np.array(dict_results7['hist_logx_logy_narrow']['trial'+str(itrial+1)]['para'])[:,ipara]
            sub.plot( xvals,yvals,
                  linewidth=lwidth, linestyle='-', color=lcol1, alpha=0.7, zorder=100)

    sub.plot( [0,500], [ranges_narrow[ipara][0],ranges_narrow[ipara][0]], linestyle='--', linewidth=lwidth/2, color='0.5', alpha=0.7)
    sub.plot( [0,500], [ranges_narrow[ipara][1],ranges_narrow[ipara][1]], linestyle='--', linewidth=lwidth/2, color='0.5', alpha=0.7)

    sub.text( 0.95*500, ranges_narrow[ipara][0]-(ranges_narrow[ipara][1]-ranges_narrow[ipara][0])*0.03, str2tex('Lower limit', usetex=usetex),
                      #transform=sub.transAxes,
                      fontsize=textsize-2,
                      color='0.5',
                      horizontalalignment='right', verticalalignment='top')
    sub.text( 0.95*500, ranges_narrow[ipara][1]+(ranges_narrow[ipara][1]-ranges_narrow[ipara][0])*0.03, str2tex('Upper limit', usetex=usetex),
                      #transform=sub.transAxes,
                      fontsize=textsize-2,
                      color='0.5',
                      horizontalalignment='right', verticalalignment='bottom')

    # limits
    sub.set_xlim([-10,510])
    sub.set_ylim(ranges_narrow[ipara][0]-(ranges_narrow[ipara][1]-ranges_narrow[ipara][0])*0.2,ranges_narrow[ipara][1]+(ranges_narrow[ipara][1]-ranges_narrow[ipara][0])*0.2)

    # no ticks, no ticklabels
    sub.set_xticks([])
    sub.set_yticks([])
    sub.set_xlabel(str2tex('Iteration',usetex=usetex))
    sub.set_ylabel(str2tex('Parameter value',usetex=usetex))

    # abc
    sub.text( 1.05, 1.0, str2tex(chr(96+iiplot),usetex=usetex),
                                        ha = 'left', va = 'top',
                                        fontweight='bold',
                                        transform=sub.transAxes,
                                        fontsize=textsize+3 )

    if (outtype == 'pdf'):
        pdf_pages.savefig(fig, transparent=transparent)
        plt.close(fig)
    elif (outtype == 'png'):
        pngfile = pngbase+"{0:04d}".format(ifig)+".png"
        fig.savefig(pngfile, transparent=transparent, bbox_inches=bbox_inches, pad_inches=pad_inches)
        plt.close(fig)

    # -------------------------
    # plot (C) - parameter values: looks great (Fig. 4h)
    # -------------------------
    ifig += 1
    iplot = 1
    iiplot += 1
    print('Plot - Fig ', chr(96+iiplot))
    fig = plt.figure(ifig)

    sub = fig.add_axes(position(nrow, ncol, iplot, hspace=hspace, vspace=vspace))

    ipara = 0
    ranges_ori = [    [-2,20],
                      [0,5],
                      [-2,2],
                      [0,10]]
    for itrial in range(ntrials):

            xvals = dict_results7['hist_logx_logy_ori']['trial'+str(itrial+1)]['iter']
            yvals = np.array(dict_results7['hist_logx_logy_ori']['trial'+str(itrial+1)]['para'])[:,ipara]
            sub.plot( xvals,yvals,
                  linewidth=lwidth, linestyle='-', color=lcol1, alpha=0.7, zorder=100)

    sub.plot( [0,500], [ranges_ori[ipara][0],ranges_ori[ipara][0]], linestyle='--', linewidth=lwidth/2, color='0.5', alpha=0.7)
    sub.plot( [0,500], [ranges_ori[ipara][1],ranges_ori[ipara][1]], linestyle='--', linewidth=lwidth/2, color='0.5', alpha=0.7)

    sub.text( 0.95*500, ranges_ori[ipara][0]-(ranges_ori[ipara][1]-ranges_ori[ipara][0])*0.03, str2tex('Lower limit', usetex=usetex),
                      #transform=sub.transAxes,
                      fontsize=textsize-2,
                      color='0.5',
                      horizontalalignment='right', verticalalignment='top')
    sub.text( 0.95*500, ranges_ori[ipara][1]+(ranges_ori[ipara][1]-ranges_ori[ipara][0])*0.03, str2tex('Upper limit', usetex=usetex),
                      #transform=sub.transAxes,
                      fontsize=textsize-2,
                      color='0.5',
                      horizontalalignment='right', verticalalignment='bottom')

    # limits
    sub.set_xlim([-10,510])
    sub.set_ylim(ranges_ori[ipara][0]-(ranges_ori[ipara][1]-ranges_ori[ipara][0])*0.2,ranges_ori[ipara][1]+(ranges_ori[ipara][1]-ranges_ori[ipara][0])*0.2)

    # no ticks, no ticklabels
    sub.set_xticks([])
    sub.set_yticks([])
    sub.set_xlabel(str2tex('Iteration',usetex=usetex))
    sub.set_ylabel(str2tex('Parameter value',usetex=usetex))

    # abc
    sub.text( 1.05, 1.0, str2tex(chr(96+iiplot),usetex=usetex),
                                        ha = 'left', va = 'top',
                                        fontweight='bold',
                                        transform=sub.transAxes,
                                        fontsize=textsize+3 )

    if (outtype == 'pdf'):
        pdf_pages.savefig(fig, transparent=transparent)
        plt.close(fig)
    elif (outtype == 'png'):
        pngfile = pngbase+"{0:04d}".format(ifig)+".png"
        fig.savefig(pngfile, transparent=transparent, bbox_inches=bbox_inches, pad_inches=pad_inches)
        plt.close(fig)

    # -------------------------
    # plot (D) - objective function values: SO: does not flatline (Fig. 6l medium budget)
    # -------------------------
    ifig += 1
    iplot = 1
    iiplot += 1
    print('Plot - Fig ', chr(96+iiplot))
    fig = plt.figure(ifig)

    sub = fig.add_axes(position(nrow, ncol, iplot, hspace=hspace, vspace=vspace))

    # plot
    ibudget  = 1000
    imetric  = 'fx'
    iexample = 'ostrich-ackley20'
    ialgo    = 'pso'

    for itrial in range(ntrials):

         xvals = dict_results9[ialgo][iexample][str(ibudget)][imetric]['hist']['trial_'+str(itrial+1)]['iter']
         yvals = dict_results9[ialgo][iexample][str(ibudget)][imetric]['hist']['trial_'+str(itrial+1)]['obfv']
         sub.plot( xvals, yvals,
                       color=lcol1,
                       alpha=0.7,
                       zorder=40)

    # no ticks, no ticklabels
    sub.set_xticks([])
    sub.set_yticks([])
    sub.set_xlabel(str2tex('Iteration',usetex=usetex))
    sub.set_ylabel(str2tex('Obj. function value',usetex=usetex))

    # abc
    sub.text( 1.05, 1.0, str2tex(chr(96+iiplot),usetex=usetex),
                                        ha = 'left', va = 'top',
                                        fontweight='bold',
                                        transform=sub.transAxes,
                                        fontsize=textsize+3 )

    sub.set_facecolor('white')

    if (outtype == 'pdf'):
        pdf_pages.savefig(fig,transparent=transparent)
        plt.close(fig)
    elif (outtype == 'png'):
        pngfile = pngbase+"{0:04d}".format(ifig)+".png"
        fig.savefig(pngfile, transparent=transparent, bbox_inches=bbox_inches, pad_inches=pad_inches)
        plt.close(fig)

    # -------------------------
    # plot (E) - objective function values: SO: wide spread of trials (Fig. 6k medium budget)
    # -------------------------
    ifig += 1
    iplot = 1
    iiplot += 1
    print('Plot - Fig ', chr(96+iiplot))
    fig = plt.figure(ifig)

    sub = fig.add_axes(position(nrow, ncol, iplot, hspace=hspace, vspace=vspace))

    # plot
    ibudget  = 2000
    imetric  = 'fx'
    iexample = 'ostrich-ackley20'
    ialgo    = 'sce'

    for itrial in range(ntrials):

         xvals = dict_results9[ialgo][iexample][str(ibudget)][imetric]['hist']['trial_'+str(itrial+1)]['iter']
         yvals = dict_results9[ialgo][iexample][str(ibudget)][imetric]['hist']['trial_'+str(itrial+1)]['obfv']
         sub.plot( xvals, yvals,
                       color=lcol1,
                       alpha=0.7,
                       zorder=40)

    # no ticks, no ticklabels
    sub.set_xticks([])
    sub.set_yticks([])
    sub.set_xlabel(str2tex('Iteration',usetex=usetex))
    sub.set_ylabel(str2tex('Obj. function value',usetex=usetex))

    # abc
    sub.text( 1.05, 1.0, str2tex(chr(96+iiplot),usetex=usetex),
                                        ha = 'left', va = 'top',
                                        fontweight='bold',
                                        transform=sub.transAxes,
                                        fontsize=textsize+3 )

    if (outtype == 'pdf'):
        pdf_pages.savefig(fig, transparent=transparent)
        plt.close(fig)
    elif (outtype == 'png'):
        pngfile = pngbase+"{0:04d}".format(ifig)+".png"
        fig.savefig(pngfile, transparent=transparent, bbox_inches=bbox_inches, pad_inches=pad_inches)
        plt.close(fig)

    # -------------------------
    # plot (F) - objective function values: SO: looks good (Fig. 8f large budget)
    # -------------------------
    ifig += 1
    iplot = 1
    iiplot += 1
    print('Plot - Fig ', chr(96+iiplot))
    fig = plt.figure(ifig)

    sub = fig.add_axes(position(nrow, ncol, iplot, hspace=hspace, vspace=vspace))

    # plot
    ibudget  = 400
    imetric  = 'rmse'
    iexample = 'ostrich-dispersivity'
    ialgo    = 'dds'

    for itrial in [1,2,3,4,5,6,7,8,9]: #range(ntrials):

         xvals = dict_results9[ialgo][iexample][str(ibudget)][imetric]['hist']['trial_'+str(itrial+1)]['iter']
         yvals = dict_results9[ialgo][iexample][str(ibudget)][imetric]['hist']['trial_'+str(itrial+1)]['obfv']
         sub.plot( xvals, yvals,
                       color=lcol1,
                       alpha=0.7,
                       zorder=40)

    # no ticks, no ticklabels
    sub.set_xticks([])
    sub.set_yticks([])
    sub.set_xlabel(str2tex('Iteration',usetex=usetex))
    sub.set_ylabel(str2tex('Obj. function value',usetex=usetex))

    sub.set_ylim([0.55,2.5])

    # abc
    sub.text( 1.05, 1.0, str2tex(chr(96+iiplot),usetex=usetex),
                                        ha = 'left', va = 'top',
                                        fontweight='bold',
                                        transform=sub.transAxes,
                                        fontsize=textsize+3 )

    if (outtype == 'pdf'):
        pdf_pages.savefig(fig, transparent=transparent)
        plt.close(fig)
    elif (outtype == 'png'):
        pngfile = pngbase+"{0:04d}".format(ifig)+".png"
        fig.savefig(pngfile, transparent=transparent, bbox_inches=bbox_inches, pad_inches=pad_inches)
        plt.close(fig)


    # -------------------------
    # plot (G) - objective function values: MO: degenerated front (Fig. 7b)
    # -------------------------
    ifig += 1
    iplot = 1
    iiplot += 1
    print('Plot - Fig ', chr(96+iiplot))
    fig = plt.figure(ifig)

    sub = fig.add_axes(position(nrow, ncol, iplot, hspace=hspace, vspace=vspace))

    ialgo = 'padds'
    iiexample_group = 1
    imetric = ['kgeb', 'kger']

    # merge all pareto fronts and plot in blue
    all_final_obfv = [ dict_results10['experiment_'+str(iiexample_group+1)][ialgo+"_"+"_".join(imetric)]['obfv']['trial_'+str(itrial+1)] for itrial in range(ntrials) ]
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
    xlim=sub.set_xlim([.97,1.02])
    ylim=sub.set_ylim([0.87,0.94])

    xvals = pareto_final[:,0]
    xvals = np.append(np.array(xlim[0]),xvals)
    xvals = np.append(xvals,xvals[-1])
    yvals = pareto_final[:,1]
    yvals = np.append(np.array(yvals[0]),yvals)
    yvals = np.append(yvals,np.array(ylim[0]))

    sub.plot( xvals, yvals,
                  linewidth=lwidth*1.5,
                  color=lcol1,
                  alpha=0.7,
                  label=str2tex('merge of all calibration trials',usetex=usetex),  # MO: Calibrating Obj. #1 and Obj. #2\n
                  zorder=20)
    sub.text( 0.973,0.908, str2tex('Pareto front',usetex=usetex),
                      #transform=sub.transAxes,
                      fontsize=textsize-2,
                      color=lcol1,
                      horizontalalignment='left', verticalalignment='top')

    # SO reeference points
    sub.plot( [0.98], [0.913],
                  linewidth=0.0,
                  marker=markers[2],
                  markersize=msize*2, markeredgewidth=msize/3,markerfacecolor='none',
                  color='0.5',
                  alpha=0.7,
                  zorder=10)
    sub.plot( [1.0], [0.89],
                  linewidth=0.0,
                  marker=markers[1],
                  markersize=msize*2, markeredgewidth=msize/3,markerfacecolor='none',
                  color='0.5',
                  alpha=0.7,
                  zorder=10)
    sub.plot( [0.9995], [0.912],
                  linewidth=0.0,
                  marker=markers[3],
                  markersize=msize*2, markeredgewidth=msize/3,markerfacecolor='none',
                  color='0.5',
                  alpha=0.7,
                  zorder=10)
    sub.text( 1.0,0.9155, str2tex('Single-objective\nreferences',usetex=usetex),
                      #transform=sub.transAxes,
                      fontsize=textsize-2,
                      color='0.5',
                      horizontalalignment='center', verticalalignment='bottom')

    # no ticks, no ticklabels
    sub.set_xticks([])
    sub.set_yticks([])
    sub.set_xlabel(str2tex('Objective 1',usetex=usetex))
    sub.set_ylabel(str2tex('Objective 2',usetex=usetex))

    # abc
    sub.text( 1.05, 1.0, str2tex(chr(96+iiplot),usetex=usetex),
                                        ha = 'left', va = 'top',
                                        fontweight='bold',
                                        transform=sub.transAxes,
                                        fontsize=textsize+3 )

    if (outtype == 'pdf'):
        pdf_pages.savefig(fig, transparent=transparent)
        plt.close(fig)
    elif (outtype == 'png'):
        pngfile = pngbase+"{0:04d}".format(ifig)+".png"
        fig.savefig(pngfile, transparent=transparent, bbox_inches=bbox_inches, pad_inches=pad_inches)
        plt.close(fig)


    # -------------------------
    # plot (H) - objective function values: MO: not converged yet front (Fig. 7a; faked --> move ref points away)
    # -------------------------
    ifig += 1
    iplot = 1
    iiplot += 1
    print('Plot - Fig ', chr(96+iiplot))
    fig = plt.figure(ifig)

    sub = fig.add_axes(position(nrow, ncol, iplot, hspace=hspace, vspace=vspace))

    ialgo = 'padds'
    iiexample_group = 0
    imetric = ['lkgelow05', 'lkgehgh95']

    # merge all pareto fronts and plot in blue
    all_final_obfv = [ dict_results10['experiment_'+str(iiexample_group+1)][ialgo+"_"+"_".join(imetric)]['obfv']['trial_'+str(itrial+1)] for itrial in range(ntrials) ]
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
    xlim=sub.set_xlim([0.15,0.90])
    ylim=sub.set_ylim([0.2,1.2])

    xvals = pareto_final[:,0]
    xvals = np.append(np.array(xlim[0]),xvals)
    xvals = np.append(xvals,xvals[-1])
    yvals = pareto_final[:,1]
    yvals = np.append(np.array(yvals[0]),yvals)
    yvals = np.append(yvals,np.array(ylim[0]))

    sub.plot( xvals, yvals,
                  linewidth=lwidth*1.5,
                  color=lcol1,
                  alpha=0.7,
                  label=str2tex('merge of all calibration trials',usetex=usetex),  # MO: Calibrating Obj. #1 and Obj. #2\n
                  zorder=20)
    sub.text( 0.19,0.83, str2tex('Pareto front',usetex=usetex),
                      #transform=sub.transAxes,
                      fontsize=textsize-2,
                      color=lcol1,
                      horizontalalignment='left', verticalalignment='top')

    # SO reeference points
    sub.plot( [0.3], [1.0],
                  linewidth=0.0,
                  marker=markers[2],
                  markersize=msize*2, markeredgewidth=msize/3,markerfacecolor='none',
                  color='0.5',
                  alpha=0.7,
                  zorder=10)
    sub.plot( [0.75], [0.4],
                  linewidth=0.0,
                  marker=markers[1],
                  markersize=msize*2, markeredgewidth=msize/3,markerfacecolor='none',
                  color='0.5',
                  alpha=0.7,
                  zorder=10)
    sub.plot( [0.7], [0.9],
                  linewidth=0.0,
                  marker=markers[3],
                  markersize=msize*2, markeredgewidth=msize/3,markerfacecolor='none',
                  color='0.5',
                  alpha=0.7,
                  zorder=10)
    sub.text( 0.7,0.95, str2tex('Single-objective\nreferences',usetex=usetex),
                      #transform=sub.transAxes,
                      fontsize=textsize-2,
                      color='0.5',
                      horizontalalignment='center', verticalalignment='bottom')

    # no ticks, no ticklabels
    sub.set_xticks([])
    sub.set_yticks([])
    sub.set_xlabel(str2tex('Objective 1',usetex=usetex))
    sub.set_ylabel(str2tex('Objective 2',usetex=usetex))

    # abc
    sub.text( 1.05, 1.0, str2tex(chr(96+iiplot),usetex=usetex),
                                        ha = 'left', va = 'top',
                                        fontweight='bold',
                                        transform=sub.transAxes,
                                        fontsize=textsize+3 )

    if (outtype == 'pdf'):
        pdf_pages.savefig(fig, transparent=transparent)
        plt.close(fig)
    elif (outtype == 'png'):
        pngfile = pngbase+"{0:04d}".format(ifig)+".png"
        fig.savefig(pngfile, transparent=transparent, bbox_inches=bbox_inches, pad_inches=pad_inches)
        plt.close(fig)


    # -------------------------
    # plot (I) - objective function values: MO: looks good (Fig. 7a)
    # -------------------------
    ifig += 1
    iplot = 1
    iiplot += 1
    print('Plot - Fig ', chr(96+iiplot))
    fig = plt.figure(ifig)

    sub = fig.add_axes(position(nrow, ncol, iplot, hspace=hspace, vspace=vspace))

    ialgo = 'padds'
    iiexample_group = 0
    imetric = ['lkgelow05', 'lkgehgh95']

    # merge all pareto fronts and plot in blue
    all_final_obfv = [ dict_results10['experiment_'+str(iiexample_group+1)][ialgo+"_"+"_".join(imetric)]['obfv']['trial_'+str(itrial+1)] for itrial in range(ntrials) ]
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
    xlim=sub.set_xlim([0.15,0.82])
    ylim=sub.set_ylim([0.2,1.1])

    xvals = pareto_final[:,0]
    xvals = np.append(np.array(xlim[0]),xvals)
    xvals = np.append(xvals,xvals[-1])
    yvals = pareto_final[:,1]
    yvals = np.append(np.array(yvals[0]),yvals)
    yvals = np.append(yvals,np.array(ylim[0]))

    sub.plot( xvals, yvals,
                  linewidth=lwidth*1.5,
                  color=lcol1,
                  alpha=0.7,
                  label=str2tex('merge of all calibration trials',usetex=usetex),  # MO: Calibrating Obj. #1 and Obj. #2\n
                  zorder=20)
    sub.text( 0.19,0.83, str2tex('Pareto front',usetex=usetex),
                      #transform=sub.transAxes,
                      fontsize=textsize-2,
                      color=lcol1,
                      horizontalalignment='left', verticalalignment='top')

    # SO reeference points
    sub.plot( [0.3], [0.89],
                  linewidth=0.0,
                  marker=markers[2],
                  markersize=msize*2, markeredgewidth=msize/3,markerfacecolor='none',
                  color='0.5',
                  alpha=0.7,
                  zorder=10)
    sub.plot( [0.67], [0.4],
                  linewidth=0.0,
                  marker=markers[1],
                  markersize=msize*2, markeredgewidth=msize/3,markerfacecolor='none',
                  color='0.5',
                  alpha=0.7,
                  zorder=10)
    sub.plot( [0.60], [0.81],
                  linewidth=0.0,
                  marker=markers[3],
                  markersize=msize*2, markeredgewidth=msize/3,markerfacecolor='none',
                  color='0.5',
                  alpha=0.7,
                  zorder=10)
    sub.text( 0.6,0.9, str2tex('Single-objective\nreferences',usetex=usetex),
                      #transform=sub.transAxes,
                      fontsize=textsize-2,
                      color='0.5',
                      horizontalalignment='center', verticalalignment='bottom')

    # no ticks, no ticklabels
    sub.set_xticks([])
    sub.set_yticks([])
    sub.set_xlabel(str2tex('Objective 1',usetex=usetex))
    sub.set_ylabel(str2tex('Objective 2',usetex=usetex))

    # abc
    sub.text( 1.05, 1.0, str2tex(chr(96+iiplot),usetex=usetex),
                                        ha = 'left', va = 'top',
                                        fontweight='bold',
                                        transform=sub.transAxes,
                                        fontsize=textsize+3 )

    if (outtype == 'pdf'):
        pdf_pages.savefig(fig, transparent=transparent)
        plt.close(fig)
    elif (outtype == 'png'):
        pngfile = pngbase+"{0:04d}".format(ifig)+".png"
        fig.savefig(pngfile, transparent=transparent, bbox_inches=bbox_inches, pad_inches=pad_inches)
        plt.close(fig)


    # -------------------------
    # plot (J) - compare sim vs obs: Only high values (Fig. 5d faked --> move up to match highflow)
    # -------------------------
    ifig += 1
    iplot = 1
    iiplot += 1
    print('Plot - Fig ', chr(96+iiplot))
    fig = plt.figure(ifig)

    sub = fig.add_axes(position(nrow, ncol, iplot, hspace=hspace, vspace=vspace))

    # plot
    imetric = 'lnse'

    start = 70
    xvals = np.array(dict_results8['dates'])[start::1]
    yvals_obs = np.array(dict_results8['Qobs'])[start::1]
    sub.plot( xvals, yvals_obs,
                      linewidth=0.0, marker='o', color='0.5',
                      markersize=msize/1.5, markeredgewidth=msize/6,markerfacecolor='white',
                      alpha=0.7, zorder=400)
    for itrial in [0,1,2,3,4,5,7,8,9]: #range(ntrials):  #

        yvals_sim = np.array(dict_results8[imetric]['Qsim']['trial_'+str(itrial+1)])[start::1]
        max_obs = np.max(yvals_obs)
        max_sim = np.max(yvals_sim)
        multi = max_obs/max_sim
        shift = 0.0

        sub.plot( [ str(datetime.datetime.fromisoformat(xx)+datetime.timedelta(days=6)) for xx in xvals ], yvals_sim * multi + shift,
                      linewidth=lwidth, linestyle='-', color=lcol1, alpha=0.7, zorder=40)

    # no ticks, no ticklabels
    sub.set_xticks([])
    sub.set_yticks([])
    sub.set_xlabel(str2tex('x',usetex=usetex))
    sub.set_ylabel(str2tex('f(x)',usetex=usetex))

    # abc
    sub.text( 1.05, 1.0, str2tex(chr(96+iiplot),usetex=usetex),
                                        ha = 'left', va = 'top',
                                        fontweight='bold',
                                        transform=sub.transAxes,
                                        fontsize=textsize+3 )

    if (outtype == 'pdf'):
        pdf_pages.savefig(fig, transparent=transparent)
        plt.close(fig)
    elif (outtype == 'png'):
        pngfile = pngbase+"{0:04d}".format(ifig)+".png"
        fig.savefig(pngfile, transparent=transparent, bbox_inches=bbox_inches, pad_inches=pad_inches)
        plt.close(fig)

    # -------------------------
    # plot (K) - compare sim vs obs: Only low values (Fig. 5d)
    # -------------------------
    ifig += 1
    iplot = 1
    iiplot += 1
    print('Plot - Fig ', chr(96+iiplot))
    fig = plt.figure(ifig)

    sub = fig.add_axes(position(nrow, ncol, iplot, hspace=hspace, vspace=vspace))

    # plot
    imetric = 'lnse'

    start = 70
    xvals = np.array(dict_results8['dates'])[start::1]
    yvals_obs = np.array(dict_results8['Qobs'])[start::1]
    sub.plot( xvals, yvals_obs,
                      linewidth=0.0, marker='o', color='0.5',
                      markersize=msize/1.5, markeredgewidth=msize/6,markerfacecolor='white',
                      alpha=0.7, zorder=400)
    for itrial in [0,1,2,3,4,5,7,8,9]: #range(ntrials):  #

        yvals_sim = np.array(dict_results8[imetric]['Qsim']['trial_'+str(itrial+1)])[start::1]
        sub.plot( [ str(datetime.datetime.fromisoformat(xx)+datetime.timedelta(days=6)) for xx in xvals ], yvals_sim,
                      linewidth=lwidth, linestyle='-', color=lcol1, alpha=0.7, zorder=40)

    # no ticks, no ticklabels
    sub.set_xticks([])
    sub.set_yticks([])
    sub.set_xlabel(str2tex('x',usetex=usetex))
    sub.set_ylabel(str2tex('f(x)',usetex=usetex))

    # abc
    sub.text( 1.05, 1.0, str2tex(chr(96+iiplot),usetex=usetex),
                                        ha = 'left', va = 'top',
                                        fontweight='bold',
                                        transform=sub.transAxes,
                                        fontsize=textsize+3 )

    if (outtype == 'pdf'):
        pdf_pages.savefig(fig, transparent=transparent)
        plt.close(fig)
    elif (outtype == 'png'):
        pngfile = pngbase+"{0:04d}".format(ifig)+".png"
        fig.savefig(pngfile, transparent=transparent, bbox_inches=bbox_inches, pad_inches=pad_inches)
        plt.close(fig)


    # -------------------------
    # plot (L) - compare sim vs obs: Only low values (Fig. 5f)
    # -------------------------
    ifig += 1
    iplot = 1
    iiplot += 1
    print('Plot - Fig ', chr(96+iiplot))
    fig = plt.figure(ifig)

    sub = fig.add_axes(position(nrow, ncol, iplot, hspace=hspace, vspace=vspace))

    # plot
    imetric = 'pbias'

    start = 70
    xvals = np.array(dict_results8['dates'])[start::1]
    yvals_obs = np.array(dict_results8['Qobs'])[start::1]
    sub.plot( xvals, yvals_obs,
                      linewidth=0.0, marker='o', color='0.5',
                      markersize=msize/1.5, markeredgewidth=msize/6,markerfacecolor='white',
                      alpha=0.7, zorder=400)
    for itrial in range(ntrials):  #

        yvals_sim = np.array(dict_results8[imetric]['Qsim']['trial_'+str(itrial+1)])[start::1]
        sub.plot( [ str(datetime.datetime.fromisoformat(xx)+datetime.timedelta(days=6)) for xx in xvals ], yvals_sim,
                      linewidth=lwidth, linestyle='-', color=lcol1, alpha=0.7, zorder=40)

    # no ticks, no ticklabels
    sub.set_xticks([])
    sub.set_yticks([])
    sub.set_xlabel(str2tex('x',usetex=usetex))
    sub.set_ylabel(str2tex('f(x)',usetex=usetex))

    # abc
    sub.text( 1.05, 1.0, str2tex(chr(96+iiplot),usetex=usetex),
                                        ha = 'left', va = 'top',
                                        fontweight='bold',
                                        transform=sub.transAxes,
                                        fontsize=textsize+3 )

    if (outtype == 'pdf'):
        pdf_pages.savefig(fig, transparent=transparent)
        plt.close(fig)
    elif (outtype == 'png'):
        pngfile = pngbase+"{0:04d}".format(ifig)+".png"
        fig.savefig(pngfile, transparent=transparent, bbox_inches=bbox_inches, pad_inches=pad_inches)
        plt.close(fig)


    # -------------------------
    # plot (M) - compare sim vs obs: poor overall (Fig. 4b)
    # -------------------------
    ifig += 1
    iplot = 1
    iiplot += 1
    print('Plot - Fig ', chr(96+iiplot))
    fig = plt.figure(ifig)

    sub = fig.add_axes(position(nrow, ncol, iplot, hspace=hspace, vspace=vspace))

    # plot
    for itrial in range(ntrials):

        para = dict_results7['para_logx_logy_narrow']['trial_'+str(itrial+1)]
        # transfform with loge
        xgridlog = np.arange( np.min(np.log(dict_results7['x'])),
                              np.max(np.log(dict_results7['x'])),
                              (np.max(np.log(dict_results7['x']))-np.min(np.log(dict_results7['x']))) / 1000. )
        # transform with log10
        xgridlog = np.arange( np.min(dict_results7['logx']),
                              np.max(dict_results7['logx']),
                              (np.max(dict_results7['logx'])-np.min(dict_results7['logx'])) / 1000. )

        # print("fit logx/logy: trial: {} para: {}".format(itrial+1,para))

        # f(x) = L/(1 + Exp[-k*(x - x0)]) - s
        # ymod  = p[0]    / ( 1.0 + np.exp( -p[1]    * ( logxvals - p[2]    ))) - p[3]
        ymodlog = para[0] / ( 1.0 + np.exp( -para[1] * ( xgridlog - para[2] ))) - para[3]

        sub.plot( xgridlog,ymodlog,
                  linewidth=lwidth, linestyle='-', color=lcol1, alpha=0.7, zorder=100)

    # data points
    sub.plot( dict_results7['logx'],dict_results7['logy'],
                  linewidth=0.0, marker='o', color='0.5',
                  markersize=msize/1, markeredgewidth=msize/4,markerfacecolor='w',
                  alpha=0.7, zorder=400)

    # no ticks, no ticklabels
    sub.set_xticks([])
    sub.set_yticks([])
    sub.set_xlabel(str2tex('x',usetex=usetex))
    sub.set_ylabel(str2tex('f(x)',usetex=usetex))

    # abc
    sub.text( 1.05, 1.0, str2tex(chr(96+iiplot),usetex=usetex),
                                        ha = 'left', va = 'top',
                                        fontweight='bold',
                                        transform=sub.transAxes,
                                        fontsize=textsize+3 )

    if (outtype == 'pdf'):
        pdf_pages.savefig(fig, transparent=transparent)
        plt.close(fig)
    elif (outtype == 'png'):
        pngfile = pngbase+"{0:04d}".format(ifig)+".png"
        fig.savefig(pngfile, transparent=transparent, bbox_inches=bbox_inches, pad_inches=pad_inches)
        plt.close(fig)


    # -------------------------
    # plot (N) - compare sim vs obs: good fit but wide spread (Fig. 4a)
    # -------------------------
    ifig += 1
    iplot = 1
    iiplot += 1
    print('Plot - Fig ', chr(96+iiplot))
    fig = plt.figure(ifig)

    sub = fig.add_axes(position(nrow, ncol, iplot, hspace=hspace, vspace=vspace))

    # plot
    for itrial in [1,2,3,4,5,6,7,8,9]: # range(ntrials):

        para = dict_results7['para_logx_logy_wide']['trial_'+str(itrial+1)]
        # transfform with loge
        xgridlog = np.arange( np.min(np.log(dict_results7['x'])),
                              np.max(np.log(dict_results7['x'])),
                              (np.max(np.log(dict_results7['x']))-np.min(np.log(dict_results7['x']))) / 1000. )
        # transform with log10
        xgridlog = np.arange( np.min(dict_results7['logx']),
                              np.max(dict_results7['logx']),
                              (np.max(dict_results7['logx'])-np.min(dict_results7['logx'])) / 1000. )

        # print("fit logx/logy: trial: {} para: {}".format(itrial+1,para))

        # f(x) = L/(1 + Exp[-k*(x - x0)]) - s
        # ymod  = p[0]    / ( 1.0 + np.exp( -p[1]    * ( logxvals - p[2]    ))) - p[3]
        ymodlog = para[0] / ( 1.0 + np.exp( -para[1] * ( xgridlog - para[2] ))) - para[3]

        sub.plot( xgridlog,ymodlog,
                  linewidth=lwidth, linestyle='-', color=lcol1, alpha=0.7, zorder=100)

    # data points
    sub.plot( dict_results7['logx'],dict_results7['logy'],
                  linewidth=0.0, marker='o', color='0.5',
                  markersize=msize/1, markeredgewidth=msize/4,markerfacecolor='w',
                  alpha=0.7, zorder=400)

    # no ticks, no ticklabels
    sub.set_xticks([])
    sub.set_yticks([])
    sub.set_xlabel(str2tex('x',usetex=usetex))
    sub.set_ylabel(str2tex('f(x)',usetex=usetex))

    # abc
    sub.text( 1.05, 1.0, str2tex(chr(96+iiplot),usetex=usetex),
                                        ha = 'left', va = 'top',
                                        fontweight='bold',
                                        transform=sub.transAxes,
                                        fontsize=textsize+3 )

    if (outtype == 'pdf'):
        pdf_pages.savefig(fig, transparent=transparent)
        plt.close(fig)
    elif (outtype == 'png'):
        pngfile = pngbase+"{0:04d}".format(ifig)+".png"
        fig.savefig(pngfile, transparent=transparent, bbox_inches=bbox_inches, pad_inches=pad_inches)
        plt.close(fig)


    # -------------------------
    # plot (O) - compare sim vs obs: looks good (Fig. 4c)
    # -------------------------
    ifig += 1
    iplot = 1
    iiplot += 1
    print('Plot - Fig ', chr(96+iiplot))
    fig = plt.figure(ifig)

    sub = fig.add_axes(position(nrow, ncol, iplot, hspace=hspace, vspace=vspace))

    # plot
    for itrial in [0,2,3,4,5,6,7,9]: #range(ntrials):  # [1,2,3,4,5,6,7,8,9]: #

        para = dict_results7['para_logx_logy_ori']['trial_'+str(itrial+1)]
        # transfform with loge
        xgridlog = np.arange( np.min(np.log(dict_results7['x'])),
                              np.max(np.log(dict_results7['x'])),
                              (np.max(np.log(dict_results7['x']))-np.min(np.log(dict_results7['x']))) / 1000. )
        # transform with log10
        xgridlog = np.arange( np.min(dict_results7['logx']),
                              np.max(dict_results7['logx']),
                              (np.max(dict_results7['logx'])-np.min(dict_results7['logx'])) / 1000. )

        # print("fit logx/logy: trial: {} para: {}".format(itrial+1,para))

        # f(x) = L/(1 + Exp[-k*(x - x0)]) - s
        # ymod  = p[0]    / ( 1.0 + np.exp( -p[1]    * ( logxvals - p[2]    ))) - p[3]
        ymodlog = para[0] / ( 1.0 + np.exp( -para[1] * ( xgridlog - para[2] ))) - para[3]

        sub.plot( xgridlog,ymodlog,
                  linewidth=lwidth, linestyle='-', color=lcol1, alpha=0.7, zorder=100)

    # data points
    sub.plot( dict_results7['logx'],dict_results7['logy'],
                  linewidth=0.0, marker='o', color='0.5',
                  markersize=msize/1, markeredgewidth=msize/4,markerfacecolor='w',
                  alpha=0.7, zorder=400)

    # no ticks, no ticklabels
    sub.set_xticks([])
    sub.set_yticks([])
    sub.set_xlabel(str2tex('x',usetex=usetex))
    sub.set_ylabel(str2tex('f(x)',usetex=usetex))

    # abc
    sub.text( 1.05, 1.0, str2tex(chr(96+iiplot),usetex=usetex),
                                        ha = 'left', va = 'top',
                                        fontweight='bold',
                                        transform=sub.transAxes,
                                        fontsize=textsize+3 )

    if (outtype == 'pdf'):
        pdf_pages.savefig(fig, transparent=transparent)
        plt.close(fig)
    elif (outtype == 'png'):
        pngfile = pngbase+"{0:04d}".format(ifig)+".png"
        fig.savefig(pngfile, transparent=transparent, bbox_inches=bbox_inches, pad_inches=pad_inches)
        plt.close(fig)



    # # -------------------------
    # # plot (F) - objective function values: SO: looks good (Fig. 8f large budget)
    # # -------------------------
    # ifig += 1
    # iplot = 1
    # iiplot += 1
    # print('Plot - Fig ', chr(96+iiplot))
    # fig = plt.figure(ifig)

    # sub = fig.add_axes(position(nrow, ncol, iplot, hspace=hspace, vspace=vspace))

    # # no ticks, no ticklabels
    # sub.set_xticks([])
    # sub.set_yticks([])
    # sub.set_xlabel(str2tex('???',usetex=usetex))

    # # abc
    # sub.text( 1.05, 1.0, str2tex(chr(96+iiplot),usetex=usetex),
    #                                     ha = 'left', va = 'top',
    #                                     fontweight='bold',
    #                                     transform=sub.transAxes,
    #                                     fontsize=textsize+3 )

    # if (outtype == 'pdf'):
    #     pdf_pages.savefig(fig, transparent=transparent)
    #     plt.close(fig)
    # elif (outtype == 'png'):
    #     pngfile = pngbase+"{0:04d}".format(ifig)+".png"
    #     fig.savefig(pngfile, transparent=transparent, bbox_inches=bbox_inches, pad_inches=pad_inches)
    #     plt.close(fig)





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
