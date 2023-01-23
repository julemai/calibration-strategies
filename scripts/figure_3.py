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
#     run figure_3.py -t pdf -p figure_3

#!/usr/bin/env python
from __future__ import print_function

"""

Plots sampling of tied parameters and parameters with constraints

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
                                      description='''Plots sampling of tied parameters and parameters with constraints.''')
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

    jsonfile = 'figure_3.json'

    # -------------------------------------------------------------------------
    # general
    # -------------------------------------------------------------------------

    nsets = 10000
    npara = 2

    # -------------------------------------------------------------------------
    # a < b
    # -------------------------------------------------------------------------

    aL = 2
    aU = 6
    bL = 2
    bU = 8
    ranges_a_le_b = np.array([ [aL, aU], [bL, bU] ])

    # -------------------------------------------------------------------------
    # a + b < c
    # -------------------------------------------------------------------------

    cthres = 8.0
    ranges_a_plus_b_le_c = np.array([ [0, cthres], [0, cthres] ])

    if not(os.path.exists(jsonfile)):

        dict_results = {}

        # -------------------------------------------------------------------------
        # a < b
        # -------------------------------------------------------------------------

        # naiive
        sample_naiive_a_le_b_a = np.random.random([nsets,npara]) * (ranges_a_le_b[:,1]-ranges_a_le_b[:,0]) + ranges_a_le_b[:,0]

        # alternative
        delta = bU-aL #bU-aU
        sample_altern_a_le_b_a = np.random.random([nsets,npara]) * (np.array([[aU,delta]])-np.array([[aL,0.0]])) + np.array([[aL,0.0]])
        sample_altern_a_le_b_a[:,1] = copy.deepcopy(sample_altern_a_le_b_a[:,0] + sample_altern_a_le_b_a[:,1])

        dict_results_tmp = {'naiive': sample_naiive_a_le_b_a, 'alternative': sample_altern_a_le_b_a}
        dict_results['a_le_b'] = dict_results_tmp

        # -------------------------------------------------------------------------
        # a + b < c
        # -------------------------------------------------------------------------

        # naiive
        sample_naiive_a_plus_b_le_c = np.random.random([nsets,npara]) * (ranges_a_plus_b_le_c[:,1]-ranges_a_plus_b_le_c[:,0]) + ranges_a_plus_b_le_c[:,0]

        # alternative
        rr = np.random.random([nsets,npara])
        sample_altern_a_plus_b_le_c = np.ones([nsets,npara]) * -9999.
        sample_altern_a_plus_b_le_c[:,0] = cthres*(1.0-(1.0-rr[:,0])**0.5)
        sample_altern_a_plus_b_le_c[:,1] = cthres*(1.0-rr[:,0])**0.5*rr[:,1]

        dict_results_tmp = {'naiive': sample_naiive_a_plus_b_le_c, 'alternative': sample_altern_a_plus_b_le_c}
        dict_results['a_plus_b_le_c'] = dict_results_tmp




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
    ncol        = 4           # number columns
    nrow        = 5           # number of rows
    textsize    = 8          # standard text size
    dxabc       = 0.95          # % of (max-min) shift to the right from left y-axis for a,b,c,... labels
    dyabc       = 0.92          # % of (max-min) shift up from lower x-axis for a,b,c,... labels
    dxsig       = 1.23        # % of (max-min) shift to the right from left y-axis for signature
    dysig       = -0.075      # % of (max-min) shift up from lower x-axis for signature
    dxtit       = 0           # % of (max-min) shift to the right from left y-axis for title
    dytit       = 1.2         # % of (max-min) shift up from lower x-axis for title
    hspace      = 0.12        # x-space between subplots
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
    llxbbox     = 0.98        # x-anchor legend bounding box
    llybbox     = 0.98        # y-anchor legend bounding box
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

    # -------------
    # Samples points
    # -------------

    xlim = ranges_a_le_b[0,:]
    ylim = ranges_a_le_b[1,:]

    # -------------------------
    # scatter plot - a < b (naiive)
    # -------------------------
    iplot += 1
    sub = fig.add_axes(position(nrow, ncol, iplot, hspace=hspace/2, vspace=vspace))

    idx = np.where(np.array(dict_results['a_le_b']['naiive'])[:,0] < np.array(dict_results['a_le_b']['naiive'])[:,1])[0]
    xvals = list(np.array(dict_results['a_le_b']['naiive'])[idx,0])
    yvals = list(np.array(dict_results['a_le_b']['naiive'])[idx,1])
    dx = 0.2
    sub.hist2d(xvals, yvals, bins=[int((np.max(xvals)-np.min(xvals))/(dx*0.5)),int((np.max(yvals)-np.min(yvals))/dx)], norm=norm1, cmap=cmap)

    idx = np.where(np.array(dict_results['a_le_b']['naiive'])[:,0] > np.array(dict_results['a_le_b']['naiive'])[:,1])[0]
    xvals = list(np.array(dict_results['a_le_b']['naiive'])[idx,0])
    yvals = list(np.array(dict_results['a_le_b']['naiive'])[idx,1])
    sub.hist2d(xvals, yvals, bins=[int((np.max(xvals)-np.min(xvals))/(dx*0.5)),int((np.max(yvals)-np.min(yvals))/dx)], norm=norm1, cmap=cmap_gray)

    sub.text(0.5,1.06, 'Accept-Reject Method', #'Na\u00efve approach',
                 fontsize=textsize+2,
                 transform=sub.transAxes, horizontalalignment='center', verticalalignment='bottom')
    sub.text(-0.5,0.5, str2tex('$x_1 < x_2$', usetex=usetex),
                 fontsize=textsize+2,
                 transform=sub.transAxes, horizontalalignment='center', verticalalignment='center', rotation=90)

    #sub.set_xlabel(str2tex('Sampled $x_{1}$', usetex=usetex))
    sub.set_ylabel(str2tex('Sampled $x_{2}$', usetex=usetex))

    dd=0.1
    sub.set_xlim([xlim[0]-(xlim[1]-xlim[0])*(dd),xlim[1]+(xlim[1]-xlim[0])*(dd)])
    sub.set_ylim([ylim[0]-(ylim[1]-ylim[0])*(dd),ylim[1]+(ylim[1]-ylim[0])*(dd*3)])

    xticks = [ranges_a_le_b[0,0],ranges_a_le_b[0,0]+(ranges_a_le_b[0,1]-ranges_a_le_b[0,0])/2.,ranges_a_le_b[0,1]]
    yticks = [ranges_a_le_b[1,0],ranges_a_le_b[1,0]+(ranges_a_le_b[1,1]-ranges_a_le_b[1,0])/2.,ranges_a_le_b[1,1]]
    sub.set_xticks(xticks)
    sub.set_yticks(yticks)
    #sub.set_xticklabels([str2tex('$x_{1,L}='+astr(xticks[0],prec=0)+'$',usetex=usetex),astr(xticks[1],prec=0),'$x_{1,U}='+astr(xticks[2],prec=0)+'$'])
    #sub.set_yticklabels([str2tex('$x_{2,L}='+astr(yticks[0],prec=0)+'$',usetex=usetex),astr(yticks[1],prec=0),'$x_{2,U}='+astr(yticks[2],prec=0)+'$'])
    sub.set_xticklabels([str2tex(astr(xticks[0],prec=0),usetex=usetex),astr(xticks[1],prec=0),astr(xticks[2],prec=0)])
    sub.set_yticklabels([str2tex(astr(yticks[0],prec=0),usetex=usetex),astr(yticks[1],prec=0),astr(yticks[2],prec=0)])

    # abc
    sub.text( 1.05, 1.0, str2tex(chr(96+iplot),usetex=usetex),
                                ha = 'left', va = 'top',
                                fontweight='bold',
                                transform=sub.transAxes,
                                fontsize=textsize+3 )

    # -------------------------
    # scatter plot - a < b (alternative)
    # -------------------------
    iplot += 1
    sub = fig.add_axes(position(nrow, ncol, iplot, hspace=hspace/2, vspace=vspace))

    idx = np.where(np.array(dict_results['a_le_b']['alternative'])[:,1] < ylim[1])[0]
    xvals = list(np.array(dict_results['a_le_b']['alternative'])[idx,0])
    yvals = list(np.array(dict_results['a_le_b']['alternative'])[idx,1])
    sub.hist2d(xvals, yvals, bins=[int((np.max(xvals)-np.min(xvals))/(dx*0.5)),int((np.max(yvals)-np.min(yvals))/dx)], norm=norm1, cmap=cmap)

    idx = np.where(np.array(dict_results['a_le_b']['alternative'])[:,1] > ylim[1])[0]
    xvals = list(np.array(dict_results['a_le_b']['alternative'])[idx,0])
    yvals = list(np.array(dict_results['a_le_b']['alternative'])[idx,1])
    sub.hist2d(xvals, yvals, bins=[int((np.max(xvals)-np.min(xvals))/(dx*0.5)),int((np.max(yvals)-np.min(yvals))/dx)], norm=norm1, cmap=cmap_gray)

    sub.text(0.5,1.06, str2tex('Delta Method', usetex=usetex),
                 fontsize=textsize+2,
                 transform=sub.transAxes, horizontalalignment='center', verticalalignment='bottom')

    #sub.set_xlabel(str2tex('Sampled $x_{1}$', usetex=usetex))

    dd=0.1
    sub.set_xlim([xlim[0]-(xlim[1]-xlim[0])*(dd),xlim[1]+(xlim[1]-xlim[0])*(dd)])
    sub.set_ylim([ylim[0]-(ylim[1]-ylim[0])*(dd),ylim[1]+(ylim[1]-ylim[0])*(dd*3)])

    xticks = [ranges_a_le_b[0,0],ranges_a_le_b[0,0]+(ranges_a_le_b[0,1]-ranges_a_le_b[0,0])/2.,ranges_a_le_b[0,1]]
    yticks = [ranges_a_le_b[1,0],ranges_a_le_b[1,0]+(ranges_a_le_b[1,1]-ranges_a_le_b[1,0])/2.,ranges_a_le_b[1,1]]
    sub.set_xticks(xticks)
    sub.set_yticks(yticks)
    #sub.set_xticklabels([str2tex('$x_{1,L}='+astr(xticks[0],prec=0)+'$',usetex=usetex),astr(xticks[1],prec=0),'$x_{1,U}='+astr(xticks[2],prec=0)+'$'])
    #sub.set_yticklabels([str2tex('$x_{2,L}='+astr(yticks[0],prec=0)+'$',usetex=usetex),astr(yticks[1],prec=0),'$x_{2,U}='+astr(yticks[2],prec=0)+'$'])
    sub.set_xticklabels([str2tex(astr(xticks[0],prec=0),usetex=usetex),astr(xticks[1],prec=0),astr(xticks[2],prec=0)])

    sub.set_yticklabels([])

    # abc
    sub.text( 1.05, 1.0, str2tex(chr(96+iplot),usetex=usetex),
                                ha = 'left', va = 'top',
                                fontweight='bold',
                                transform=sub.transAxes,
                                fontsize=textsize+3 )


    # fill row
    iplot += 1

    # colorbar -gray
    # [left, bottom, width, height]
    pos_cbar = position(nrow, ncol, iplot, hspace=hspace/2, vspace=vspace)
    pos_cbar *= [1.0,1.0,0.1,1.0]   # [0.74375 0.444   0.15625 0.112  ]
    #print("pos cbar: ",pos_cbar)
    csub    = fig.add_axes( pos_cbar )

    if inorm == 'log':
        ticks = [ 10.0**(np.log10(min_sti) + ii*(np.log10(max_sti) - np.log10(min_sti))/len(cc)) for ii in range(len(cc)+1) ]
        cbar = mpl.colorbar.ColorbarBase(csub, norm=norm, cmap=cmap, orientation='vertical', extend='min')
        cbar.set_ticklabels([ "{:.1e}".format(itick) if (iitick%5 ==0) else "" for iitick,itick in enumerate(ticks) ])  # print only every fifth label
        cbar.set_label(str2tex("Density [-]",usetex=usetex))
    elif inorm == 'pow':
        ticks = [0]+[ 10**ii for ii in np.arange(0,5,1) ]  # [0, 1, 10, 100, 1000, 10000]
        ticks = [0, 1, 10, 100, 300, 1000, 3000, 10000]
        cbar = mpl.colorbar.ColorbarBase(csub, norm=norm1, ticks=ticks, cmap=cmap_gray, orientation='vertical', extend='max')  #
        cbar.set_ticklabels([ " ".format(itick) for itick in ticks ])
        cbar.set_label(str2tex(" ",usetex=usetex))
    elif inorm == 'linear':
        ticks =  [0, 2,5,10,20,50]
        cbar = mpl.colorbar.ColorbarBase(csub, norm=mpl.colors.NoNorm(), ticks=ticks, cmap=cmap_gray, orientation='vertical', extend='max')  #
        cbar.set_ticklabels([ "".format(itick) for itick in ticks ])
        cbar.set_label(str2tex(" ",usetex=usetex))
    else:
        raise ValueError('Norm for colormap not known.')

    csub.text( -0.1, 0.0, str2tex('Invalid samples', usetex=usetex),
                  transform=csub.transAxes,
                  fontsize=textsize-2,
                  color='0.7',
                  rotation=90,
                  horizontalalignment='right', verticalalignment='bottom')

    # colorbar -color
    # [left, bottom, width, height]
    pos_cbar = position(nrow, ncol, iplot, hspace=hspace/2, vspace=vspace)
    pos_cbar *= [1.07,1.0,0.1,1.0]   # [0.74375 0.444   0.15625 0.112  ]
    #print("pos cbar: ",pos_cbar)
    csub    = fig.add_axes( pos_cbar )

    if inorm == 'log':
        ticks = [ 10.0**(np.log10(min_sti) + ii*(np.log10(max_sti) - np.log10(min_sti))/len(cc)) for ii in range(len(cc)+1) ]
        cbar = mpl.colorbar.ColorbarBase(csub, norm=norm, cmap=cmap, orientation='vertical', extend='min')
        cbar.set_ticklabels([ "{:.1e}".format(itick) if (iitick%5 ==0) else "" for iitick,itick in enumerate(ticks) ])  # print only every fifth label
        cbar.set_label(str2tex("Density [-]",usetex=usetex))
    elif inorm == 'pow':
        ticks = [0]+[ 10**ii for ii in np.arange(0,5,1) ]  # [0, 1, 10, 100, 1000, 10000]
        ticks = [0, 1, 10, 100, 300, 1000, 3000, 10000]
        cbar = mpl.colorbar.ColorbarBase(csub, norm=norm1, ticks=ticks, cmap=cmap, orientation='vertical', extend='max')  #
        cbar.set_ticklabels([ "{:.0e}".format(itick) for itick in ticks ])
        cbar.set_label(str2tex("Frequency [-]",usetex=usetex))
    elif inorm == 'linear':
        ticks = [0, 2,5,10,20,50]
        cbar = mpl.colorbar.ColorbarBase(csub, norm=mpl.colors.NoNorm(), ticks=ticks, cmap=cmap, orientation='vertical', extend='max')  #
        cbar.set_ticklabels([ "{:.0e}".format(itick) for itick in ticks ])
        cbar.set_label(str2tex("Frequency [-]",usetex=usetex))
    else:
        raise ValueError('Norm for colormap not known.')

    csub.text( -0.1, 0.0, str2tex('Valid samples', usetex=usetex),
                  transform=csub.transAxes,
                  fontsize=textsize-2,
                  color='0.7',
                  rotation=90,
                  horizontalalignment='right', verticalalignment='bottom')

    # color bins
    #for ibin in range(cmap.N):
    #    print("Color bin #",ibin+1,"  :: [",((max_pow1**pow_lambda1)/cmap.N*(ibin))**(1./pow_lambda1),',',((max_pow1**pow_lambda1)/cmap.N*(ibin+1))**(1./pow_lambda1),']')

    # fill row
    iplot += 1





    xlim = ranges_a_plus_b_le_c[0,:]
    ylim = ranges_a_plus_b_le_c[1,:]

    # -------------------------
    # scatter plot - a + b < c (naiive)
    # -------------------------
    iplot += 1
    sub = fig.add_axes(position(nrow, ncol, iplot, hspace=hspace/2, vspace=vspace))

    idx = np.where(np.array(dict_results['a_plus_b_le_c']['naiive'])[:,0] + np.array(dict_results['a_plus_b_le_c']['naiive'])[:,1] < cthres)[0]
    xvals = list(np.array(dict_results['a_plus_b_le_c']['naiive'])[idx,0])
    yvals = list(np.array(dict_results['a_plus_b_le_c']['naiive'])[idx,1])
    dx = 0.2
    sub.hist2d(xvals, yvals, bins=[int((np.max(xvals)-np.min(xvals))/dx),int((np.max(yvals)-np.min(yvals))/dx)], norm=norm1, cmap=cmap)

    idx = np.where(np.array(dict_results['a_plus_b_le_c']['naiive'])[:,0] + np.array(dict_results['a_plus_b_le_c']['naiive'])[:,1] > cthres)[0]
    if (len(idx) > 0) :
        xvals = list(np.array(dict_results['a_plus_b_le_c']['naiive'])[idx,0])
        yvals = list(np.array(dict_results['a_plus_b_le_c']['naiive'])[idx,1])
        sub.hist2d(xvals, yvals, bins=[int((np.max(xvals)-np.min(xvals))/dx),int((np.max(yvals)-np.min(yvals))/dx)], norm=norm1, cmap=cmap_gray)

    sub.text(0.5,1.06, 'Accept-Reject Method', #'Na\u00efve approach',
                 fontsize=textsize+2,
                 transform=sub.transAxes, horizontalalignment='center', verticalalignment='bottom')
    sub.text(-0.5,0.5, str2tex('$x_1 + x_2 < c$', usetex=usetex),
                 fontsize=textsize+2,
                 transform=sub.transAxes, horizontalalignment='center', verticalalignment='center', rotation=90)

    sub.set_xlabel(str2tex('Sampled $x_{1}$', usetex=usetex))
    sub.set_ylabel(str2tex('Sampled $x_{2}$', usetex=usetex))

    dd=0.1
    sub.set_xlim([xlim[0]-(xlim[1]-xlim[0])*(dd),xlim[1]+(xlim[1]-xlim[0])*(dd)])
    sub.set_ylim([ylim[0]-(ylim[1]-ylim[0])*(dd),ylim[1]+(ylim[1]-ylim[0])*(dd)])

    xticks = [ranges_a_plus_b_le_c[0,0],ranges_a_plus_b_le_c[0,0]+(ranges_a_plus_b_le_c[0,1]-ranges_a_plus_b_le_c[0,0])/2.,ranges_a_plus_b_le_c[0,1]]
    yticks = [ranges_a_plus_b_le_c[1,0],ranges_a_plus_b_le_c[1,0]+(ranges_a_plus_b_le_c[1,1]-ranges_a_plus_b_le_c[1,0])/2.,ranges_a_plus_b_le_c[1,1]]
    sub.set_xticks(xticks)
    sub.set_yticks(yticks)
    #sub.set_xticklabels([str2tex('$x_{1,L}='+astr(xticks[0],prec=0)+'$',usetex=usetex),astr(xticks[1],prec=0),'$x_{1,U}='+astr(xticks[2],prec=0)+'$'])
    #sub.set_yticklabels([str2tex('$x_{2,L}='+astr(yticks[0],prec=0)+'$',usetex=usetex),astr(yticks[1],prec=0),'$x_{2,U}='+astr(yticks[2],prec=0)+'$'])
    sub.set_xticklabels([str2tex(astr(xticks[0],prec=0),usetex=usetex),astr(xticks[1],prec=0),astr(xticks[2],prec=0)])
    sub.set_yticklabels([str2tex(astr(yticks[0],prec=0),usetex=usetex),astr(yticks[1],prec=0),astr(yticks[2],prec=0)])

    # abc
    sub.text( 1.05, 1.0, str2tex(chr(96+iplot-2),usetex=usetex),
                                ha = 'left', va = 'top',
                                fontweight='bold',
                                transform=sub.transAxes,
                                fontsize=textsize+3 )



    # -------------------------
    # scatter plot - a + b < c (alternative)
    # -------------------------
    iplot += 1
    sub = fig.add_axes(position(nrow, ncol, iplot, hspace=hspace/2, vspace=vspace))

    idx = np.where(np.array(dict_results['a_plus_b_le_c']['alternative'])[:,0] + np.array(dict_results['a_plus_b_le_c']['alternative'])[:,1] < cthres)[0]
    xvals = list(np.array(dict_results['a_plus_b_le_c']['alternative'])[idx,0])
    yvals = list(np.array(dict_results['a_plus_b_le_c']['alternative'])[idx,1])
    dx = 0.2
    sub.hist2d(xvals, yvals, bins=[int((np.max(xvals)-np.min(xvals))/dx),int((np.max(yvals)-np.min(yvals))/dx)], norm=norm1, cmap=cmap)

    idx = np.where(np.array(dict_results['a_plus_b_le_c']['alternative'])[:,0] + np.array(dict_results['a_plus_b_le_c']['alternative'])[:,1] > cthres)[0]
    if (len(idx) > 0) :
        xvals = list(np.array(dict_results['a_plus_b_le_c']['alternative'])[idx,0])
        yvals = list(np.array(dict_results['a_plus_b_le_c']['alternative'])[idx,1])
        sub.hist2d(xvals, yvals, bins=[int((np.max(xvals)-np.min(xvals))/dx),int((np.max(yvals)-np.min(yvals))/dx)], norm=norm1, cmap=cmap_gray)

    sub.text(0.5,1.06, 'Pie-Share Method',
                 fontsize=textsize+2,
                 transform=sub.transAxes, horizontalalignment='center', verticalalignment='bottom')

    sub.set_xlabel(str2tex('Sampled $x_{1}$', usetex=usetex))

    dd=0.1
    sub.set_xlim([xlim[0]-(xlim[1]-xlim[0])*(dd),xlim[1]+(xlim[1]-xlim[0])*(dd)])
    sub.set_ylim([ylim[0]-(ylim[1]-ylim[0])*(dd),ylim[1]+(ylim[1]-ylim[0])*(dd)])

    xticks = [ranges_a_plus_b_le_c[0,0],ranges_a_plus_b_le_c[0,0]+(ranges_a_plus_b_le_c[0,1]-ranges_a_plus_b_le_c[0,0])/2.,ranges_a_plus_b_le_c[0,1]]
    yticks = [ranges_a_plus_b_le_c[1,0],ranges_a_plus_b_le_c[1,0]+(ranges_a_plus_b_le_c[1,1]-ranges_a_plus_b_le_c[1,0])/2.,ranges_a_plus_b_le_c[1,1]]
    sub.set_xticks(xticks)
    sub.set_yticks(yticks)
    #sub.set_xticklabels([str2tex('$x_{1,L}='+astr(xticks[0],prec=0)+'$',usetex=usetex),astr(xticks[1],prec=0),'$x_{1,U}='+astr(xticks[2],prec=0)+'$'])
    #sub.set_yticklabels([str2tex('$x_{2,L}='+astr(yticks[0],prec=0)+'$',usetex=usetex),astr(yticks[1],prec=0),'$x_{2,U}='+astr(yticks[2],prec=0)+'$'])
    sub.set_xticklabels([str2tex(astr(xticks[0],prec=0),usetex=usetex),astr(xticks[1],prec=0),astr(xticks[2],prec=0)])
    sub.set_yticklabels([str2tex(astr(yticks[0],prec=0),usetex=usetex),astr(yticks[1],prec=0),astr(yticks[2],prec=0)])

    # abc
    sub.text( 1.05, 1.0, str2tex(chr(96+iplot-2),usetex=usetex),
                                ha = 'left', va = 'top',
                                fontweight='bold',
                                transform=sub.transAxes,
                                fontsize=textsize+3 )

    # fill row
    iplot += 1

    # colorbar -gray
    # [left, bottom, width, height]
    pos_cbar = position(nrow, ncol, iplot, hspace=hspace/2, vspace=vspace)
    pos_cbar *= [1.0,1.0,0.1,1.0]   # [0.74375 0.444   0.15625 0.112  ]
    #print("pos cbar: ",pos_cbar)
    csub    = fig.add_axes( pos_cbar )

    if inorm == 'log':
        ticks = [ 10.0**(np.log10(min_sti) + ii*(np.log10(max_sti) - np.log10(min_sti))/len(cc)) for ii in range(len(cc)+1) ]
        cbar = mpl.colorbar.ColorbarBase(csub, norm=norm, cmap=cmap, orientation='vertical', extend='min')
        cbar.set_ticklabels([ "{:.1e}".format(itick) if (iitick%5 ==0) else "" for iitick,itick in enumerate(ticks) ])  # print only every fifth label
        cbar.set_label(str2tex("Density [-]",usetex=usetex))
    elif inorm == 'pow':
        ticks = [0]+[ 10**ii for ii in np.arange(0,5,1) ]  # [0, 1, 10, 100, 1000, 10000]
        ticks = [0, 1, 10, 100, 300, 1000, 3000, 10000]
        cbar = mpl.colorbar.ColorbarBase(csub, norm=norm1, ticks=ticks, cmap=cmap_gray, orientation='vertical', extend='max')  #
        cbar.set_ticklabels([ " ".format(itick) for itick in ticks ])
        cbar.set_label(str2tex(" ",usetex=usetex))
    elif inorm == 'linear':
        ticks =  [0, 2,5,10,20,50]
        cbar = mpl.colorbar.ColorbarBase(csub, norm=mpl.colors.NoNorm(), ticks=ticks, cmap=cmap_gray, orientation='vertical', extend='max')  #
        cbar.set_ticklabels([ "".format(itick) for itick in ticks ])
        cbar.set_label(str2tex(" ",usetex=usetex))
    else:
        raise ValueError('Norm for colormap not known.')

    csub.text( -0.1, 0.0, str2tex('Invalid samples', usetex=usetex),
                  transform=csub.transAxes,
                  fontsize=textsize-2,
                  color='0.7',
                  rotation=90,
                  horizontalalignment='right', verticalalignment='bottom')

    # colorbar -color
    # [left, bottom, width, height]
    pos_cbar = position(nrow, ncol, iplot, hspace=hspace/2, vspace=vspace)
    pos_cbar *= [1.07,1.0,0.1,1.0]   # [0.74375 0.444   0.15625 0.112  ]
    #print("pos cbar: ",pos_cbar)
    csub    = fig.add_axes( pos_cbar )

    if inorm == 'log':
        ticks = [ 10.0**(np.log10(min_sti) + ii*(np.log10(max_sti) - np.log10(min_sti))/len(cc)) for ii in range(len(cc)+1) ]
        cbar = mpl.colorbar.ColorbarBase(csub, norm=norm, cmap=cmap, orientation='vertical', extend='min')
        cbar.set_ticklabels([ "{:.1e}".format(itick) if (iitick%5 ==0) else "" for iitick,itick in enumerate(ticks) ])  # print only every fifth label
        cbar.set_label(str2tex("Density [-]",usetex=usetex))
    elif inorm == 'pow':
        ticks = [0]+[ 10**ii for ii in np.arange(0,5,1) ]  # [0, 1, 10, 100, 1000, 10000]
        ticks = [0, 1, 10, 100, 300, 1000, 3000, 10000]
        cbar = mpl.colorbar.ColorbarBase(csub, norm=norm1, ticks=ticks, cmap=cmap, orientation='vertical', extend='max')  #
        cbar.set_ticklabels([ "{:.0e}".format(itick) for itick in ticks ])
        cbar.set_label(str2tex("Frequency [-]",usetex=usetex))
    elif inorm == 'linear':
        ticks = [0, 2,5,10,20,50]
        cbar = mpl.colorbar.ColorbarBase(csub, norm=mpl.colors.NoNorm(), ticks=ticks, cmap=cmap, orientation='vertical', extend='max')  #
        cbar.set_ticklabels([ "{:.0e}".format(itick) for itick in ticks ])
        cbar.set_label(str2tex("Frequency [-]",usetex=usetex))
    else:
        raise ValueError('Norm for colormap not known.')

    csub.text( -0.1, 0.0, str2tex('Valid samples', usetex=usetex),
                  transform=csub.transAxes,
                  fontsize=textsize-2,
                  color='0.7',
                  rotation=90,
                  horizontalalignment='right', verticalalignment='bottom')

    # color bins
    #for ibin in range(cmap.N):
    #    print("Color bin #",ibin+1,"  :: [",((max_pow1**pow_lambda1)/cmap.N*(ibin))**(1./pow_lambda1),',',((max_pow1**pow_lambda1)/cmap.N*(ibin+1))**(1./pow_lambda1),']')

    # fill row
    iplot += 1


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
