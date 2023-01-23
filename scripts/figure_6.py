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
#     run figure_6.py -t pdf -p figure_6

#!/usr/bin/env python
from __future__ import print_function

"""

Plots results of random sampling vs stratified sampling vs calibration

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
    ntrials     = 10

    parser   = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                      description='''Plots results of random sampling vs stratified sampling vs calibration.''')
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
    from   scipy import stats
    import lhsmdu   # Latin-Hypercube sampling
    import json as json

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

    # colors
    cols1 = color.get_brewer('YlOrRd9', rgb=True)
    cols1 = color.get_brewer( 'WhiteYellowOrangeRed',rgb=True)[30:]
    cols1 = color.get_brewer( 'dark_rainbow_256',rgb=True)   # blue to red

    cols2 = color.get_brewer('YlOrRd9', rgb=True)[::-1]
    cols2 = color.get_brewer( 'WhiteYellowOrangeRed',rgb=True)[30:][::-1]
    cols2 = color.get_brewer( 'dark_rainbow_256',rgb=True)[::-1]  # red to blue

    cols3 = [cols2[0],cols2[95],cols2[-1]]  # red, yellow, blue
    cols3 = [color.colours('gray'),cols2[0],color.colours('white')]  # gray red white

    def objective_function(p):

        # two-dimensional Gaussian function as described on Wikipedia using parameters described there
        # https://en.wikipedia.org/wiki/Gaussian_function

        # optimum at p1=x=0.0 and p2=y=0.0 is A=1.0
        A = 1.0
        a = 0.5
        b = 0.0
        c = 0.125

        x0 = 0.0
        y0 = 0.0

        func = A * np.exp(-(a*(p[0] - x0)**2 + 2*b*(p[0] - x0)*(p[1] - y0) + c*(p[1] - y0)**2))

        return func

    def ackley(x):
        """
        Ackley function (>= 2-D).

        Global Optimum: 0.0, at origin.

        Parameters
        ----------
        x : array
            multi-dimensional x-values (len(x) >= 2)

        Returns
        -------
        float
           Value of Ackley function.
        """
        a = 20.0
        b = 0.2
        c = 2.0*np.pi

        n  = np.size(x)
        s1 = np.sum(x**2)
        s2 = np.sum(np.cos(c*x))
        f  = -a * np.exp(-b*np.sqrt(1.0/n*s1)) - np.exp(1.0/n*s2) + a + np.exp(1.0)

        return f

    # fix seed for reproducible results
    np.random.seed(seed=123)

    jsonfile = dir_path+'/figure_6.json'

    if not(os.path.exists(jsonfile)):

        dict_results = {}

        for npara in [2,10]:

            # range of parameters
            ranges = np.array([ [-2.0, 2.0 ] for ipara in range(npara) ])

            nsets = npara*100
            dict_result_para = {}

            for itrial in range(ntrials):

                dict_results_trial = {}

                # -------------------------------------------------------------------------
                # Sample random
                # -------------------------------------------------------------------------
                print("Perform random sampling (npara="+str(npara)+", trial="+str(itrial+1)+") ... ")
                parasets_random = np.random.random([nsets,npara]) * (ranges[:,1]-ranges[:,0]) + ranges[:,0]
                objective_random = [ ackley(pp) for pp in parasets_random ]

                dict_results_trial['parasets_random']  = list( [ list(pp) for pp in parasets_random ] )
                dict_results_trial['objective_random'] = list(objective_random)

                # -------------------------------------------------------------------------
                # Sample stratified (LHS)
                # -------------------------------------------------------------------------
                print("Perform LHS sampling (npara="+str(npara)+", trial="+str(itrial+1)+") ... ")
                parasets_lhs = lhsmdu.sample(npara,nsets)
                parasets_lhs = np.transpose(np.array(parasets_lhs)) * (ranges[:,1]-ranges[:,0]) + ranges[:,0]
                objective_lhs = [ ackley(pp) for pp in parasets_lhs ]

                dict_results_trial['parasets_lhs']  = list( [ list(pp) for pp in parasets_lhs ] )
                dict_results_trial['objective_lhs'] = list(objective_lhs)

                # -------------------------------------------------------------------------
                # Use DDS
                # -------------------------------------------------------------------------
                print("Perform DDS sampling (read actually only results) (npara="+str(npara)+", trial="+str(itrial+1)+") ... ")

                # run DDS via Ostrich beforehand
                #   cd example_6/calibrate_with_dds
                #   ./run_all.sh

                # read Ostrich results
                # Header:
                #         Run   obj.function   par_p01   par_p02   par_p03   par_p04   par_r01   par_r02
                infile = 'example_6/calibrate_with_dds_n='+str(npara)+'/trial_'+str(itrial+1)+'/OstModel0.txt'
                ff = open(infile,'r')
                content = ff.readlines()
                ff.close()

                results_str   = [ [ float(ii) for ii in cc.strip().split() ] for cc in content[1:] ]
                results_float = np.array([ [ float(ii) for ii in cc.strip().split() ] for cc in content[1:] ])

                parasets_dds = results_float[:,2:2+npara]
                objective_dds = results_float[:,1] #* -1.0   # results contain the objective function value that was MINIMIZED

                dict_results_trial['parasets_dds']  = list( [ list(pp) for pp in parasets_dds ] )
                dict_results_trial['objective_dds'] = list(objective_dds)

                # -------------------------------------------------------------------------
                # Use SCE
                # -------------------------------------------------------------------------
                print("Perform SCE sampling (read actually only results) (npara="+str(npara)+", trial="+str(itrial+1)+") ... ")

                # run SCE via Ostrich beforehand
                #   cd example_6/calibrate_with_sce
                #   ./run_all.sh

                # read Ostrich results
                # Header:
                #         Run   obj.function   par_p01   par_p02   par_p03   par_p04   par_r01   par_r02
                infile = 'example_6/calibrate_with_sce_n='+str(npara)+'/trial_'+str(itrial+1)+'/OstModel0.txt'
                ff = open(infile,'r')
                content = ff.readlines()
                ff.close()

                results_str   = [ [ float(ii) for ii in cc.strip().split() ] for cc in content[1:] ]
                results_float = np.array([ [ float(ii) for ii in cc.strip().split() ] for cc in content[1:] ])

                parasets_sce = results_float[:,2:2+npara]
                objective_sce = results_float[:,1] #* -1.0   # results contain the objective function value that was MINIMIZED

                dict_results_trial['parasets_sce']  = list( [ list(pp) for pp in parasets_sce ] )
                dict_results_trial['objective_sce'] = list(objective_sce)

                dict_result_para['trial_'+str(itrial)] = dict_results_trial

            dict_results['npara_'+str(npara)] = dict_result_para

        # save results to file such that it can be used again later
        # create json object from dictionary
        json_dict = json.dumps(dict_results)

        # open file for writing, "w"
        ff = open(jsonfile,"w")

        # write json object to file
        ff.write(json_dict)

        # close file
        ff.close()


    # read from json file
    with open(jsonfile) as ff:
        dict_results = json.load(ff)

    parasets_random  = [ [[] for itrial in range(ntrials) ] for ipara in [2,10]]
    objective_random = [ [[] for itrial in range(ntrials) ] for ipara in [2,10]]
    parasets_lhs     = [ [[] for itrial in range(ntrials) ] for ipara in [2,10]]
    objective_lhs    = [ [[] for itrial in range(ntrials) ] for ipara in [2,10]]
    parasets_dds     = [ [[] for itrial in range(ntrials) ] for ipara in [2,10]]
    objective_dds    = [ [[] for itrial in range(ntrials) ] for ipara in [2,10]]
    parasets_sce     = [ [[] for itrial in range(ntrials) ] for ipara in [2,10]]
    objective_sce    = [ [[] for itrial in range(ntrials) ] for ipara in [2,10]]

    for iipara,ipara in enumerate([2,10]):
        for itrial in range(ntrials):
            parasets_random[iipara][itrial]  =  dict_results['npara_'+str(ipara)]['trial_'+str(itrial)]['parasets_random']
            objective_random[iipara][itrial] =  dict_results['npara_'+str(ipara)]['trial_'+str(itrial)]['objective_random']
            parasets_lhs[iipara][itrial]     =  dict_results['npara_'+str(ipara)]['trial_'+str(itrial)]['parasets_lhs']
            objective_lhs[iipara][itrial]    =  dict_results['npara_'+str(ipara)]['trial_'+str(itrial)]['objective_lhs']
            parasets_dds[iipara][itrial]     =  dict_results['npara_'+str(ipara)]['trial_'+str(itrial)]['parasets_dds']
            objective_dds[iipara][itrial]    =  dict_results['npara_'+str(ipara)]['trial_'+str(itrial)]['objective_dds']
            parasets_sce[iipara][itrial]     =  dict_results['npara_'+str(ipara)]['trial_'+str(itrial)]['parasets_sce']
            objective_sce[iipara][itrial]    =  dict_results['npara_'+str(ipara)]['trial_'+str(itrial)]['objective_sce']

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
    hspace      = 0.1        # x-space between subplots
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
    cc = color.get_brewer('blues7', rgb=True)
    low_cc = tuple([1.0,1.0,1.0])
    #del cc[0]  # drop darkest two pink color
    #del cc[0]  # drop darkest two pink color
    cc = list([low_cc])+cc      # prepend "white"
    cmap = mpl.colors.ListedColormap(cc)

    inorm = 'pow'

    if inorm == 'log':
        min_sti = 0.01
        max_sti = 1.0
        norm = mcolors.LogNorm(min_sti,max_sti)
    elif inorm == 'pow':
        # n=2 --> samples=200/trial
        pow_lambda1 = 0.2
        max_pow1    = 200.
        norm1 = mcolors.PowerNorm(pow_lambda1,vmin=0,vmax=max_pow1)
        # n=10 --> samples=1000/trial
        pow_lambda2 = 0.2
        max_pow2    = 1000.
        norm2 = mcolors.PowerNorm(pow_lambda2,vmin=0,vmax=max_pow2)
    elif inorm == 'linear':
        norm = None
    else:
        raise ValueError('Norm for colormap not known.')


    ifig = 0

    # -------------------------------------------------
    # arithmetic mean Sobol' indexes
    # -------------------------------------------------

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

    xlim = [-2.3,2.3]
    ylim = [-2.3,2.3]
    iipara = 0

    # -------------------------
    # scatter plot - Random
    # -------------------------
    iplot += 1
    sub = fig.add_axes(position(nrow, ncol, iplot, hspace=hspace/2, vspace=vspace))

    # for itrial in range(ntrials):
    #     scatter_random = sub.scatter(  np.array(parasets_random[iipara][itrial])[:,0],  np.array(parasets_random[iipara][itrial])[:,1], s=msize*0.3, color=lcol1, alpha=0.7 )

    xvals = []
    yvals = []
    for itrial in range(10):
        xvals += list(np.array(parasets_random[iipara][itrial])[:,0])
        yvals += list(np.array(parasets_random[iipara][itrial])[:,1])
    sub.hist2d(xvals, yvals, bins=30, norm=norm1, cmap=cmap)

    sub.text(0.5,1.06, str2tex('Random sampling', usetex=usetex),
                 fontsize=textsize+2,
                 transform=sub.transAxes, horizontalalignment='center', verticalalignment='bottom')
    sub.text(-0.5,-0.22, str2tex('Ackley Function (2-dimensional)', usetex=usetex),
                 fontsize=textsize+2,
                 transform=sub.transAxes, horizontalalignment='center', verticalalignment='center', rotation=90)

    sub.set_xlabel(str2tex('Sampled $x_{1}$', usetex=usetex))
    sub.set_ylabel(str2tex('Sampled $x_{2}$', usetex=usetex))
    sub.set_xlim(xlim)
    sub.set_ylim(ylim)

    # abc
    sub.text( 1.05, 1.0, str2tex(chr(96+iplot),usetex=usetex),
                                ha = 'left', va = 'top',
                                fontweight='bold',
                                transform=sub.transAxes,
                                fontsize=textsize+3 )

    # -------------------------
    # scatter plot - LHS
    # -------------------------
    iplot += 1
    sub = fig.add_axes(position(nrow, ncol, iplot, hspace=hspace/2, vspace=vspace))

    # for itrial in range(10):
    #     scatter_lhs = sub.scatter(  np.array(parasets_lhs[iipara][itrial])[:,0],  np.array(parasets_lhs[iipara][itrial])[:,1], s=msize*0.3, color=lcol1, alpha=0.7 )

    xvals = []
    yvals = []
    for itrial in range(10):
        xvals += list(np.array(parasets_lhs[iipara][itrial])[:,0])
        yvals += list(np.array(parasets_lhs[iipara][itrial])[:,1])
    sub.hist2d(xvals, yvals, bins=30, norm=norm1, cmap=cmap)

    sub.text(0.5,1.06, str2tex('Stratified sampling', usetex=usetex),
                 fontsize=textsize+2,
                 transform=sub.transAxes, horizontalalignment='center', verticalalignment='bottom')

    sub.set_xlabel(str2tex('Sampled $x_{1}$', usetex=usetex))
    sub.set_xlim(xlim)
    sub.set_ylim(ylim)

    sub.set_yticklabels([])

    # abc
    sub.text( 1.05, 1.0, str2tex(chr(96+iplot),usetex=usetex),
                                ha = 'left', va = 'top',
                                fontweight='bold',
                                transform=sub.transAxes,
                                fontsize=textsize+3 )

    # # -------------------------
    # # scatter plot - DDS
    # # -------------------------
    # iplot += 1
    # sub = fig.add_axes(position(nrow, ncol, iplot, hspace=hspace/2, vspace=vspace))

    # for itrial in range(10):
    #     scatter_dds = sub.scatter(  np.array(parasets_dds[iipara][itrial])[:,0],  np.array(parasets_dds[iipara][itrial])[:,1], s=msize*0.3, color=lcol1, alpha=0.7 )

    # sub.text(0.5,1.06, str2tex('DDS sampling', usetex=usetex),
    #              fontsize=textsize+2,
    #              transform=sub.transAxes, horizontalalignment='center', verticalalignment='bottom')

    # sub.set_xlabel(str2tex('Sampled $x_{1}$', usetex=usetex))
    # sub.set_xlim(xlim)
    # sub.set_ylim(ylim)

    # sub.set_yticklabels([])

    # -------------------------
    # scatter plot - SCE
    # -------------------------
    iplot += 1
    sub = fig.add_axes(position(nrow, ncol, iplot, hspace=hspace/2, vspace=vspace))

    # for itrial in range(10):
    #     scatter_sce = sub.scatter( np.array(parasets_sce[iipara][itrial])[:,0],  np.array(parasets_sce[iipara][itrial])[:,1], s=msize*0.3, color=lcol1, alpha=0.7 )

    xvals = []
    yvals = []
    for itrial in range(10):
        xvals += list(np.array(parasets_sce[iipara][itrial])[:,0])
        yvals += list(np.array(parasets_sce[iipara][itrial])[:,1])
    sub.hist2d(xvals, yvals, bins=30, norm=norm1, cmap=cmap)

    sub.text(0.5,1.06, str2tex('SCE sampling', usetex=usetex),
                 fontsize=textsize+2,
                 transform=sub.transAxes, horizontalalignment='center', verticalalignment='bottom')

    sub.set_xlabel(str2tex('Sampled $x_{1}$', usetex=usetex))
    sub.set_xlim(xlim)
    sub.set_ylim(ylim)

    sub.set_yticklabels([])

    # abc
    sub.text( 1.05, 1.0, str2tex(chr(96+iplot),usetex=usetex),
                                ha = 'left', va = 'top',
                                fontweight='bold',
                                transform=sub.transAxes,
                                fontsize=textsize+3 )

    # fill row
    iplot += 1
    pos_cbar = position(nrow, ncol, iplot, hspace=hspace/2, vspace=vspace)
    #print("pos cbar: ",pos_cbar)

    # colorbar
    # [left, bottom, width, height]
    pos_cbar *= [1.0,1.0,0.1,1.0]   # [0.74375 0.444   0.15625 0.112  ]
    #print("pos cbar: ",pos_cbar)
    csub    = fig.add_axes( pos_cbar )

    if inorm == 'log':
        ticks = [ 10.0**(np.log10(min_sti) + ii*(np.log10(max_sti) - np.log10(min_sti))/len(cc)) for ii in range(len(cc)+1) ]
        cbar = mpl.colorbar.ColorbarBase(csub, norm=norm, cmap=cmap, orientation='vertical', extend='min')
        cbar.set_ticklabels([ "${:.1e}$".format(itick) if (iitick%5 ==0) else "" for iitick,itick in enumerate(ticks) ])  # print only every fifth label
        cbar.set_label(str2tex("Density [-]",usetex=usetex))
    elif inorm == 'pow':
        ticks = [0]+[ 10**ii for ii in np.arange(0,5,1) ]  # [0, 1, 10, 100, 1000, 10000]
        ticks = [0, 1, 10, 100, 300, 1000, 3000, 10000]
        cbar = mpl.colorbar.ColorbarBase(csub, norm=norm1, ticks=ticks, cmap=cmap, orientation='vertical', extend='max')  #
        cbar.set_ticklabels([ "${:.0e}$".format(itick) for itick in ticks ])
        cbar.set_label(str2tex("Frequency [-]",usetex=usetex))
    else:
        raise ValueError('Norm for colormap not known.')

    # color bins
    #for ibin in range(cmap.N):
    #    print("Color bin #",ibin+1,"  :: [",((max_pow1**pow_lambda1)/cmap.N*(ibin))**(1./pow_lambda1),',',((max_pow1**pow_lambda1)/cmap.N*(ibin+1))**(1./pow_lambda1),']')





    npara  = 2
    iipara = 0
    nsets = npara*100
    xlim = [0,nsets]
    ylim = [-1,9.]


    # -------------------------
    # objective function - Random - npara=2
    # -------------------------
    iplot += 1
    sub = fig.add_axes(position(nrow, ncol, iplot, hspace=hspace/2, vspace=vspace))

    for itrial in range(ntrials):
        tmp = np.ones(nsets) * np.nan
        tmp = [ np.min(objective_random[iipara][itrial][0:iii+1]) for iii,ii in enumerate(objective_random[iipara][itrial]) if iii < nsets ]
        sub.plot( range(nsets), tmp, linewidth=lwidth, color=lcol1, alpha=0.7 )

    sub.plot( [0,nsets], [0.0,0.0], linestyle='--', linewidth=lwidth/2, color='0.7', alpha=0.7)
    sub.text( 0.05*nsets, -0.1, str2tex('Global Minimum', usetex=usetex),
                  #transform=sub.transAxes,
                  fontsize=textsize-2,
                  color='0.7',
                  horizontalalignment='left', verticalalignment='top')

    #sub.text(0.5,1.03, str2tex('Random sampling', usetex=usetex),
    #             transform=sub.transAxes, horizontalalignment='center', verticalalignment='bottom')

    sub.set_xlabel(str2tex('Samples drawn', usetex=usetex))
    sub.set_ylabel(str2tex('Minimal Function Value', usetex=usetex))
    sub.set_xlim(xlim)
    sub.set_ylim(ylim)

    # abc
    sub.text( 1.05, 1.0, str2tex(chr(96+iplot-1),usetex=usetex),
                                ha = 'left', va = 'top',
                                fontweight='bold',
                                transform=sub.transAxes,
                                fontsize=textsize+3 )

    # -------------------------
    # objective function - LHS - npara=2
    # -------------------------
    iplot += 1
    sub = fig.add_axes(position(nrow, ncol, iplot, hspace=hspace/2, vspace=vspace))

    for itrial in range(ntrials):
        tmp = np.ones(nsets) * np.nan
        tmp = [ np.min(objective_lhs[iipara][itrial][0:iii+1]) for iii,ii in enumerate(objective_lhs[iipara][itrial]) if iii < nsets ]
        obj_lhs = sub.plot( range(nsets), tmp, linewidth=lwidth, color=lcol1, alpha=0.7 )

    sub.plot( [0,nsets], [0.0,0.0], linestyle='--', linewidth=lwidth/2, color='0.7', alpha=0.7)

    #sub.text(0.5,1.03, str2tex('Stratified sampling', usetex=usetex),
    #             transform=sub.transAxes, horizontalalignment='center', verticalalignment='bottom')

    sub.set_xlabel(str2tex('Samples drawn', usetex=usetex))
    sub.set_xlim(xlim)
    sub.set_ylim(ylim)

    sub.set_yticklabels([])

    # abc
    sub.text( 1.05, 1.0, str2tex(chr(96+iplot-1),usetex=usetex),
                                ha = 'left', va = 'top',
                                fontweight='bold',
                                transform=sub.transAxes,
                                fontsize=textsize+3 )

    # # -------------------------
    # # objective function - DDS - npara=2
    # # -------------------------
    # iplot += 1
    # sub = fig.add_axes(position(nrow, ncol, iplot, hspace=hspace/2, vspace=vspace))

    # for itrial in range(ntrials):
    #     tmp = np.ones(nsets) * np.nan
    #     tmp = [ np.min(objective_dds[iipara][itrial][0:iii+1]) for iii,ii in enumerate(objective_dds[iipara][itrial]) if iii < nsets ]
    #     print('trial ',itrial,': ', tmp[0:4])
    #     obj_dds = sub.plot( range(nsets), tmp, linewidth=lwidth, color=lcol1, alpha=0.7 )

    # #sub.text(0.5,1.03, str2tex('DDS sampling', usetex=usetex),
    # #             transform=sub.transAxes, horizontalalignment='center', verticalalignment='bottom')

    # sub.set_xlabel(str2tex('Samples drawn', usetex=usetex))
    # sub.set_xlim(xlim)
    # sub.set_ylim(ylim)

    # sub.set_yticklabels([])

    # -------------------------
    # objective function - SCE - npara=2
    # -------------------------
    iplot += 1
    sub = fig.add_axes(position(nrow, ncol, iplot, hspace=hspace/2, vspace=vspace))

    for itrial in range(ntrials):
        tmp = np.ones(nsets) * np.nan
        tmp = [ np.min(objective_sce[iipara][itrial][0:iii+1]) for iii,ii in enumerate(objective_sce[iipara][itrial]) if iii < nsets ]
        tmp2 = np.ones(nsets) * np.nan
        tmp2[0:len(tmp)] = tmp
        obj_sce = sub.plot( range(nsets), tmp2, linewidth=lwidth, color=lcol1, alpha=0.7 )

    sub.plot( [0,nsets], [0.0,0.0], linestyle='--', linewidth=lwidth/2, color='0.7', alpha=0.7)

    sub.set_xlabel(str2tex('Samples drawn', usetex=usetex))
    sub.set_xlim(xlim)
    sub.set_ylim(ylim)

    sub.set_yticklabels([])

    # abc
    sub.text( 1.05, 1.0, str2tex(chr(96+iplot-1),usetex=usetex),
                                ha = 'left', va = 'top',
                                fontweight='bold',
                                transform=sub.transAxes,
                                fontsize=textsize+3 )

    # fill row
    iplot += 1




    xlim = [-2.3,2.3]
    ylim = [-2.3,2.3]
    iipara = 1

    # -------------------------
    # scatter plot - Random
    # -------------------------
    iplot += 1
    sub = fig.add_axes(position(nrow, ncol, iplot, hspace=hspace/2, vspace=vspace))

    # for itrial in range(ntrials):
    #     scatter_random = sub.scatter(  np.array(parasets_random[iipara][itrial])[:,0],  np.array(parasets_random[iipara][itrial])[:,1], s=msize*0.3, color=lcol1, alpha=0.7 )

    xvals = []
    yvals = []
    for itrial in range(10):
        xvals += list(np.array(parasets_random[iipara][itrial])[:,0])
        yvals += list(np.array(parasets_random[iipara][itrial])[:,1])
    sub.hist2d(xvals, yvals, bins=30, norm=norm2, cmap=cmap)

    #sub.text(0.5,1.03, str2tex('Random sampling', usetex=usetex),
    #             transform=sub.transAxes, horizontalalignment='center', verticalalignment='bottom')
    sub.text(-0.5,-0.22, str2tex('Ackley Function (10-dimensional)', usetex=usetex),
                 fontsize=textsize+2,
                 transform=sub.transAxes, horizontalalignment='center', verticalalignment='center', rotation=90)

    sub.set_xlabel(str2tex('Sampled $x_{1}$', usetex=usetex))
    sub.set_ylabel(str2tex('Sampled $x_{2}$', usetex=usetex))
    sub.set_xlim(xlim)
    sub.set_ylim(ylim)

    # abc
    sub.text( 1.05, 1.0, str2tex(chr(96+iplot-2),usetex=usetex),
                                ha = 'left', va = 'top',
                                fontweight='bold',
                                transform=sub.transAxes,
                                fontsize=textsize+3 )

    # -------------------------
    # scatter plot - LHS
    # -------------------------
    iplot += 1
    sub = fig.add_axes(position(nrow, ncol, iplot, hspace=hspace/2, vspace=vspace))

    # for itrial in range(10):
    #     scatter_lhs = sub.scatter(  np.array(parasets_lhs[iipara][itrial])[:,0],  np.array(parasets_lhs[iipara][itrial])[:,1], s=msize*0.3, color=lcol1, alpha=0.7 )

    xvals = []
    yvals = []
    for itrial in range(10):
        xvals += list(np.array(parasets_lhs[iipara][itrial])[:,0])
        yvals += list(np.array(parasets_lhs[iipara][itrial])[:,1])
    sub.hist2d(xvals, yvals, bins=30, norm=norm2, cmap=cmap)

    #sub.text(0.5,1.03, str2tex('Stratified sampling', usetex=usetex),
    #             transform=sub.transAxes, horizontalalignment='center', verticalalignment='bottom')

    sub.set_xlabel(str2tex('Sampled $x_{1}$', usetex=usetex))
    sub.set_xlim(xlim)
    sub.set_ylim(ylim)

    sub.set_yticklabels([])

    # abc
    sub.text( 1.05, 1.0, str2tex(chr(96+iplot-2),usetex=usetex),
                                ha = 'left', va = 'top',
                                fontweight='bold',
                                transform=sub.transAxes,
                                fontsize=textsize+3 )

    # # -------------------------
    # # scatter plot - DDS
    # # -------------------------
    # iplot += 1
    # sub = fig.add_axes(position(nrow, ncol, iplot, hspace=hspace/2, vspace=vspace))

    # for itrial in range(10):
    #     scatter_dds = sub.scatter(  np.array(parasets_dds[iipara][itrial])[:,0],  np.array(parasets_dds[iipara][itrial])[:,1], s=msize*0.3, color=lcol1, alpha=0.7 )

    # #sub.text(0.5,1.03, str2tex('DDS sampling', usetex=usetex),
    # #             transform=sub.transAxes, horizontalalignment='center', verticalalignment='bottom')

    # sub.set_xlabel(str2tex('Sampled $x_{1}$', usetex=usetex))
    # sub.set_xlim(xlim)
    # sub.set_ylim(ylim)

    # sub.set_yticklabels([])

    # -------------------------
    # scatter plot - SCE
    # -------------------------
    iplot += 1
    sub = fig.add_axes(position(nrow, ncol, iplot, hspace=hspace/2, vspace=vspace))

    # for itrial in range(10):
    #     scatter_sce = sub.scatter( np.array(parasets_sce[iipara][itrial])[:,0],  np.array(parasets_sce[iipara][itrial])[:,1], s=msize*0.3, color=lcol1, alpha=0.7 )

    xvals = []
    yvals = []
    for itrial in range(10):
        xvals += list(np.array(parasets_sce[iipara][itrial])[:,0])
        yvals += list(np.array(parasets_sce[iipara][itrial])[:,1])
    sub.hist2d(xvals, yvals, bins=30, norm=norm2, cmap=cmap)

    #sub.text(0.5,1.03, str2tex('SCE sampling', usetex=usetex),
    #             transform=sub.transAxes, horizontalalignment='center', verticalalignment='bottom')

    sub.set_xlabel(str2tex('Sampled $x_{1}$', usetex=usetex))
    sub.set_xlim(xlim)
    sub.set_ylim(ylim)

    sub.set_yticklabels([])

    # abc
    sub.text( 1.05, 1.0, str2tex(chr(96+iplot-2),usetex=usetex),
                                ha = 'left', va = 'top',
                                fontweight='bold',
                                transform=sub.transAxes,
                                fontsize=textsize+3 )

    # fill row
    iplot += 1
    pos_cbar = position(nrow, ncol, iplot, hspace=hspace/2, vspace=vspace)
    #print("pos cbar: ",pos_cbar)

    # colorbar
    # [left, bottom, width, height]
    pos_cbar *= [1.0,1.0,0.1,1.0]   # [0.74375 0.444   0.15625 0.112  ]
    #print("pos cbar: ",pos_cbar)
    csub    = fig.add_axes( pos_cbar )

    if inorm == 'log':
        ticks = [ 10.0**(np.log10(min_sti) + ii*(np.log10(max_sti) - np.log10(min_sti))/len(cc)) for ii in range(len(cc)+1) ]
        cbar = mpl.colorbar.ColorbarBase(csub, norm=norm, cmap=cmap, orientation='vertical', extend='min')
        cbar.set_ticklabels([ "${:.1e}$".format(itick) if (iitick%5 ==0) else "" for iitick,itick in enumerate(ticks) ])  # print only every fifth label
        cbar.set_label(str2tex("Density [-]",usetex=usetex))
    elif inorm == 'pow':
        ticks = [0]+[ 10**ii for ii in np.arange(0,5,1) ]  # [0, 1, 10, 100, 1000, 10000]
        ticks = [0, 1, 10, 100, 300, 1000, 3000, 10000]
        cbar = mpl.colorbar.ColorbarBase(csub, norm=norm2, ticks=ticks, cmap=cmap, orientation='vertical', extend='max')  #
        cbar.set_ticklabels([ "${:.0e}$".format(itick) for itick in ticks ])
        cbar.set_label(str2tex("Frequency [-]",usetex=usetex))
    else:
        raise ValueError('Norm for colormap not known.')

    # color bins
    #for ibin in range(cmap.N):
    #    print("Color bin #",ibin+1,"  :: [",((max_pow2**pow_lambda2)/cmap.N*(ibin))**(1./pow_lambda2),',',((max_pow2**pow_lambda2)/cmap.N*(ibin+1))**(1./pow_lambda2),']')









    npara  = 10
    iipara = 1
    nsets = npara*100
    xlim = [0,nsets]
    ylim = [-1,9.]

    # -------------------------
    # objective function - Random - npara=10
    # -------------------------
    iplot += 1
    sub = fig.add_axes(position(nrow, ncol, iplot, hspace=hspace/2, vspace=vspace))

    for itrial in range(ntrials):
        tmp = np.ones(nsets) * np.nan
        tmp = [ np.min(objective_random[iipara][itrial][0:iii+1]) for iii,ii in enumerate(objective_random[iipara][itrial]) if iii < nsets ]
        sub.plot( range(nsets), tmp, linewidth=lwidth, color=lcol1, alpha=0.7 )

    #sub.text(0.5,1.03, str2tex('Random sampling', usetex=usetex),
    #             transform=sub.transAxes, horizontalalignment='center', verticalalignment='bottom')

    sub.plot( [0,nsets], [0.0,0.0], linestyle='--', linewidth=lwidth/2, color='0.7', alpha=0.7)
    sub.text( 0.05*nsets, -0.1, str2tex('Global Minimum', usetex=usetex),
                  #transform=sub.transAxes,
                  fontsize=textsize-2,
                  color='0.7',
                  horizontalalignment='left', verticalalignment='top')

    sub.set_xlabel(str2tex('Samples drawn', usetex=usetex))
    sub.set_ylabel(str2tex('Minimal Function Value', usetex=usetex))
    sub.set_xlim(xlim)
    sub.set_ylim(ylim)

    # abc
    sub.text( 1.05, 1.0, str2tex(chr(96+iplot-3),usetex=usetex),
                                ha = 'left', va = 'top',
                                fontweight='bold',
                                transform=sub.transAxes,
                                fontsize=textsize+3 )

    # -------------------------
    # objective function - LHS - npara=10
    # -------------------------
    iplot += 1
    sub = fig.add_axes(position(nrow, ncol, iplot, hspace=hspace/2, vspace=vspace))

    for itrial in range(ntrials):
        tmp = np.ones(nsets) * np.nan
        tmp = [ np.min(objective_lhs[iipara][itrial][0:iii+1]) for iii,ii in enumerate(objective_lhs[iipara][itrial]) if iii < nsets ]
        obj_lhs = sub.plot( range(nsets), tmp, linewidth=lwidth, color=lcol1, alpha=0.7 )

    #sub.text(0.5,1.03, str2tex('Stratified sampling', usetex=usetex),
    #             transform=sub.transAxes, horizontalalignment='center', verticalalignment='bottom')

    sub.plot( [0,nsets], [0.0,0.0], linestyle='--', linewidth=lwidth/2, color='0.7', alpha=0.7)

    sub.set_xlabel(str2tex('Samples drawn', usetex=usetex))
    sub.set_xlim(xlim)
    sub.set_ylim(ylim)

    sub.set_yticklabels([])

    # abc
    sub.text( 1.05, 1.0, str2tex(chr(96+iplot-3),usetex=usetex),
                                ha = 'left', va = 'top',
                                fontweight='bold',
                                transform=sub.transAxes,
                                fontsize=textsize+3 )

    # # -------------------------
    # # objective function - DDS - npara=10
    # # -------------------------
    # iplot += 1
    # sub = fig.add_axes(position(nrow, ncol, iplot, hspace=hspace/2, vspace=vspace))

    # for itrial in range(ntrials):
    #     tmp = np.ones(nsets) * np.nan
    #     tmp = [ np.min(objective_dds[iipara][itrial][0:iii+1]) for iii,ii in enumerate(objective_dds[iipara][itrial]) if iii < nsets ]
    #     print('trial ',itrial,': ', tmp[0:4])
    #     obj_dds = sub.plot( range(nsets), tmp, linewidth=lwidth, color=lcol1, alpha=0.7 )

    # #sub.text(0.5,1.03, str2tex('DDS sampling', usetex=usetex),
    # #             transform=sub.transAxes, horizontalalignment='center', verticalalignment='bottom')

    # sub.set_xlabel(str2tex('Samples drawn', usetex=usetex))
    # sub.set_xlim(xlim)
    # sub.set_ylim(ylim)

    # sub.set_yticklabels([])

    # -------------------------
    # objective function - SCE - npara=10
    # -------------------------
    iplot += 1
    sub = fig.add_axes(position(nrow, ncol, iplot, hspace=hspace/2, vspace=vspace))

    for itrial in range(ntrials):
        tmp = np.ones(nsets) * np.nan
        tmp = [ np.min(objective_sce[iipara][itrial][0:iii+1]) for iii,ii in enumerate(objective_sce[iipara][itrial]) if iii < nsets ]
        tmp2 = np.ones(nsets) * np.nan
        tmp2[0:len(tmp)] = tmp
        obj_sce = sub.plot( range(nsets), tmp2, linewidth=lwidth, color=lcol1, alpha=0.7 )

    sub.plot( [0,nsets], [0.0,0.0], linestyle='--', linewidth=lwidth/2, color='0.7', alpha=0.7)

    sub.set_xlabel(str2tex('Samples drawn', usetex=usetex))
    sub.set_xlim(xlim)
    sub.set_ylim(ylim)

    sub.set_yticklabels([])

    # abc
    sub.text( 1.05, 1.0, str2tex(chr(96+iplot-3),usetex=usetex),
                                ha = 'left', va = 'top',
                                fontweight='bold',
                                transform=sub.transAxes,
                                fontsize=textsize+3 )

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
