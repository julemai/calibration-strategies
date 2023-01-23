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
#     run figure_8.py -t pdf -p figure_8

#!/usr/bin/env python
from __future__ import print_function

"""

Plots results of hydrologic calibration experiments using different metrics

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
                                      description='''Plots results of calibration experiments using different ranges for parameters.''')
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

    jsonfile = 'figure_8.json'

    # -------------------------------------------------------------------------
    # general
    # -------------------------------------------------------------------------

    budget = 900
    npara = 9
    ntrials = 10

    # names of metrics
    metric_names_str   = [ 'RMSE$_\mathrm{Q}$', 'KGE$_\mathrm{Q}$', 'NSE$_\mathrm{Q}$', 'NSE$_\mathrm{\log{Q}}$', 'r$^\mathrm{2}_\mathrm{Q}$', 'PBIAS$_\mathrm{Q}$']  # for plots
    metric_names       = [ 'rmse', 'kge', 'nse', 'lnse', 'r2', 'pbias' ]  # in folder names
    metric_names_raven = [ 'DIAG_RMSE','DIAG_KLING_GUPTA','DIAG_NASH_SUTCLIFFE','DIAG_LOG_NASH','DIAG_R2','DIAG_PCT_BIAS'] # in Raven diagnostics


    if not(os.path.exists(jsonfile)):

        dict_results = {}

        year = 2001    # plot will show only 1 year
        dict_results['year'] = year

        # read SCE calibration results per metric
        for imetric in metric_names:

            para = {}
            obfv = {}
            hist = {}
            Qsim = {}
            for itrial in range(ntrials):

                # get para values over course of calibration
                f = open("example_8/ostrich-gr4j-salmon_sce_"+imetric+"/trial_"+str(itrial+1)+"/OstModel0.txt", "r")
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
                history['obfv'] = oobfv
                history['para'] = np.array(ppara)
                hist['trial_'+str(itrial+1)] = history

                # get calibrated values (last parameter set in history)
                para['trial_'+str(itrial+1)] = history['para'][-1]

                # get objective function values
                f = open("example_8/ostrich-gr4j-salmon_sce_"+imetric+"/trial_"+str(itrial+1)+"/best/output/Diagnostics.csv", "r")
                content = f.readlines()
                f.close()
                header = content[0].strip().split(',')
                idx = [ header.index(imetric_rvn) for imetric_rvn in metric_names_raven ]
                vals =  content[1].strip().split(',')
                obfv['trial_'+str(itrial+1)] = { metric_names[iii] : float(vals[ii]) for iii,ii in enumerate(idx) }  # assumes "metric_names" and "metric_names_raven" have same order

                # get best simulated hydrograph (correponds to para = best parameter set)
                filename = "example_8/ostrich-gr4j-salmon_sce_"+imetric+"/trial_"+str(itrial+1)+"/best/output/Hydrographs.csv"
                start_date = datetime.datetime(year,1,1,0,0)
                end_date   = datetime.datetime(year+1,1,1,0,0)
                result = read_raven_hydrograph(filename,start_date=start_date,end_date=end_date)
                dates = [ str(ii) for ii in list(result['dates']) ]
                sim = list(result['Qsim'])
                obs = list(result['Qobs'])
                Qsim['trial_'+str(itrial+1)] = sim

                # this overwrites very often
                if not( 'dates' in list(dict_results.keys()) ):
                    dict_results['dates'] = dates
                if not( 'Qobs' in list(dict_results.keys()) ):
                    dict_results['Qobs']  = obs

            dict_results[imetric] = { 'para': para, 'obfv': obfv, 'hist': hist, 'Qsim': Qsim }

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
    lcols       = color.colours(['black','blue','green','yellow'])
    markers     = ['o','v','s','^']

    # Legend
    llxbbox     = 1.1        # x-anchor legend bounding box
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

    #dates = np.array([ datetime.datetime.strptime( ii, '%Y-%m-%d %H:%M:%S') for ii in  dict_results['dates'] ])
    mydates = dict_results['dates']

    xpoints = 1 # plots only every nth point

    # -------------------------
    # plot - hydrographs for each metric calibrated
    # -------------------------
    for iimetric,imetric in enumerate(metric_names):
        iplot += 1
        sub = fig.add_axes(position(nrow, ncol, iplot, hspace=hspace, vspace=vspace))

        sub.text(0.5,1.01, str2tex('Calibrate '+metric_names_str[iimetric], usetex=usetex),
                     fontsize=textsize+2,
                     transform=sub.transAxes, horizontalalignment='center', verticalalignment='bottom')

        # which one is best trial
        all_final_obfv = [ dict_results[imetric]['obfv']['trial_'+str(itrial+1)][imetric] for itrial in range(ntrials) ]
        # print("metric: {} --> trials: {}".format(imetric,all_final_obfv))
        if imetric in ['rmse']:
            idx_best_trial = np.argmin(all_final_obfv)
        elif imetric in ['pbias']:
            idx_best_trial = np.argmin(np.abs(all_final_obfv))
        elif imetric in ['nse', 'lnse', 'kge', 'r2', ]:
            idx_best_trial = np.argmax(all_final_obfv)
        else:
            raise ValueError('Metric {} not known'.formt(imetric))

        # labels of all metrics evaluated for best trial to plot
        all_metrics_best_trial = dict_results[imetric]['obfv']['trial_'+str(idx_best_trial+1)]
        for iii,ii in enumerate(all_metrics_best_trial):
            if ii == imetric:
                color = 'k'
            else:
                color = '0.5'
            istr =  metric_names_str[metric_names.index(ii)]+' = '+astr(all_metrics_best_trial[ii],prec=2)
            sub.text(0.98,0.92-iii*0.10, str2tex( istr, usetex=usetex),
                     fontsize=textsize-2, color=color,
                     transform=sub.transAxes, horizontalalignment='right', verticalalignment='center')

        # simulated hydrographs (best blue, others gray)
        first_label = False
        for itrial in range(ntrials):

            if itrial == idx_best_trial:
                color = lcol1
                alpha = 1.0
                zorder = 100
                label = str2tex('Simulated Discharge $Q_\mathrm{sim}^\mathrm{best}$\n(best calibration trial)',usetex=usetex) #str2tex('$Q_\mathrm{sim}^\mathrm{best}$',usetex=usetex)
            else:
                color = '0.8'
                alpha = 0.7
                zorder = 10
                if not(first_label):
                    label = str2tex('Simulated Discharge $Q_\mathrm{sim}$\n(all calibration trials) ',usetex=usetex) #str2tex('$Q_\mathrm{sim}$',usetex=usetex)
                    first_label = True
                else:
                    label = ''

            sub.plot( np.array(mydates)[::xpoints], np.array(dict_results[imetric]['Qsim']['trial_'+str(itrial+1)])[::xpoints],
                      linewidth=lwidth, linestyle='-', color=color, alpha=alpha, zorder=zorder, label=label)

        # observed streamflow
        label = str2tex('Observed Dicharge $Q_{obs}$',usetex=usetex)
        sub.plot( np.array(mydates)[::xpoints], np.array(dict_results['Qobs'])[::xpoints],
                      linewidth=0.0, marker='o', color=lcol1,
                      markersize=msize/2, markeredgewidth=msize/6,markerfacecolor='none',
                      alpha=0.7, zorder=400, label=label)

        #sub.set_xlabel(str2tex('Simulation Period [Year '+str(2001)+']', usetex=usetex))
        if iplot%2 == 1:
            sub.set_ylabel(str2tex('Discharge Q [$\mathrm{m}^\mathrm{3} \mathrm{s}^\mathrm{-1}$]', usetex=usetex))
        else:
            sub.set_yticklabels([])

        # format x-axis with dates
        monthlength = np.array([31,28,31,30,31,30,31,31,30,31,30,31])
        sub.set_xlim([0,365])
        if (iplot+1)//3 == 2: # last row
            sub.set_xlabel(str2tex('Months of Year '+str(dict_results['year']),usetex=usetex), color='black')
        sub.set_xticks(np.cumsum(monthlength)) #, ['J','F','M','A','M','J','J','A','S','O','N','D'])
        sub.set_xticklabels('')
        sub.set_xticks((monthlength*1.0/2.)+np.cumsum(np.append(0,monthlength)[0:12]), minor=True)
        if (iplot+1)//3 == 2: # last row
            sub.set_xticklabels(['J','F','M','A','M','J','J','A','S','O','N','D'], minor=True)
        sub.tick_params(which='minor',length=0)   # dont show minor ticks; only labels

        # # limits
        # xmin=np.min(dict_results['logx'])
        # xmax=np.max(dict_results['logx'])
        # ymin=np.min(dict_results['logy'])
        # ymax=np.max(dict_results['logy'])
        # delta=0.02
        # sub.set_xlim([xmin-delta*(xmax-xmin),xmax+delta*(xmax-xmin)])
        # sub.set_ylim([ymin-delta*(ymax-ymin),ymax+delta*4*(ymax-ymin)])

        # legend
        if iplot == 5:
            sub.legend(frameon=frameon, ncol=3,
                                labelspacing=llrspace, handletextpad=llhtextpad, handlelength=llhlength,
                                loc='lower center', bbox_to_anchor=(llxbbox,llybbox), scatterpoints=1, numpoints=1,
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
