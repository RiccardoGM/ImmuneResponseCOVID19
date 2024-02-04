# Module with custom functions

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# ---- # ---- # ---- # ---- # ---- # ---- # ---- # ---- #

def SetPlotParams(magnification=1.0, ratio=float(2.2/2.7), height=None, width=None, fontsize=11., ylabelsize=None, xlabelsize=None, lines_w=1.5, axes_lines_w=0.7, legendmarker=True, tex=False, autolayout=True, handlelength=1.5):
    
    #plt.style.use('ggplot')

    if (ylabelsize==None):
        ylabelsize = fontsize
    if (xlabelsize==None):
        xlabelsize = fontsize

    ratio = ratio  # usually this is 2.2/2.7
    fig_width = 2.9 * magnification # width in inches
    fig_height = fig_width*ratio  # height in inches
    if height!=None:
        fig_height = height
    if width!=None:
        fig_width = width
    fig_size = [fig_width,fig_height]
    plt.rcParams['figure.figsize'] = fig_size
    plt.rcParams['figure.autolayout'] = autolayout

    plt.rcParams['lines.linewidth'] = lines_w
    #plt.rcParams['lines.markeredgewidth'] = 0.25
    #plt.rcParams['lines.markersize'] = 1
    plt.rcParams['lines.markeredgewidth'] = 1.
    plt.rcParams['errorbar.capsize'] = 1 #1.5

    plt.rcParams['font.size'] = fontsize
    plt.rcParams['legend.frameon'] = False
    plt.rcParams['legend.numpoints'] = 1
    plt.rcParams['legend.markerscale'] = 1
    plt.rcParams['legend.handlelength'] = handlelength
    plt.rcParams['legend.labelspacing'] = 0.3
    plt.rcParams['legend.columnspacing'] = 0.3
    if legendmarker==False:
        plt.rcParams['legend.numpoints'] = 1
        plt.rcParams['legend.markerscale'] = 0
        plt.rcParams['legend.handlelength'] = 0
        plt.rcParams['legend.labelspacing'] = 0  
    plt.rcParams['legend.fontsize'] = fontsize
    plt.rcParams['axes.facecolor'] = '1'
    plt.rcParams['axes.edgecolor'] = '0.0'
    plt.rcParams['axes.linewidth'] = axes_lines_w

    plt.rcParams['grid.color'] = '0.85'
    plt.rcParams['grid.linestyle'] = '-'
    plt.rcParams['grid.linewidth'] = axes_lines_w
    plt.rcParams['grid.alpha'] = '1.'

    plt.rcParams['axes.labelcolor'] = '0'
    plt.rcParams['axes.labelsize'] = fontsize
    plt.rcParams['axes.titlesize'] = fontsize
    plt.rcParams['axes.spines.right'] = False
    plt.rcParams['axes.spines.top'] = False

    plt.rcParams['xtick.labelsize'] = xlabelsize
    plt.rcParams['ytick.labelsize'] = ylabelsize
    plt.rcParams['xtick.color'] = '0'
    plt.rcParams['ytick.color'] = '0'

    plt.rcParams['xtick.major.size'] = 3.
    plt.rcParams['xtick.major.width'] = axes_lines_w
    plt.rcParams['xtick.minor.size'] = 0
    plt.rcParams['ytick.major.size'] = 3.
    plt.rcParams['ytick.major.width'] = axes_lines_w
    plt.rcParams['ytick.minor.size'] = 0
    plt.rcParams['xtick.major.pad']= 5.
    plt.rcParams['ytick.major.pad']= 5.

    #plt.rcParams['font.sans-serif'] = 'Helvetica'
    #plt.rcParams['font.serif'] = 'mc'
    plt.rcParams['text.usetex'] = tex # set to True for TeX-like fonts
    #plt.rcParams['text.latex.preamble'] = r'\usepackage{sfmath} \usepackage{upgreek} \usepackage{siunitx}'
    #mpl.rc('text.latex', preamble=r'\usepackage{sfmath}')
    #plt.rcParams['text.latex.preamble'] = r'\usepackage{siunitx} \sisetup{detect-all} \usepackage{helvet} \usepackage{sansmath} \sansmath'
    #plt.rcParams['mathtext.default'] = 'regular'
    #plt.rcParams['mathtext.fontset'] = 'stixsans'
    #plt.rcParams['font.family'] = 'sans-serif'
    #plt.rc('font', family='sans-serif')

    #sn.set_context("paper", rc={"font.size":11, "axes.titlesize":11, "axes.labelsize":11}) # uncomment for seaborn
    

# ---- # ---- # ---- # ---- # ---- # ---- # ---- # ---- #

def isnumeric(s):
    try:
        float(str(s))
        return True
    except ValueError:
        return False


# ---- # ---- # ---- # ---- # ---- # ---- # ---- # ---- #
    
def MLR_axis(model, Data, Data_test=np.array([]), use_bias=False):
    
    '''
        model: sklearn MLR model fitted to Data
        Data: input data (n_data x n_dim)
    '''
    
    Result = {}
    
    Data_0 = Data.copy()
    
    bias = model.intercept_.item()
    v1_0 = model.coef_.reshape(-1, 1)
    v1_0_norm = np.sqrt(sum(v1_0**2))
    v1_0_rep = np.repeat(v1_0.reshape((1, -1)), Data_0.shape[0], axis=0)
    
    X1_0 = np.dot(Data_0, v1_0)
    Data_0_parallel = v1_0_rep * X1_0 / v1_0_norm**2
    Data_0_perpendicular = Data_0 - Data_0_parallel
    DR_pca = PCA(n_components=1)
    X2_0 = DR_pca.fit(Data_0_perpendicular).transform(Data_0_perpendicular) 
    X2D_0 = np.concatenate((X1_0+bias, X2_0), axis=1) # +bias!
    
    
    v1 = np.insert(v1_0, 0, bias, axis=0)
    v1_norm = np.sqrt(sum(v1**2))
    v1_rep = np.repeat(v1.reshape((1, -1)), Data_0.shape[0], axis=0)
    
    Data = np.insert(Data_0, 0, np.ones_like(Data_0.shape[0]), axis=1)
    X1 = np.dot(Data, v1)
    Data_parallel = v1_rep * X1 / v1_norm**2
    Data_perpendicular = Data - Data_parallel
    #print(np.dot(Data_perpendicular, Data_parallel.transpose()))
    DR_pca_bias = PCA(n_components=1)
    X2 = DR_pca_bias.fit(Data_perpendicular).transform(Data_perpendicular) 
    X2D = np.concatenate((X1, X2), axis=1)
    
    if use_bias:
        Result['Train'] = X2D
    else:
        Result['Train'] = X2D_0
    
    if Data_test.any():
        Data_test_0 = Data_test.copy()
        
        X1_test_0 = np.dot(Data_test_0, v1_0)
        v1_0_rep_t = np.repeat(v1_0.reshape((1, -1)), Data_test_0.shape[0], axis=0)
        Data_test_0_parallel = v1_0_rep_t * X1_test_0 / v1_0_norm**2
        Data_test_0_perpendicular = Data_test_0 - Data_test_0_parallel
        #print(np.dot(Data_perpendicular, Data_parallel.transpose()))
        #DR_pca = PCA(n_components=1)
        X2_test_0 = DR_pca.transform(Data_test_0_perpendicular) 
        X2D_test_0 = np.concatenate((X1_test_0+bias, X2_test_0), axis=1)
        
        Data_test = np.insert(Data_test_0, 0, np.ones_like(Data_test_0.shape[0]), axis=1)
        X1_test = np.dot(Data_test, v1)
        v1_rep_t = np.repeat(v1.reshape((1, -1)), Data_test_0.shape[0], axis=0)
        Data_test_parallel = v1_rep_t * X1_test / v1_norm**2
        Data_test_perpendicular = Data_test - Data_test_parallel
        #print(np.dot(Data_perpendicular, Data_parallel.transpose()))
        #DR_pca = PCA(n_components=1)
        X2_test = DR_pca_bias.transform(Data_test_perpendicular) 
        X2D_test = np.concatenate((X1_test, X2_test), axis=1)
        
        if use_bias:
            Result['Test'] = X2D_test
        else:
            Result['Test'] = X2D_test_0
        
    return Result


# ---- # ---- # ---- # ---- # ---- # ---- # ---- # ---- #

def change_names(names, greekletters=False, cytokines=False, bloodtests=False, flowcyt=False):
    
    new_names = []
    for i, element in enumerate(names):
        if '(charlson comorbidity index)' in element:
            new_name = 'CCI'
        elif 'parenchima' in element:
            new_name = 'par. ext.'
        elif element=='MONO DR IFI':
            new_name = 'Mono IFI'
        elif element=='Mono DR %':
            new_name = 'HLADR + (% Mono)'
        elif element=='LRTE % dei CD4':
            new_name = 'LRTE (% CD4)'
        elif element=='score_Cesselli':
            new_name = 'Ces. score'
        elif element=='TC1_level':
            new_name = 'CT severity'
        elif element=='PCR':
            new_name = 'CRP'
        elif element=='PROADM':
            new_name = 'proADM'
        else:
            new_name = element
            if 'Linfo' in new_name:
                new_name = new_name.replace('Linfo', 'Lymph')
            if 'IOT' in new_name:
                new_name = new_name.replace('IOT', 'OTI')
            if '_' in new_name:
                new_name = new_name.replace('_', ' ')
            if 'POS' in new_name:
                new_name = new_name.replace('POS', '+')
            if '4 C' in new_name:
                new_name = new_name.replace('4 C', '4C')
            if 'T ' in new_name:
                new_name = new_name.replace('T ', '')
            if 'B ' in new_name:
                new_name = new_name.replace('B ', '')

            if '% CD4 ' in new_name:
                new_name = new_name.replace('% CD4 ', '') + ' (% CD4)'
            elif '% CD8 ' in new_name:
                new_name = new_name.replace('% CD8 ', '') + ' (% CD8)'
            elif ' %' in new_name:
                new_name = new_name.replace(' %', ' (%lymph)')
                
        if cytokines:
            new_name = new_name.replace('/uL', ' $pg$/uL')
        elif flowcyt:
            new_name = new_name.replace('/uL', ' $U$/uL')
        elif bloodtests:
            new_name = new_name + ' $U$/uL'
            
        if greekletters:
            if 'uL' in new_name:
                new_name = new_name.replace('uL', '$\mu l$')            
            
        new_names.append(new_name)
            
    return new_names


# ---- # ---- # ---- # ---- # ---- # ---- # ---- # ---- #

def change_names_2(names, units=False):
    
    new_names = []
    for i, element in enumerate(names):
        
        new_name = element
        
        if element=='CCI (charlson comorbidity index)':
            new_name = 'CCI'

        # delta onset
        elif element=='delta_onset':
            new_name = '$\mathrm{\Delta t_{ons}}$'
            
        # CT scan
        elif element=='parenchima_width':
            new_name = 'par. ext.'
        elif element=='TC1_level':
            new_name = 'CT severity'

            
        # Flowcyt
        elif element=='Mono/uL':
            new_name = 'Mono'        
        elif element=='MONO DR IFI':
            new_name = 'Mono$^{\mathrm{+}}$ IFI'
        elif element=='Mono DR %':
            new_name = 'Mono$^{\mathrm{+}}$ \%'
            
        elif element=='WBC/uL':
            new_name = 'WBC'
        elif element=='Linfo/uL':
            new_name = 'Lymph'
            
        elif element=='NeutroBaEu/uL':
            new_name = 'Neutro$^*$'
            
        elif element=='LRTE % dei CD4':
            new_name = 'RTE \%$_{\mathrm{CD4}}$'
        elif element=='LRTE/uL':
            new_name = 'RTE'
        elif element=='T CD3 %':
            new_name = 'CD3 \%lym'
        elif element=='T CD3/uL':
            new_name = 'CD3'
        elif element=='T CD3 HLADR/uL':
            new_name = 'CD3$^{\mathrm{+}}$'
        elif element=='T CD3 HLADR %':
            new_name = 'CD3$^{\mathrm{+}}$ \%lym'
            
        elif element=='T CD4 %':
            new_name = 'CD4 \%lym'
        elif element=='T CD4/uL':
            new_name = 'CD4'
        elif element=='T CD4 HLADR %':
            new_name = 'CD4$^{\mathrm{+}}$ \%lym'
        elif element=='% T CD4 HLADR POS':
            new_name = 'CD4$^{\mathrm{+}}$ \%'
            
        elif element=='T CD8 %':
            new_name = 'CD8 \%lym'
        elif element=='T CD8/uL':
            new_name = 'CD8'
        elif element=='T CD8 HLADR %':
            new_name = 'CD8$^{\mathrm{+}}$ \%lym'
        elif element=='% T CD8 HLADR POS':
            new_name = 'CD8$^{\mathrm{+}}$ \%'
            
        elif element=='B CD19/uL':
            new_name = 'CD19'    
        elif element=='B CD19 %':
            new_name = 'CD19 \%lym'            
        elif element=='NK/uL':
            new_name = 'NK'
        elif element=='NK %':
            new_name = 'NK \%lym'

            
        # Biomarkers
        elif element=='PCR':
            new_name = 'CRP'
        elif element=='PROADM':
            new_name = 'proADM'

            
        # Scores
        elif element=='score_Cesselli':
            new_name = 'Ces. score'
        elif element=='4 C score':
            new_name = '4C'
            
            
         # Cytokines
        elif element=='IFNGC':
            new_name = 'IFN-$\mathrm{\gamma}$'    
            
        # Outcome data
        elif element=='hospitalization_length':
            new_name = 'hosp. stay'
        elif 'IOT' in element:
            new_name = new_name.replace('IOT', 'OTI')
        elif 'merged_' in element:
            new_name = new_name.replace('merged', '')
        elif '_' in element:
            new_name = new_name.replace('_', ' ')

            
        # Add units
        if units:
            # U/ul
            list1 = ['WBC/uL', 'Mono/uL', 'Linfo/uL','T CD3/uL', 'LRTE/uL',
                     'T CD3/uL', 'T CD4/uL','T CD8/uL','CD4/CD8','NK/uL','B CD19/uL',
                     'T CD3 HLADR/uL','T NK-like/uL','LRTE/uL','MONO DR IFI', 
                     'LDH'] + ['NeutroBaEu/uL']
            
            if element in list1:
                new_name += ' $U$/$\mu$$l$'
                
            if element=='PROADM':
                new_name += ' $ng$/$ml$'
                
            if element=='PCR':
                new_name += ' $mg$/$l$'

                
            # pg/ul
            list2 = ['IFNGC', 'IL10', 'IL1B', 'IL2R', 'IL6', 'IL6C', 'IL8', 'IP10']
            if element in list2:
                new_name += ' $pg$/$ml$'                 
            
        new_names.append(new_name)
            
    return new_names


# ---- # ---- # ---- # ---- # ---- # ---- # ---- # ---- #

def lighten_color(color, amount=0.5):
    """
    Lightens the given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.

    Examples:
    >> lighten_color('g', 0.3)
    >> lighten_color('#F034A3', 0.6)
    >> lighten_color((.3,.55,.1), 0.5)
    """
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])