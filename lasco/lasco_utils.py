import math 
import astropy.time
from datetime import datetime
from scipy.ndimage import zoom
import numpy as np 

def poly(x, coeff):
    return sum(c * (x**i) for i, c in enumerate(coeff))


def offset_bias(header,readport,telescope,summm=False):
    
    time = astropy.time.Time(datetime.strptime(header["date"], "%Y/%m/%d %H:%M:%S.%f"),format='datetime', scale='utc')
    mjd = time.mjd
    port = readport.upper()
    tel = telescope.upper()
    time = 0


    b = 0

    if tel == "C1":
        if port == "A":
            b = 364
        elif port == "B":
            b = 331
        elif port == "C":
            # bias calculation
            dd = (mjd - 50395)
            bias = 351.958 + 30.739 * (1 - math.exp(-dd / 468.308))
            b = round(bias)
        elif port == "D":
            b = 522

    elif tel == "C2":
        if port == "A":
            b = 364
        elif port == "B":
            b = 540
        elif port == "C":
            firstday = 50079
            coeff = None

            if mjd <= 51057:
                coeff = [470.97732, 0.12792513, -3.6621933e-05]
            if 51057 < mjd < 51819:
                coeff = [551.67277, 0.091365902, -0.00012637790, 7.4049597e-08]
                firstday = 51099
            if 51819 <= mjd < 51915:
                coeff = [574.5788, 0.019772032]
                firstday = 51558
            if 51915 <= mjd < 54792:
                coeff = [581.81517, 0.019221920, -2.3110489e-06]
                firstday = 51915
            if 54792 <= mjd < 55044:
                coeff = [617.70556, 0.010290491, -6.0131545e-06]
                firstday = 54792
            if 55044 <= mjd < 56450:
                coeff = [619.99733, 0.0059081617, -3.3932229e-07]
                firstday = 55044
            if 56450 <= mjd < 57388:
                coeff = [627.61246, 0.0049003351, -5.3812001e-07]
                firstday = 56450
            if 57388 <= mjd < 58571:
                coeff = [631.20515, 0.0056815108, -1.3439596e-06]
                firstday = 57290
            if 58571 <= mjd < 58802:
                coeff = [651.50189, -0.028926206, 7.8531807e-05, -5.8964538e-08]
                firstday = 58578
            if mjd >= 58802:
                coeff = [648.41904, -0.0020514176, 1.8072963e-05]
                firstday = 58802

            dd = mjd - firstday
            b = poly(dd, coeff)

        elif port == "D":
            b = 526

    elif tel == "C3":
        if port == "A":
            b = 314
        elif port == "B":
            b = 346
        elif port == "C":
            if mjd < 50072:
                b = 319
            else:
                firstday = None
                coeff = None
                if mjd <= 51057:
                    coeff = [322.21639, 0.011775379, 4.4256968E-05, -3.167423e-08]
                    firstday = 50072
                if 51057 < mjd <= 51696:
                    coeff = [354.50857, 0.062196067, -8.8114799e-05, 5.0505447e-08]
                    firstday = 51099
                if 51696 < mjd < 51915:
                    coeff = [369.02719, 0.014994955, -4.0873204e-06]
                    firstday = 51558
                if 51915 <= mjd < 54792:
                    coeff = [374.11139, 0.010731823, -1.0726207e-06]
                    firstday = 51915
                if 54792 <= mjd < 55044:
                    coeff = [395.85091, 0.0079344115, -6.2530780e-06]
                    firstday = 54792
                if 55044 <= mjd < 56170:
                    coeff = [397.52040, 0.0040765192]
                    firstday = 55044
                if 56170 <= mjd < 56366:
                    coeff = [407.04606, -0.024819780, 0.00011694347]
                    firstday = 56170
                if 56360 <= mjd < 56478:
                    coeff = [406.01009, 0.0046765547, -9.912626e-07]
                    firstday = 56360
                if 56478 <= mjd < 56597:
                    coeff = [406.72179, 0.0045780623, -2.3134855e-06]
                    firstday = 56478
                if 56597 <= mjd < 57174:
                    coeff = [406.77706, 0.0040538719, -1.6571028e-06]
                    firstday = 56478
                if mjd >= 57024:
                    coeff = [408.38117, 0.0027558157, -1.3694218e-07]
                    firstday = 57024

                dd = mjd - firstday
                b = poly(dd, coeff)

        elif port == "D":
            b = 283

    elif tel == "EIT":
        if port == "A":
            b = 1017
        elif port == "B":
            b = 840
        elif port == "C":
            b = 1041
        elif port == "D":
            b = 844

    if summm:
        lebsum = max(header["lebxum"],1) * max(header["lebyum"],1)
        b = lebsum*b
    return b 


## pass a string to it please
def substense(telescope):
    telescope = telescope.upper()

    if telescope=='EIT':
        tel = 4
    elif telescope == 'Mk3':
        tel = 5
    elif telescope =="MK4":
        tel = 6
    else:
        tel = int(telescope[1:2])

    match tel:
        case 1:
            f = 5.8
        case 2:
            f = 11.9
        case 3:
            f = 56.0
        case 4:
            f = 2.59
        case 5:
            f = 9.19
        case 6:
            f= 9.19
        case _:
            print("wrong tel ", tel)

    return f 





def get_sec_pixel(header,full=None):
    detector = header["detector"]

    #skip all the verifications of header and which spacecraft, lets just assume it is called correctly 

    sec_pix = substense(detector)
    if full is not None:
        factor = (1024.0 / float(full))
        return sec_pix * factor
    

    sum_check = (header["sumcol"] + 1) * header["lebxsum"]

    binfac = float(header["r2col"] - header["r1col"] + 1) / header["naxis1"]
    if binfac != sum_check:
            print("Warning: Result not correct for Subfields!")
    sec_pix = sec_pix * binfac

    return sec_pix


def rebin(array, new_shape):
    """
    Rebin an array to a new shape by interpolation.
    
    Parameters:
    array (numpy.ndarray): Input array to be rebinned.
    new_shape (tuple): New shape (rows, columns) for the output array.
    
    Returns:
    numpy.ndarray: Rebinned array.
    """
    ## CHANGE added this function
    shape = array.shape
    zoom_factors = [n / o for n, o in zip(new_shape, shape)]

    return zoom(array, zoom_factors, order=1)



def find_miss_blocks(data,header):

    side = 32 

    hdrfieldstruct = getl05hdrparam(header)

    imax = header["NAXIS1"]/side
    jmax = header["NAXIS2"]/side


    red_image = rebin(data, (imax, jmax))
    
    red_imsq=np.zeros((32//hdrfieldstruct["rebindex"],32//hdrfieldstruct["rebindex"]))

    idep=int(hdrfieldstruct["fxstart"]/side/hdrfieldstruct["rebindex"])
    jdep=int(hdrfieldstruct["fystart"]/side/hdrfieldstruct["rebindex"])
    ifin=int((hdrfieldstruct["fxend"]+1)/side/hdrfieldstruct["rebindex"]-1)
    jfin=int((hdrfieldstruct["fyend"]+1)/side/hdrfieldstruct["rebindex"]-1)
    red_imsq[jdep:jfin+1, idep:ifin+1] = red_image
    red_image=red_imsq


    missbmap = (red_image != 0)
    missbmapall = missbmap.copy()       

    return get_tmask(hdrfieldstruct["detector"],hdrfieldstruct["rebindex"]/2 )


    return red_image


def getl05hdrparam(hdr, offsetbias=None):
    # Extract basic parameters
    bitpix = hdr['BITPIX']
    detector = str(hdr['DETECTOR'])
    sx = hdr['NAXIS1']
    sy = hdr['NAXIS2']
    pm = t_params(hdr["detector"].upper())
    # fystart
    try:
        fystart = hdr['R1ROW']-1 
    except:
        try:
            fystart = hdr['P1ROW'] - 1
        except:
            fystart = 0


    fystart = int(fystart)

    try:
        fxstart = hdr['R1COL']-20
    except:
        try:
            fxstart = hdr['P1COL'] - 20
        except:
            fxstart = 0

    fxstart = int(fxstart)

    # fyend
    try:
        fyend = hdr['R2ROW']-1
    except:
        try:
            fyend = hdr['P2ROW'] - 1
        except:
            fyend = 0
            
    fyend = int(fyend)
    
    # fxend
    try:
        fxend = hdr['R2COL']-20
    except:
        try:
            fxend = hdr['P2COL'] - 20
        except:
            fxend = 0
            
    fxend = int(fxend)
 

    # Rebinnings
    nrebinx = (fxend - fxstart + 1) / sx
    nrebiny = (fyend - fystart + 1) / sy

    if bitpix > 0:
        # Integer coded
        if offsetbias is not None:
            offsbias = offset_bias(hdr,hdr["readport"],hdr["detector"], summm=True)
        else:
            offsbias = 0.0

        lebxsum = hdr['LEBXSUM']
        lebysum = hdr['LEBYSUM']

        B_thresh = pm['BIAS'] * float(lebxsum * lebysum)
        if (offsetbias is not None and 
            (offsbias < B_thresh * 0.95 or offsbias > B_thresh * 1.5)):
            print(f"Warning: strange bias from offset_bias? {offsbias} -> {B_thresh}")
            offsbias = B_thresh

        sumcolx = max(1, hdr['SUMCOL'])
        sumrowy = max(1, hdr['SUMROW'])

    else:
        # Likely floating-point image
        offsbias = 0.0
        lebxsum = 1
        lebysum = 1
        sumcolx = nrebinx
        sumrowy = nrebiny

    filename = str(hdr['FILENAME'])

    # rebindex cases
    if nrebinx == 1 and nrebiny == 1:
        rebindex = 1
    elif nrebinx == 2 and nrebiny == 2:
        rebindex = 2
    elif nrebinx == 4 and nrebiny == 4:
        rebindex = 4
    else:
        print(f"Unknown format for {filename} image?")
        rebindex = -1

    # Return as dictionary
    return dict(
        filename=filename,
        bitpix=bitpix,
        detector=detector,
        sx=sx,
        sy=sy,
        fystart=fystart,
        fxstart=fxstart,
        fyend=fyend,
        fxend=fxend,
        nrebinx=nrebinx,
        nrebiny=nrebiny,
        offsbias=offsbias,
        lebxsum=lebxsum,
        lebysum=lebysum,
        sumcolx=sumcolx,
        sumrowy=sumrowy,
        rebindex=rebindex
    )

def t_params(detector):

    pm = {'PIXEL': 0.0, 'SCALE': 0.0, 'FOCAL': 0.0, 'FIELD': 0.0, 
        'DFIELD': 0.0, 'BIAS': 0.0, 'OCCULTER': 0.0, 'DISTORTION': [0.,0.,0.],
        'DISCNTR': [0.,0.], 'CENTER': [0.,0.]}
    

    if detector == 'C1':
        pm["pixel"] = 0.021
        pm["focal"] = 768.
        pm["bias" ] = 332.
        pm["occulter"] = 170.
        pm["distortion"] = [0.,0.,0.]
        pm["center"] = [511.5,511.5]                         
        occenter  = [511.5,511.5]
    elif detector == 'C2':
        pm["pixel"] = 0.021
        pm["focal"] = 364.
        pm["bias" ] = 470.
        pm["occulter"] = 166.
        pm["distortion"] = [.0051344125, -.00012233862, 1.0978595e-07]  
        pm["center"] = [513.5,505.5]                          
        occenter  = [512.,506.]

    elif detector == 'C3':
        pm["pixel"] = 0.021
        pm["focal"] = 77.2
        pm["bias" ] = 319.2
        pm["occulter"] = 67.
        pm["distortion"] = [-0.018757004,0.00018664969]  
        pm["center"] = [518.05,531.4]                        
        occenter  = [516.3,529.5]

    pm["scale"] = np.rad2deg(pm["pixel"]/pm["focal"])
    pm["field"] = pm["scale"]*1024
    pm["dfield"] = np.sqrt(2.)*pm["field"]
    pm["discntr"] = np.array(occenter)*pm["pixel"]

    return pm 

def get_tmask(camera,rebin_index):
    imsksize = 32
    imsk = np.zeros((imsksize,imsksize))+1
    channel = camera.upper()

    if channel=="C2":
        if(rebin_index==0):
            lpmsk =  [0,1,30, 31,32, 63,960,991,992, 993,1022,1023]
            nlp = len(lpmsk)
            lcmsk = [397, 398, 399, 400, 401,429, 430, 431, 432, 433, 434, 460, 461, 462, 463, 464, 465, 466, 467, 492, 493, 494, 495, 496, 497, 498, 499, 524, 525, 526, 527, 528, 529, 530, 531,556, 557, 558, 559, 560, 561, 562, 563,589, 590, 591, 592, 593, 594,622, 623, 624, 625 ]
            nlc = len(lcmsk)
        elif rebin_index ==1:
            nlp = 0
            lcmsk = [ 398, 399, 400, 401, 430, 431, 432, 433, 460, 461, 462, 463, 464, 465, 466, 467, 492, 493, 494, 495, 496, 497, 498, 499, 524, 525, 526, 527, 528, 529, 530, 531, 556, 557, 558, 559, 560, 561, 562, 563, 590, 591, 592, 593,622, 623, 624, 625 ]
            nlc = len(lcmsk)
            
        else:
            nlp = 0
            nlc = 0


        # Mask of frange
        lfmsk = np.array([
            333, 334, 335, 336, 337, 338,
            365, 366, 367, 368, 369, 370, 371,
                                    403, 404,
            426, 427,                  436, 437,
            458, 459,                  468, 469,
            490, 491,                  500, 501,
            522, 523,                  532, 533,
            554, 555,                  564, 565,
                587, 588,        595, 596,
                619, 620, 621,   626, 627, 628,
                    652, 653, 654, 655, 656, 657, 658, 659,
                                686, 687, 688, 689
        ])
        nlf = len(lfmsk)

        # Mask of frange diameter NS
        lfnsmsk = np.array([
            335, 336,
            367, 368,
            655, 656,
            687, 688
        ])
        nlfns = len(lfnsmsk)

        # Mask of frange diameter EW
        lfewmsk = np.array([
            458, 459, 468, 469,
            490, 491, 500, 501,
            522, 523, 532, 533
        ])
        nlfew = len(lfewmsk)


    elif channel=="C3":
        if rebin_index <= 0:
            lpmsk = np.array([
                0,  1,  2,  3,  4,  5,  6,  7, 24, 25, 26, 27, 28, 29, 30, 31,
                32, 33, 34, 35, 36, 37,                 58, 59, 60, 61, 62, 63,
                64, 65, 66, 67, 68,                        91, 92, 93, 94, 95,
                96, 97, 98, 99,100,                               125,126,127,
                128,129,130,131,132,                                  158,159,
                160,161,                                              190,191,
                192,                                                      223,
                224,
                800,
                832,                                                       863,
                864,865,                                                   895,
                896,897,898,                                           926,927,
                928,929,930,931,                                    957,958,959,
                960,961,962,963,964,                            988,989,990,991,
                992,993,994,995,996,997,           1018,1019,1020,1021,1022,1023
            ])
            nlp = len(lpmsk)

            lcmsk = np.array([
                495, 496,
                527, 528, 529,
                559, 560
            ])
            nlc = len(lcmsk)

        elif rebin_index == 1:
            lpmsk = np.array([
                0,   1,   2,   3,   4,   5,  26,  27,  28,  29,  30,  31,
                32,  33,  34,  35,  36,  37,  58,  59,  60,  61,  62,  63,
                64,  65,  66,  67,                      94,  95,
                96,  97,  98,  99,                     126, 127,
                128, 129,                               158, 159,
                160, 161,                               190, 191,
                896, 897,                               926, 927,
                928, 929,                               958, 959,
                960, 961, 962, 963,             988, 989, 990, 991,
                992, 993, 994, 995,            1020,1021,1022,1023
            ])
            nlp = len(lpmsk)
            nlc = 0
        else:
            lpmsk = np.array([
                0,  1,  2,  3,
                32, 33, 34, 35,
                64, 65, 66, 67,
                96, 97, 98, 99
            ])
            nlp = len(lpmsk)
            nlc = 0

        # Mask of frange
        lfmsk = np.array([
            462, 463, 464, 465,
            494,       497, 498,
            526,            530,
            558,       561, 562,
                591, 592, 593
        ])
        nlf = len(lfmsk)

        # Mask of frange diameter NS
        lfnsmsk = np.array([
            463, 464,
            591, 592
        ])
        nlfns = len(lfnsmsk)

        # Mask of frange diameter EW
        lfewmsk = np.array([
            526, 530
        ])
        nlfew = len(lfewmsk)

    else:
        print(f'%GET_TMASK: WARNING! Unknown channel name: {channel}')
        nlp = nlc = nlf = nlfns = nlfew = 0

    # Apply masks to imsk
    if nlp != 0:
        print(lpmsk)
        lpmskx = lpmsk // 32
        lpmsky = lpmsk % 32
        imsk[lpmsky,lpmskx] = -2
    # if nlc != 0:
    #     imsk[lcmsk] = -1
    # if nlf != 0:
    #     imsk[lfmsk] =  2
    # if nlfns != 0:
    #     imsk[lfnsmsk] =  3
    # if nlfew != 0:
    #     imsk[lfewmsk] =  4


    return imsk
