from datetime import datetime,timedelta
from .secchi_functions import * 
import  matplotlib.pyplot as plt 


def cor_prep(data,header,post_conj, calpath, pointpath,ftpsc):

#Cosmic Ray and Star Removal(OFF)


    ## chose mean, probably not the best choice but fuck it
    missing = scc_get_missing(header)
    if(header['MISSLIST'] != ''):
        data[missing] = np.nanmean(data)

    
    #COR2 calibrate
    data,header = cor_calibrate(data,header,post_conj,calpath)

    data =  cor2_warp(data,header,ftpsc)
    # data,header = scc_roll_image(data,header)

    smask = get_smask(header, calpath, post_conj, silent=True)
    data = smask * data


    if header["DETECTOR"] == 'COR2':
        header["CDELT1"] = 14.7*(2**(header["SUMMED"]-1))
        header["CDELT2"] = 14.7*(2**(header["SUMMED"]-1))

   
    return data, header

def cor_calibrate(data,header,post_conj,calpath,exptime_off=False,bias_off=False,calimg_off=False,calfac_off=False):

    data,header = scc_sebip(data,header)
    if(exptime_off):
        exptime = 1.0
    else:
        exptime = header["EXPTIME"]

    if exptime != 1.0:
        header['HISTORY'] = 'Exposure Normalized to 1 Second '+ str(exptime)
    
    if bias_off:
        bias = 0.0
    else:
        bias = get_biasmean(header)

    if bias != 0.0:
        header['HISTORY'] = 'Bias Subtracted '+ str(bias)
        header["OFFSETCR"]=bias

    if calimg_off:
        calimg = 1.0
    else:
        calimg,cal_version = get_calimg(header,calpath,post_conj)
    if(calimg.shape[0]>1):
        header['HISTORY'] = 'Applied Vignetting '+ str(bias)

    if calfac_off:
        calfac = 1.0
    else:
        calfac,header = get_calfac(header)
    
    if calfac != 1.0:
        header['HISTORY'] = 'Applied Calibration Factor '+ str(calfac)
    
    data = (data - bias)*calfac/exptime * calimg
    
    return data,header

def update_polariz_header(hdrs,polval,taiend,date_avg):
    hdr = hdrs[0].copy()
    hdr['POLAR'] = polval
    hdr['N_IMAGES'] = 3
    hdr['CROTA'] = np.mean([h['CROTA'] for h in hdrs])
    angle = hdr['CROTA'] * (np.pi / 180)
    hdr['PC1_1'] = np.cos(angle)
    hdr['PC1_2'] = -np.sin(angle)
    hdr['PC2_1'] = np.sin(angle)
    hdr['PC2_2'] = np.cos(angle)
    
    # -- Update other header fields as averages/totals
    hdr['expcmd'] = np.mean([h['expcmd'] for h in hdrs])
    hdr['exptime'] = -1
    hdr['biasmean'] = np.mean([h['biasmean'] for h in hdrs])
    hdr['biassdev'] = np.max([h['biassdev'] for h in hdrs])
    hdr['ceb_t'] = np.mean([h['ceb_t'] for h in hdrs])
    hdr['temp_ccd'] = np.mean([h['temp_ccd'] for h in hdrs])
    hdr['readtime'] = np.mean([h['readtime'] for h in hdrs])
    hdr['cleartim'] = np.mean([h['cleartim'] for h in hdrs])
    hdr['ip_time'] = np.sum([h['ip_time'] for h in hdrs])
    hdr['compfact'] = np.mean([h['compfact'] for h in hdrs])
    hdr['nmissing'] = np.sum([h['nmissing'] for h in hdrs])
    hdr['tempaft1'] = np.mean([h['tempaft1'] for h in hdrs])
    hdr['tempaft2'] = np.mean([h['tempaft2'] for h in hdrs])
    hdr['tempmid1'] = np.mean([h['tempmid1'] for h in hdrs])
    hdr['tempmid2'] = np.mean([h['tempmid2'] for h in hdrs])
    hdr['tempfwd1'] = np.mean([h['tempfwd1'] for h in hdrs])
    hdr['tempfwd2'] = np.mean([h['tempfwd2'] for h in hdrs])
    hdr['tempthrm'] = np.mean([h['tempthrm'] for h in hdrs])
    hdr['temp_ceb'] = np.mean([h['temp_ceb'] for h in hdrs])

    
    hdr['DATE_END'] = datetime.strftime(taiend, '%Y-%m-%dT%H:%M:%S.%f')
    hdr['DATE_AVG'] = datetime.strftime(date_avg, '%Y-%m-%dT%H:%M:%S.%f') 

    return hdr
    

def cor_polariz_python(hdrs, images):
    # -- History info from IDL source
    histinfo = "$Id: cor_polariz.pro,v 1.25 2012/06/26 21:37:23 nathan Exp $"[:-2]
    
    ## images should be ordered from 0 angle to 240 

    taiobs=[datetime.strptime(hdrs[i]["DATE-OBS"], '%Y-%m-%dT%H:%M:%S.%f')   for i in range(0,len(hdrs))]


   
    # -- Ensure unique polarizer angles
    pols = [h['POLAR'] for h in hdrs]
    if len(set(pols)) < 3:
        raise ValueError("IMAGES MUST HAVE 3 DIFFERENT POLARIZATION ANGLES")
    
    # # -- Check consistent OBS_ID
    obs_ids = [h['OBS_ID'] for h in hdrs]
    # if len(set(obs_ids)) > 1:
    #         print("IMAGES NOT FROM SAME OBS SEQUENCE: OBS_ID =", obs_ids)
    

   

    angle1, angle2,angle3 = pols[0],pols[1],pols[2]

    im1 = images[0]
    im2 = images[1]
    im3 = images[2]


    # print(angle1,angle2,angle3)
    matrix = np.array([
            [0.5,0.5*np.cos(2.0*np.deg2rad(angle1)),0.5*np.sin(2.0*np.deg2rad(angle1))],
            [0.5,0.5*np.cos(2.0*np.deg2rad(angle2)),0.5*np.sin(2.0*np.deg2rad(angle2))],
            [0.5,0.5*np.cos(2.0*np.deg2rad(angle3)),0.5*np.sin(2.0*np.deg2rad(angle3))],
    ])
    conv_matrix_inv = np.linalg.inv(matrix)



    I = im1*conv_matrix_inv[0,0] + im2*conv_matrix_inv[0,1] + im3*conv_matrix_inv[0,2]
    Q = im1*conv_matrix_inv[1,0] + im2*conv_matrix_inv[1,1] + im3*conv_matrix_inv[1,2] 
    U = im1*conv_matrix_inv[2,0] + im2*conv_matrix_inv[2,1] + im3*conv_matrix_inv[2,2]

    pbim = np.sqrt((Q**2) + (U**2))
    B  = I 


     # # -- Compute date_end and date_avg
    taiend = taiobs[2] + timedelta(seconds=hdrs[2]['exptime'])
    date_avg = taiobs[0] + (taiend - taiobs[0]) / 2
   

    
    hdr_pb = update_polariz_header(hdrs,1002,taiend,date_avg)
    hdr_B = update_polariz_header(hdrs,1001,taiend,date_avg)

    ### think about a way to integrate these maybe later 
    # mu =  0.5 * np.arctan2(U, Q)
    # polval_mu = 1004

  
    # ppimage = 100. * (pbim / I)
    # if PERCENT:
    #     if not silent: print("Returning PERCENT POLARIZED")
    #     im = ppimage; polval = 1003; fnstr = 'p'
   
    
    return pbim,B, hdr_pb,hdr_B

def cor_calfac(data,header):
    calfac_off = True
    for i in range(0,len(data)): 
        if calfac_off:
            sumcount = header[i]['ipsum'] - 1
            divfactor = (2 ** sumcount) ** 2
            data[i] = data[i] / divfactor

            if header[i]['ipsum'] > 1:
                
                
                header[i]['history'] = f'image Divided by {divfactor} to account for IPSUM'
            
            header[i]['ipsum'] = 1
            header[i]['bunit'] = header[i]['bunit'] + '/CCDPIX'

        
        header[i] = scc_update_hdr(data[i], header[i])

        calfac,header[i] = get_calfac(header[i],'MSB')
        
        ## from HI and RAL not sure here what value it should be 
        # calfac = calfac*2.223e15

        data[i] = data[i] * calfac  

    return data, header

def sun_center(header):
    
    wcsh= wcs.WCS(header)

    # scale = (keyword_set(FULL) ? float(full) / hdr0.NAXIS1 : 1.0)

    x, y = wcsh.all_world2pix([0], [0], 0)
    sxcen = x*1.0
    sycen = y*1.0
    return sxcen, sycen 



def cor2_warp(data,header,sc):
    gridsize = 32
    image_size = 2048
    n_cells = (image_size // gridsize) +1   # Same as ((2048/32)+1) = 65

    w = np.arange(n_cells ** 2)
    
    y = w // n_cells
    x = w - y * n_cells  # same as w % n_cells

    x = x * gridsize
    y = y * gridsize



    xc, yc = sun_center(header)
    sumxy = 2 ** (int(header["SUMMED"]) - 1)
 
    # Compute distance from the sun center
    r = np.sqrt((x - sumxy * xc) ** 2 + (y - sumxy * yc) ** 2)


    if (sc == 'A'):
        cf = [1.04872e-05, -0.00801293, -0.243670]
    else:
        cf = [1.96029e-05, -0.0201616,   4.68841] 
        
    r0 = r + (cf[2]+(cf[1]*r)+(cf[0]*(r*r)))    # apply distortion function

    r0 = r0 / sumxy
    x = x / sumxy
    y = y / sumxy

    # # Polar to Cartesian transformation
    theta = np.arctan2(y - yc, x - xc)
    x0 = r0 * np.cos(theta) + xc
    y0 = r0 * np.sin(theta) + yc


    from skimage.transform import warp, estimate_transform

    tform = estimate_transform('affine',np.vstack([x,y]).T, np.vstack([x0,y0]).T)
    # Apply warp
    warped = warp(data, tform.inverse, output_shape=(2048,2048))

    return warped
