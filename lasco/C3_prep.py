import os 
import io_utils
from multiprocessing import cpu_count
import numpy as np 
from datetime import datetime
from .lasco_utils import * 
from astropy.time import Time
from astropy.io import fits 






 
def C3_calfactor(header,NOSUM=False):
    filter = header["FILTER"].replace(" ", "").upper()
    polarizer = header["POLAR"].replace(" ", "").upper()


    dte = datetime.strptime(header["DATE-OBS"],"%Y/%m/%d")
    mjd      = int(Time(dte,format='datetime').mjd)

 
    if filter=="ORANGE":
        cal_factor=0.0297	
        polref=cal_factor/0.25256		
        if polarizer =='CLEAR':
            cal_factor=cal_factor*1.
        elif polarizer == '+60DEG':	
            cal_factor=polref
        elif polarizer == '0DEG':	
            cal_factor=polref*0.9648
        elif polarizer == '-60DEG':
            cal_factor=polref*1.0798
        else:		
            cal_factor=cal_factor*1.
        
    elif filter=="BLUE":
        cal_factor=0.0975	
        polref=cal_factor/.25256		
        if polarizer =='CLEAR':
            cal_factor=cal_factor*1.
        elif polarizer == '+60DEG':	
            cal_factor=polref
        elif polarizer == '0DEG':	
            cal_factor=polref*0.9734
        elif polarizer == '-60DEG':
            cal_factor=polref*1.0613
        else:		
            cal_factor=cal_factor*1.

    elif filter=="CLEAR" :
        cal_factor=7.43e-8*(mjd-50000)+5.96e-3	
        polref=cal_factor/.25256		
        if polarizer =='CLEAR':
            cal_factor=cal_factor*1.
        elif polarizer == '+60DEG':	
            cal_factor=polref
        elif polarizer == '0DEG':	
            cal_factor=polref*0.9832
        elif polarizer == '-60DEG':
            cal_factor=polref*1.0235
        elif polarizer =='H_ALPHA':		
            cal_factor=cal_factor*1.541
        else:
            cal_factor=0.0


    elif filter=="DEEPRD" :
        cal_factor=0.0259	
        polref=cal_factor/.25256		
        if polarizer =='CLEAR':
            cal_factor=cal_factor*1.
        elif polarizer == '+60DEG':	
            cal_factor=polref
        elif polarizer == '0DEG':	
            cal_factor=polref*0.9983
        elif polarizer == '-60DEG':
            cal_factor=polref*1.0300
        else:		
            cal_factor=cal_factor*1.

    elif filter=="IR" :
        cal_factor=0.0887	
        polref=cal_factor/.25256		
        if polarizer =='CLEAR':
            cal_factor=cal_factor*1.
        elif polarizer == '+60DEG':	
            cal_factor=polref
        elif polarizer == '0DEG':	
            cal_factor=polref*0.9833
        elif polarizer == '-60DEG':
            cal_factor=polref*1.0288
        else:		
            cal_factor=cal_factor*1.
    else:
        cal_factor = 0.0        

    if NOSUM:
        if (header["sumcol"] > 0)  : cal_factor=cal_factor/header["sumcol"]
        if (header["sumrow"] > 0)  : cal_factor=cal_factor/header["sumrow"]
        if (header["lebxsum"] > 1)  : cal_factor=cal_factor/header["lebxsum"]
        if (header["lebysum"] > 1)  : cal_factor=cal_factor/header["lebysum"]

    cal_factor = cal_factor*1.e-10
    return cal_factor


def C3_CALIBRATE(data,header,swdir,no_calfac=False,fuzzy=False):
    valid,exp_factor,exp_bias = GET_EXP_FACTOR(header,swdir)
    header["EXPTIME"] = exp_factor * header["EXPTIME"]
    header["offset"]  = exp_bias


    if no_calfac:
        calfac = 1.0
    else:
        calfac = C3_calfactor(header)
    print(calfac)

    dte = datetime.strptime(header["DATE-OBS"],"%Y/%m/%d")
    mjd      = int(Time(dte,format='datetime').mjd)
   
    # lets download all the file if not found :
    # 
    calibfacdir = swdir+"soho/lasco/calib/"

    if not os.path.exists(calibfacdir):
        os.makedirs(calibfacdir)

    num_cpus = cpu_count()
    io_utils.multi_process_dl(num_cpus,"https://hesperia.gsfc.nasa.gov/ssw/soho/lasco/lasco/data/calib/",".dat",calibfacdir)
    io_utils.multi_process_dl(num_cpus,"https://hesperia.gsfc.nasa.gov/ssw/soho/lasco/lasco/data/calib/",".fts",calibfacdir)

    vig_fn = ''
    if mjd<51000:
        vig_fn = 'c3vig_preint_final.fts' 
    else:
        vig_fn = 'c3vig_postint_final.fts' 

    if(vig_fn != '') :
        vig_full = fits.open(calibfacdir + vig_fn)[0].data

    msk_fn='c3_cl_mask_lvl1.fts'        # KB July 28, 2005
    mask_full  = fits.open(calibfacdir + msk_fn)[0].data
    ramp_full = fits.open(calibfacdir + 'C3ramp.fts')[0].data
    vig = vig_full

    if header["R1COL"]!= 20 or header["R1ROW"] !=1 or header["R2COL"]!= 1043 or header["R2ROW"] !=1024:
        x1 = header["R1COL"]-20
        x2 = header["R2COL"]-20
        y1 = header["R1ROW"]-1
        y2 = header["R2ROW"]-1
        vig  = vig[x1:x2+1,y1:y2+1]
        ramp = ramp_full[x1:x2+1,y1:y2+1]
        mask = mask_full[x1:x2+1,y1:y2+1]

    else:
        ramp = ramp_full
        mask = mask_full


    summsg = 'F'
    summing = (header["sumcol"]>1)*(header["sumrow"]>1)
    if summing>1:
        for i in range(1, summing+1, 4):  
           
            cols = vig.shape[1] // 2
            rows = vig.shape[0] // 2
            
            # Replace REBIN with a proper resizing function
            vig = np.resize(vig, (rows, cols))
            ramp = np.resize(ramp, (rows, cols))
            mask = np.resize(mask, (rows, cols))
            bkg = np.resize(bkg, (rows, cols))


    if header["FILEORIG"] == 0: #monthly image
        data = data/header["exptime"]
        data = data*calfac*vig - ramp
        return data 

    if (header["FILTER"] != 'Clear'):
        ramp = 0 

    if(header["POLAR"] in ["PB","TI","UP", "JY","JZ","Qs","Us","Qt","Jr","Jt"]):
        data = data/header["EXPTIME"]
        data = data*calfac*vig
        return data 
    else:
        zz = np.where(data.flatten()<0)[0]
        
        img = (data-exp_bias)/header["EXPTIME"]

        if fuzzy and header["FILTER"]=='Clear' and zz.shape[0]>1000:
            print("fuzzy not implemented lol ")
        
        if img.shape == vig.shape:
            img = img *vig*calfac - ramp 
            img = img * mask 
    
    return img 




def c3_distortion(r,scalef):

    mm = r*.021	


    cf=[-0.0151657,0.000165455,0.0] # hardcoding the ceofs here because easier 


    f1 = mm*(cf[0]+cf[1]* mm**2)
    f1 = r + f1/.021


    secs = substense('c3')	
    return secs * f1

def C3_warp(data,header):
    gridsize = 32
    image_size = 1024
    n_cells = (image_size // gridsize) +1  

    w = np.arange(n_cells ** 2)
    
    y = w // n_cells
    x = w - y * n_cells  # same as w % n_cells

    x = x * gridsize
    y = y * gridsize



    xc, yc = sun_center(header)


    x1 = header["R1COL"]-20
    x2 = header["R2COL"]-20
    y1 = header["R1ROW"]-1
    y2 = header["R2ROW"]-1

    sumx = header["lebxsum"] * max(header["sumcol"], 1)
    sumy = header["lebysum"] * max(header["sumrow"], 1)

  

    if ( sumx > 1 ):
        x = x/sumx
        xc = xc/sumx
        x1 = x1/sumx
        x2 = x2/sumx
    if ( sumy > 1 ):
        y = y/sumy
        yc = yc/sumy
        y1 = y1/sumx
        y2 = y2/sumx

    


    scalef = get_sec_pixel(header)
    r= np.sqrt((sumx*(x-xc))**2+(sumy*(y-yc))**2)
    

    r0= c3_distortion(r,scalef) / (sumx * scalef)


    theta= np.arctan2((y-yc),(x-xc))
    x0= r0*np.cos(theta)+xc
    y0= r0*np.sin(theta)+yc


    
    from skimage.transform import warp, estimate_transform

    tform = estimate_transform('affine',np.column_stack([x,y]), np.column_stack([x0,y0]))
    # Apply warp
    warped = warp(data, tform.inverse, output_shape=(1024,1024),order=1)
    
    warped = warped[x1:x2,y1:y2]

    return warped
