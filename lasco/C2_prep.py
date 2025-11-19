from .lasco_utils import * 
from astropy.io import fits 
from skimage.transform import warp, estimate_transform
from skimage.transform import resize
import time 

def C2_CALIBRATE(data,header,swdir,no_calfac=False,fuzzy=False):
    valid,exp_factor,exp_bias = GET_EXP_FACTOR(header,swdir)

    header["EXPTIME"] = exp_factor * header["EXPTIME"]
    header["offset"]  = exp_bias

    if no_calfac:
        calfac = 1.0
    else:
        calfac = C2_calfactor(header)


    calibfacdir = swdir+"soho/lasco/calib/"

   

    vig_fn = ''
    vig_fn = 'c2vig_final.fts'
    vig_full = fits.open(calibfacdir + vig_fn)[0].data


    if header["R1COL"]!= 20 or header["R1ROW"] !=1 or header["R2COL"]!= 1043 or header["R2ROW"] !=1024:
        x1 = header["R1COL"]-20
        x2 = header["R2COL"]-20
        y1 = header["R1ROW"]-1
        y2 = header["R2ROW"]-1
        vig  = vig[x1:x2+1,y1:y2+1]
    else:
        vig = vig_full

    
    summsg = 'F'
    summing = (header["sumcol"]>1)*(header["sumrow"]>1)
    if summing>1:
        for i in range(1, summing+1, 4):  
           
            cols = vig.shape[1] // 2
            rows = vig.shape[0] // 2
            
            # Replace REBIN with a proper resizing function
            vig = np.resize(vig, (rows, cols))
   
    
    summing = header["lebxsum"]*header["lebysum"]
    if summing>1:
        for i in range(1, summing+1, 4):  
           
            cols = vig.shape[1] // 2
            rows = vig.shape[0] // 2
            
            # Replace REBIN with a proper resizing function
            vig = resize(vig, (rows, cols))
    
    
    if(header["POLAR"] in ["PB","TI","UP", "JY","JZ","Qs","Us","Qt","Jr","Jt"]):
        data = data/header["EXPTIME"]
        data = data*calfac*vig
        return data 
    else:
        data = (data-exp_bias)*calfac/header["EXPTIME"]
        data = data * vig
   
    return data 



def C2_calfactor(header,NOSUM=False):
    filter = header["FILTER"].replace(" ", "").upper()
    polarizer = header["POLAR"].replace(" ", "").upper()


    dte = datetime.strptime(header["DATE-OBS"],"%Y/%m/%d")
    mjd      = int(Time(dte,format='datetime').mjd)

 
    if filter=="ORANGE":
        cal_factor=0.06047	
        cal_factor=cal_factor=4.60403e-07*mjd+0.0374116
        polref=cal_factor/0.25256		
        if polarizer =='CLEAR':
            cal_factor=cal_factor*1.
        elif polarizer == '+60DEG':	
            cal_factor=polref
        elif polarizer == '0DEG':	
            cal_factor=polref
        elif polarizer == '-60DEG':
            cal_factor=polref
        else:		
            cal_factor=cal_factor*1.
        
    elif filter=="BLUE":
        cal_factor=0.1033	
        polref=cal_factor/.25256		
        if polarizer =='CLEAR':
            cal_factor=cal_factor*1.
        elif polarizer == '+60DEG':	
            cal_factor=polref
        elif polarizer == '0DEG':	
            cal_factor=polref
        elif polarizer == '-60DEG':
            cal_factor=polref
        else:		
            cal_factor=cal_factor*1.

    elif filter=="DEEPRD" :
        cal_factor=0.1033	
        polref=cal_factor/.25256		
        if polarizer =='CLEAR':
            cal_factor=cal_factor*1.
        elif polarizer == '+60DEG':	
            cal_factor=polref
        elif polarizer == '0DEG':	
            cal_factor=polref
        elif polarizer == '-60DEG':
            cal_factor=polref
        else:		
            cal_factor=0.0

    elif filter=="HALPHA" :
        cal_factor=0.01055		
        polref=cal_factor/.25256		
        if polarizer =='CLEAR':
            cal_factor=cal_factor*1.
        elif polarizer == '+60DEG':	
            cal_factor=polref
        elif polarizer == '0DEG':	
            cal_factor=polref
        elif polarizer == '-60DEG':
            cal_factor=polref
        else:		
            cal_factor=cal_factor*1.

    elif filter=="LENS" :
        cal_factor=0.01055		
        polref=cal_factor/.25256		
        if polarizer =='CLEAR':
            cal_factor=cal_factor*1.
        elif polarizer == '+60DEG':	
            cal_factor=polref
        elif polarizer == '0DEG':	
            cal_factor=polref
        elif polarizer == '-60DEG':
            cal_factor=polref
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



def c2_distortion(r,scalef):
    mm = r*0.021
    cf = [0.0051344125,-0.00012233862,1.0978595e-7]
    f1 = mm*(cf[0]+cf[1]* mm**2 + cf[2]*mm**4)
    f1 = r + f1/0.021

    return substense('c2') * f1 
    

def  C2_warp(data,header):
    gridsize = 32
    image_size = header["NAXIS1"]
    n_cells = (image_size // gridsize) +1  

    w = np.arange(n_cells ** 2)
    
    y = w // n_cells
    x = w - y * n_cells  # same as w % n_cells

    x = x * gridsize
    y = y * gridsize



    xc, yc = sun_center(header)


    sumx = header["lebxsum"] * max(header["sumcol"], 1)
    sumy = header["lebysum"] * max(header["sumrow"], 1)

  

    if ( sumx > 0 ):
        x = x/sumx
        xc = xc/sumx
     
    if ( sumy > 0 ):
        y = y/sumy
        yc = yc/sumy
      

    


    scalef = get_sec_pixel(header)
    r= np.sqrt((sumx*(x-xc))**2+(sumy*(y-yc))**2)
    

    r0= c2_distortion(r,scalef) / (sumx * scalef)


    theta= np.arctan2((y-yc),(x-xc))
    x0= r0*np.cos(theta)+xc
    y0= r0*np.sin(theta)+yc


    

    tform = estimate_transform('affine',np.column_stack([x,y]), np.column_stack([x0,y0]))
    # Apply warp
    warped = warp(data, tform.inverse, output_shape=(image_size,image_size),order=1)
    

    return warped