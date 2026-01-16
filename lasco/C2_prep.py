from .lasco_utils import * 
from astropy.io import fits 
from skimage.transform import warp, estimate_transform
from skimage.transform import resize
import matplotlib.pyplot as plt 
import cv2


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
        vig  = vig[y1:y2+1,x1:x2+1]
    else:
        vig = vig_full
    
   

    
    summsg = 'F'
    summing = max(header["sumcol"],1)*max(header["sumrow"],1)
    if summing>1:
        for i in range(1, summing+1, 4):  
           
            cols = vig.shape[1] // 2
            rows = vig.shape[0] // 2
            
            # Replace REBIN with a proper resizing function
            vig = np.resize(vig, (rows, cols),preserve_range=True)
   
    
    summing = header["lebxsum"]*header["lebysum"]
    if summing>1:
        for i in range(1, summing+1, 4):  
           
            cols = vig.shape[1] // 2
            rows = vig.shape[0] // 2
            
            # Replace REBIN with a proper resizing function
            vig = resize(vig, (rows, cols),preserve_range=True)
    
    
    if(header["POLAR"] in ["PB","TI","UP", "JY","JZ","Qs","Us","Qt","Jr","Jt"]):
        data = data/header["EXPTIME"]
        data = data*calfac*vig
        return data 
    else:
        try:
            data = (data-exp_bias)*calfac/header["EXPTIME"]
           
            data = data * vig
        except:
            print(data,exp_bias)
            exit()
   
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



def c2_distortion(r,arcs=None):
    mm = r*0.021
    cf = [0.0051344125,-0.00012233862,1.0978595e-7]
    f1 = mm*(cf[0]+cf[1]* mm**2 + cf[2]*mm**4)
    f1 = r + f1/0.021
    if arcs is not None:
        return arcs*f1 
    
    else:
        return substense('c2') * f1 



def  C2_warp(data,header):
    gridsize = 32
    image_size = header["NAXIS1"]
    # n_cells = (image_size // gridsize) +1  

    # w = np.arange(n_cells ** 2)
    
    # y = w // n_cells
    # x = w - y * n_cells  # same as w % n_cells

    # x = x * gridsize
    # y = y * gridsize
    w = np.arange(33 * 33)

    y = w // 33          # IDL integer division
    x = w - y * 33

    x = x * 32
    y = y * 32


    # image = reduce_std_size(image0,hdr, /no_rebin, /nocal)
    data,header = reduce_std_size(data,header,nocal=True,norebin=True,full=False)

    # try:
    #     xc, yc = sun_center(header)
       
    # except:
    #     return None 
    
    #shortcut for occltr_cntr.pro that is used in current idl code 
    xc,yc = 512.634,505.293
    
    if xc<0 or yc<0:
        xc = 512.634
        yc = 505.293


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

    


    

    # tform = estimate_transform('affine',np.column_stack([x,y]), np.column_stack([x0,y0]))
    # # Apply warp
    # warped = warp(data, tform.inverse, output_shape=(image_size,image_size),order=1)

     # start = time.time()
    tfm, _ = cv2.estimateAffinePartial2D(np.vstack([x,y]).T, np.vstack([x0,y0]).T)
    # print(" compute transform2",time.time()-start)
    # start = time.time()
    warped = cv2.warpAffine(data, tfm, (data.shape[0],data.shape[1]))

    # if image_size==512:
    #     print(header["NAXIS1"],header["NAXIS2"],header["cdelt1"],header["cdelt2"])
  
    #     plt.scatter(y0,x0)
    #     plt.scatter(y,x)
    #     plt.show()
    #     fig,ax=plt.subplots(1,2)
    #     ax[0].imshow(data)
    #     ax[1].imshow(warped)
    #     plt.show()
    

    return warped