from natsort import natsorted 
import glob 
from astropy.io import fits 
from .C3_prep import * 
from .C2_prep import * 

import matplotlib.pyplot as plt 
import matplotlib
from skimage.transform import resize
import time 

matplotlib.use("Qt5Agg")



def lasco_prep(dates,path,swdir,ins):
    calibfacdir = swdir+"soho/lasco/calib/"

    if not os.path.exists(calibfacdir):
        os.makedirs(calibfacdir)

    num_cpus = cpu_count()
    io_utils.multi_process_dl(num_cpus,"https://hesperia.gsfc.nasa.gov/ssw/soho/lasco/lasco/data/calib/",".dat",calibfacdir)
    io_utils.multi_process_dl(num_cpus,"https://hesperia.gsfc.nasa.gov/ssw/soho/lasco/lasco/data/calib/",".fts",calibfacdir)
    io_utils.multi_process_dl(num_cpus,"https://hesperia.gsfc.nasa.gov/ssw/soho/lasco/lasco/data/calib/",".txt",calibfacdir)

    

    for d in dates:
        savepath = path + d

        files_list = natsorted(glob.glob(path.replace("L1","L0")+str(d)+"/*"))
        for f in files_list:
            print(f)
            hdul = fits.open(f)
            data = hdul[0].data
            header = hdul[0].header

            xsumming = (header["SUMCOL"]>1)*(header["LEBXSUM"]>1)
            ysumming = (header["SUMROW"]>1)*(header["LEBYSUM"]>1)
            summing = xsumming*ysumming

            if(summing>1):
                print("do something with it, fixwrap and dofull=0")



            if header["R2COL"] - header["R1COL"] + header["R2ROW"] - header["R1ROW"] - 1023-1023 !=0:
                #a = reduce_std_size(a,header,full=dofull)
                continue 
            fname = header["filename"]
            source = fname[1:2] 
            root = fname.split(".")[0]
            yymmdd = header["DATE-OBS"][2:4]+header["DATE-OBS"][5:7]+header["DATE-OBS"][8:10]
            

            # Replace character at position 1 in root depending on source
            if source == '1':
                root = root[:1] + '4' + root[2:]
            elif source == '2':
                root = root[:1] + '5' + root[2:]
            elif source == '3':
                root = root[:1] + '4' + root[2:]

            # For monthly images
            if source in ('m', 'd'):
                print("monthly stuff exiting ")
                # if (strmid(root,2,1) EQ 'r') THEN 
                #     root = strmid(root,0,3)+'1'+strmid(root,3,13) $
                # ELSE root = strmid(root,0,2)+'1'+strmid(root,2,12) 
                #     yymmdd = 'monthly'
                #     hdr.r1col=20
                #     hdr.r2col=1043
                #     hdr.r1row=1
                #     hdr.r2row=1024
                exit()

            outname=root+'.fts'

            ## here remove the history from the file, not sure how to make it work yet skipping it 
            # histlen = 1
            # cmntlen = 1  # unused in your snippet but keeping for parity
            # inc = 0

            # # Loop until an empty history entry (length 0)
            # while histlen > 0:
            #     hist_entry = header['HISTORY'][inc]
            #     histlen = len(hist_entry)
            #     print(hist_entry)
            #     # If entry length > 2 and does NOT contain 'bias'
            #     if histlen > 2 and 'bias' not in hist_entry:
            #         # Add to FITS header (fxaddpar equivalent in astropy.io.fits is .append or .add_history)
            #         header.add_history(' ' + hist_entry.strip())

            #     inc += 1

            if ins=="C3":
                if data.shape[0]==1024:
                    data =  C3_CALIBRATE(data,header,swdir)
                    data = C3_warp(data,header)

                    # xnorm = 518.0		# IDL coordinates
                    # ynorm = 531.5		# nbr, 27Jul00
                    # mbstrings=1
                    # plt.imshow(find_miss_blocks(data,header))
                    # plt.show()
                    # print(path+header["filename"])
                    # exit()
                
                    for i in range(0,len(header["history"])):
                        header["history"][i] = header["history"][i].replace("\t","")
                
                    if not os.path.exists(savepath):
                        os.makedirs(savepath)
                    fits.writeto(savepath+"/"+outname,data.astype(np.float32), header, output_verify='ignore', overwrite=True)
            elif ins=="C2":
                data =  C2_CALIBRATE(data,header,swdir)
                data = C2_warp(data,header)
                # name_mask = get_cal_name('C3_cl*msk*.dat',yymmdd,swdir)
                # mask_hdul = fits.open(name_mask)
                # mask_data = mask_hdul[0].data
                # mask_data = C2_warp(mask_data,header)
               
                zz = np.where(data <= 0)
                maskall = np.ones((header["NAXIS1"],header["NAXIS2"]))
                print(maskall.shape)
                # Check if there is more than one index
                if zz[0].size > 1:     
                    maskall[zz] = 0.0 

                cols = data.shape[1] // 32
                rows = data.shape[0] // 32
                
                # Replace REBIN with a proper resizing function
                data2 = resize(data, (rows, cols))
                zblocks = np.where(data2 <= 0)
                if zblocks[0].size>0:
                    if summing >1:
                        nmissing = zblocks[0].size - 8
                    else:
                        nmissing = zblocks[0].size
                if(nmissing>17):
                    # maskall = np.ones((32,32))
                    # maskall[zblocks] = 0.0
                    # maskall = resize(maskall,(1024,1024))
                    maskall = C2_warp(maskall,header)
                    maskall[maskall<0] = 0.0
                data = data*maskall
                if not os.path.exists(savepath):
                    os.makedirs(savepath)
                fits.writeto(savepath+"/"+outname,data.astype(np.float32), header, output_verify='ignore', overwrite=True)



# // camera= strupcase(strtrim(hdr.detector,2))


