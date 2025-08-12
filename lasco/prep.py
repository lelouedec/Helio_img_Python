from natsort import natsorted 
import glob 
from astropy.io import fits 
from .C3_prep import * 
import matplotlib.pyplot as plt 
import matplotlib

matplotlib.use("Qt5Agg")



def lasco_prep(dates,path,swdir,ins):

    for d in dates:

        files_list = natsorted(glob.glob(path.replace("L1","L0")+str(d)+"/*"))
        for f in files_list:
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
                data =  C3_CALIBRATE(data,header,swdir)
                data = C3_warp(data,header)

                xnorm = 518.0		# IDL coordinates
                ynorm = 531.5		# nbr, 27Jul00
                mbstrings=1
                plt.imshow(find_miss_blocks(data,header))
                plt.show()
            

            # exit()


# // camera= strupcase(strtrim(hdr.detector,2))


