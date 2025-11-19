from datetime import datetime
import numpy as np 
from .secchi_functions import * 
import sys 
from .hi_functions import * 
from .cor_functions import * 
import tqdm
import glob 
import os 
import shutil
from astropy.io import fits 
from collections import Counter
from natsort import natsorted 
from scipy.ndimage import shift
import cv2 


def ignore_extended_attributes(func, filename, exc_info):
    is_meta_file = os.path.basename(filename).startswith("._")
    if not (func is os.unlink and is_meta_file):
        raise

def reduction(hdul,hdul_data,hdul_header,ftpsc,ins,bflag,calpath,pointpath,silent=False,polarized=False):
        
        for i in range(len(hdul_data)):
            hdul_header[i] = fix_secchi_hdr(hdul_header[i])

        ## CHANGE rectify inserted here
        for i in range(len(hdul_data)):
            if hdul_header[i]['rectify'] != True:
                hdul_header[i]['r1col'] = hdul_header[i]['p1col']
                hdul_header[i]['r2col'] = hdul_header[i]['p2col']
                hdul_header[i]['r1row'] = hdul_header[i]['p1row']
                hdul_header[i]['r2row'] = hdul_header[i]['p2row']

                hdul_data[i], hdul_header[i] = secchi_rectify(hdul_data[i], hdul_header[i])

        rectify_on =  True
        precomcorrect_on = False

     

        if precomcorrect_on == False:

            if ftpsc == 'A':
                date_cutoff = datetime.strptime('2007-02-03T13:15', '%Y-%m-%dT%H:%M')
            else:
                date_cutoff = datetime.strptime('2007-02-21T21:00', '%Y-%m-%dT%H:%M')

            precomcorrect_on = (ins == 'cor1') and (hdul_header[0]['date-obs'] < date_cutoff) and (hdul_header[0]['date'] < datetime.strptime('2008-01-17', '%Y-%m-%d'))

        if precomcorrect_on == True:

            for i in range(len(hdul_data)):
                xh = hdul[i][1].header
                cnt_exp = np.where(xh['EXPTIME'] == 0)[0]

                if len(cnt_exp) <= 0:
                    hdul_header[i]['EXPTIME'] = np.sum(xh['EXPTIME'])

                hdul_data[i], hdul_header[i] = precommcorrect(hdul_data[i], hdul_header[i], silent=silent)
        
        if bflag == 'science':
            if ins == 'hi_1':
                norm_img = 30
                acc_img = 15

            else:
                norm_img = 99
                acc_img = 99

        if bflag == 'beacon' :
            norm_img = 1
            acc_img = 1

        if ins=="cor2":
            norm_img = 3
            acc_img = 3
            
        indices = np.arange(len(hdul_data))
        bad_img = []
    
        n_images = [hdul_header[i]['n_images'] for i in range(len(hdul_header))]

        if not all(val == norm_img for val in n_images):

            bad_ind = [i for i in range(len(n_images)) if (n_images[i] != norm_img) and (n_images[i] != acc_img)]
            bad_img+=bad_ind

 
        crval1_test = [int(np.sign(hdul_header[i]['crval1'])) for i in range(len(hdul_header))]

        if len(set(crval1_test)) > 1 and ins!="cor2":

            common_crval = Counter(crval1_test)
            com_val, count = common_crval.most_common()[0]
            
            bad_ind = [i for i in range(len(crval1_test)) if crval1_test[i] != com_val]
            
            bad_img += bad_ind
                
            if len(bad_ind) >= len(indices):
                print('Too many corrupted images - can\'t determine correct CRVAL1. Exiting...')
                sys.exit()
   
        if bflag == 'science' and ins!="cor2":
            #Must find way to do this for beacon also
            datamin_test = [hdul_header[i]['DATAMIN'] for i in range(len(hdul_header))]
            
            if not all(val == norm_img for val in datamin_test):
                
                bad_ind = [i for i in range(len(datamin_test)) if datamin_test[i] != norm_img]
                bad_img += bad_ind

        if bflag == 'beacon' and ins!="cor2":
            test_data = np.array([hdul_data[i] for i in range(len(hdul_header))])
            test_data = np.where(test_data == 0, np.nan, test_data)
            
            bad_ind = [i for i in range(len(test_data)) if np.isnan(test_data[i]).all() == True]
            bad_img += bad_ind     

        if bflag == 'beacon' and ins!="cor2" :
            num_zero = np.array([hdul_header[i]['DATAZER'] for i in range(len(hdul_header))])

            num_pixels = np.array([hdul_header[i]['NAXIS1'] * hdul_header[i]['NAXIS2'] for i in range(len(hdul_header))])
            zero_percentage = num_zero / num_pixels

            # for i in range(len(zero_percentage)):
            #     if zero_percentage[i] > 0.05:
            #         print('', hdul_header[i]['DATE-OBS'], ' has ', np.round(zero_percentage[i],2), ' % pixels with value 0.0')
            #         plt.imshow(hdul_data[i], cmap='gray')
            #         plt.title(hdul_header[i]['DATE-OBS'])
            #         plt.show()

            bad_ind = [i for i in range(len(zero_percentage)) if np.round(zero_percentage[i],2) > 0.34]
            bad_img += bad_ind

        base = 34
        misslist_str = [hdul_header[i]['MISSLIST'] for i in range(len(hdul_header))]
        len_misslist = [len(misslist_str[i]) for i in range(len(misslist_str))]
        
        for i in range(len(len_misslist)):
            if len_misslist[i] % 2 != 0:
                misslist_str[i] = ' ' + misslist_str[i]
                len_misslist[i] += 1

        dex = [np.arange(0, len_misslist[i], 2).astype(int) for i in range(len(len_misslist))]

        misslist = [[int(misslist_str[j][i:i+2].strip(), base) for i in dex[j]] for j in range(len(misslist_str))]

        n = [len(misslist[i]) for i in range(len(misslist))]

        bad_ind = [i for i in range(len(n)) if n[i] != hdul_header[i]['NMISSING']]

        bad_img += bad_ind

        indices = list(np.setdiff1d(indices, np.array(bad_img)))

        clean_data = []
        clean_header = []
        
        for i in range(len(hdul)):
            if i in indices:
                clean_data.append(hdul_data[i])
                clean_header.append(hdul_header[i])
                hdul[i].close()
            else:
                hdul[i].close()
        
        clean_data = np.array(clean_data, dtype=np.float32)

        if bflag == 'beacon':
            for i in range(len(clean_header)):
                hi_fix_beacon_date(clean_header[i])

        crval1 = [clean_header[i]['crval1'] for i in range(len(clean_header))]

        if ftpsc == 'A':    
            post_conj = [int(np.sign(crval1[i])) for i in range(len(crval1))]
    
        if ftpsc == 'B':    
            post_conj = [int(-1*np.sign(crval1[i])) for i in range(len(crval1))]
        
        if len(clean_header) == 0:
            print('No clean files found for ', ins, ' on ',  hdul_header[len(hdul_header)-1]["DATE-END"][:4])
            return [], []

        if(ins!='cor2'):
            if len(set(post_conj)) == 1 :

                post_conj = post_conj[0]
        
                if post_conj == -1:
                    post_conj = False
                if post_conj == 1:
                    post_conj = True

            else:
                print('Corrupted CRVAL1 in header. Exiting...')
                sys.exit()
        
        trim_off = False
        
        
        if trim_off == False:
            for i in range(len(clean_data)):
                clean_data[i], clean_header[i]= scc_img_trim(clean_data[i], clean_header[i], silent=silent)
                
        # print("time sorting bad and good data",time.time()-start_time)
        ### is it really  unecessary ? 
        # for i in range(len(clean_data)):
        #     clean_data[i], clean_header[i] = scc_putin_array(clean_data[i], clean_header[i], 1024,trim_off=trim_off, silent=silent)

        ## TODO: Implement discri_pobj.pro

        ## TODO: Implement EUVI_PREP.pro

        if ins=='cor2':
            print(post_conj)
            for i in range(len(clean_data)):
                clean_data[i], clean_header[i]  = cor_prep(clean_data[i], clean_header[i], post_conj[i], calpath, pointpath,ftpsc)

            if(polarized):
                pbs = []
                Bs = []
                hdr_pbs = []
                hdr_Bs = []
                polars = [clean_header[i]["POLAR"] for i in range(0,len(clean_header))]
                print(polars)
                for i in range(0,len(clean_data),3):
                    try: 
                        pbim,B, hdr_pb,hdr_B = cor_polariz_python([clean_header[i],clean_header[i+1],clean_header[i+2]], [clean_data[i],clean_data[i+1],clean_data[i+2]])

                        pbs.append(pbim)
                        Bs.append(B)
                        hdr_pbs.append(hdr_pb)
                        hdr_Bs.append(hdr_B)
                    except:
                        print("error in polariz")

                
                # pbs, hdr_pbs = cor_calfac(pbs,hdr_pbs)
                # Bs, hdr_Bs = cor_calfac(Bs,hdr_Bs)

                return pbs, hdr_pbs, Bs, hdr_Bs 
            else:
                # Bs, hdr_bs = cor_calfac(clean_data,clean_header)
                return clean_data,clean_header
            





        elif ins == 'hi_1':
            
            nocalfac_butcorrforipsum = True

            kw_args = {
                'rectify_on' : rectify_on,
                'precomcorrect_on' : precomcorrect_on,
                'trim_off' : trim_off,
                'nocalfac_butcorrforipsum': nocalfac_butcorrforipsum,
                'calibrate_on': True,
                'smask_on': False,
                'fill_mean': True,
                'fill_value': None,
                'update_hdr_on': True,
                'sebip_off': False,
                'calimg_off': False,
                'desmear_off': False,
                'calfac_off': nocalfac_butcorrforipsum,
                'exptime_off': False,
                'bias_off': False,
                'silent': silent,
            }

            for i in range(len(clean_data)):
                clean_data[i], clean_header[i]  = hi_prep(clean_data[i], clean_header[i], post_conj, calpath, pointpath, **kw_args)
              
        elif ins == 'hi_2':

            nocalfac_butcorrforipsum = True

            kw_args = {
                'rectify_on' : rectify_on,
                'precomcorrect_on' : precomcorrect_on,
                'trim_off' : trim_off,
                'nocalfac_butcorrforipsum': nocalfac_butcorrforipsum,
                'calibrate_on': True,
                'smask_on': True,
                'fill_mean': True,
                'fill_value': None,
                'update_hdr_on': True,
                'sebip_off': False,
                'calimg_off': False,
                'desmear_off': False,
                'calfac_off': nocalfac_butcorrforipsum,
                'exptime_off': False,
                'bias_off': False,
                'silent': silent,
            }

            for i in range(len(clean_data)):
                clean_data[i], clean_header[i] = hi_prep(clean_data[i], clean_header[i], post_conj, calpath, pointpath, **kw_args)
        return clean_data,clean_header

scs = {
    'SA': 'ahead',
    'SB': 'behind',
}

def data_reduction(datelist, path, datpath, ftpsc, ins, bflag, silent=True):

    if not silent:
        print('----------------')
        print('DATA REDUCTION')
        print('----------------')

    sc = scs[ftpsc]

    calpath = datpath + 'stereo/secchi/calibration/'
    pointpath = datpath + 'stereo/secchi/data' + '/' + 'hi/'

    for d in tqdm.tqdm(range(0,len(datelist))):

        savepath = path + datelist[d] 
    

        if ins=="cor2" and bflag == "science":
            suffix = ['/*n4*.fts','/*d4*.fts']
        elif (ins=="hi_1" or ins=="hi_2") and bflag == "beacon":
            suffix = ['/*s7*.fts']
        else:
            suffix = ['/*s4*.fts']

        
        fitsfiles = []
        

        for s in suffix:
            for file in sorted(glob.glob(savepath.replace("L1","L0") + s)):
                fitsfiles.append(file)
            


        if len(fitsfiles) == 0:  
            print('No files found for ', ins, ' on ', hdul_header[len(hdul_header)-1]["DATE-END"][:4])
            continue
        
       
        #if there was already something in the folder, we remove it all and restart from scratch 
        print(savepath + '/')
        if os.path.exists(savepath + '/'):
            shutil.rmtree(savepath + '/',onerror=ignore_extended_attributes)
            os.makedirs(savepath + '/')

        else:
            os.makedirs(savepath + '/')

        if not silent:
            print('----------------------------------------')
            print('Starting data reduction for', ins, '...')
            print('----------------------------------------')

        # # correct for on-board sebip modifications to image (division by 2, 4, 16 etc.)
        # # calls function scc_sebip

        hdul_data = []
        hdul_header = []
        hduls = []

        hdul_data2 = []
        hdul_header2 = []
        hduls2 = []
        for i in range(len(fitsfiles)):
            hdul = fits.open(fitsfiles[i])
            if(len(hdul)>1):
                if( 'd4' in fitsfiles[i]):
                    
                    try:
                        if(hdul[0].data.shape[0]==2048):
                            hduls2.append(hdul)
                            hdul_data2.append(hdul[0].data)
                            hdul_header2.append(hdul[0].header)
                    except TypeError:
                        print('Error reading file ', fitsfiles[i]) #happens if file is truncated, smaller than expected
                        continue
                else:
                    hduls.append(hdul)
                    try:
                        hdul_data.append(hdul[0].data)
                        hdul_header.append(hdul[0].header)
                    except TypeError:
                        print('Error reading file ', fitsfiles[i]) #happens if file is truncated, smaller than expected
                    continue
       
        hdul_data = np.array(hdul_data)
        hdul_data2 = np.array(hdul_data2)

        if ins=="cor2":
            pbs, hdr_pbs, Bs, hdr_Bs = reduction(hduls,hdul_data,hdul_header,ftpsc[-1],ins,bflag,calpath,pointpath,silent=True,polarized=True)
            
            for i in range(0,len(pbs)):
                primary_hdu = fits.PrimaryHDU(data=pbs[i],header=hdr_pbs[i])
                totalbrightness_Hhdu = fits.ImageHDU(data=Bs[i],header=hdr_Bs[i])
                hdul = fits.HDUList([primary_hdu, totalbrightness_Hhdu])
                newname = datetime.strptime(hdr_pbs[i]['DATE_AVG'], '%Y-%m-%dT%H:%M:%S.%f').strftime('%Y%m%d_%H%M%S') +'_c2l1_' + ftpsc + '.fts'
                hdul.writeto(savepath +'/' + newname, output_verify='silentfix', overwrite=True)


            Bs, hdr_Bs = reduction(hduls2,hdul_data2,hdul_header2,ftpsc[-1],ins,bflag,calpath,pointpath,silent=True,polarized=False)
            for i in range(0,len(Bs)):
                fits.writeto(savepath + '/' + hdr_Bs[i]['filename'], Bs[i].astype(np.float32), hdr_Bs[i], output_verify='silentfix', overwrite=True)


        else:


            clean_data,clean_header = reduction(hduls,hdul_data,hdul_header,ftpsc[-1],ins,bflag,calpath,pointpath,silent=True)

            if len(clean_data) != 0:
                for i in range(0,len(clean_data)):
                    if bflag == 'science':
                        newname = datetime.strptime(clean_header[i]['DATE-END'], '%Y-%m-%dT%H:%M:%S.%f').strftime('%Y%m%d_%H%M%S') + '_1b' + ins.replace('i_', '') + ftpsc + '.fts'
                    if bflag == 'beacon':
                        newname = datetime.strptime(clean_header[i]['DATE-END'], '%Y-%m-%dT%H:%M:%S.%f').strftime('%Y%m%d_%H%M%S') + '_17' + ins.replace('i_', '') + ftpsc + '.fts'

                    fits.writeto(savepath + '/' + newname, clean_data[i, :, :].astype(np.float32), clean_header[i], output_verify='silentfix', overwrite=True)

            else:
                continue



def create_running_differences(datelist, path,ftpsc, ins, bflag):

    fitsfiles = []
    for d in range(0,len(datelist)):
        savepath = path + datelist[d]

        
        for file in natsorted(glob.glob(savepath.replace("Rdif","L1") + "/*")):
                fitsfiles.append(file)

   
    # cadence of instruments in minutes
    if bflag == 'beacon':
        if ins == 'hi_1' or ins == 'cor2':
            cadence = 120.0
        elif ins=='cor2':
            cadence = 15

    elif bflag == 'science':
        if ins == 'hi_1':
            cadence = 40.0
        elif ins == 'hi_2':
            cadence = 120.0
        elif ins == 'cor2':
            cadence = 15.0
    maxgap = -3.5

    data    = []
    headers = []

    for f in fitsfiles:
        hdul = fits.open(f)
        if ins == 'cor2' and 'd4' not in f:
            data.append(hdul[1].data)

        else:
            data.append(hdul[0].data)              
            headers.append(hdul[0].header)
 
        hdul.close()

    
    for i in range(0,len(data)-1):

        time1 = datetime.strptime(headers[i]['DATE-END'], '%Y-%m-%dT%H:%M:%S.%f')
        time2 = datetime.strptime(headers[i+1]['DATE-END'], '%Y-%m-%dT%H:%M:%S.%f')
        timediff = np.abs(time2-time1).total_seconds()/60.0

        if (timediff <= -maxgap * cadence) & (timediff >= (cadence-5)):
            print(time1,time2)
            crval = [headers[i]['crval1a'],headers[i]['crval2a']]

            center = [headers[i+1]['crpix1'] - 1, headers[i+1]['crpix2'] - 1]

            wcoord = wcs.WCS(headers[i+1], key='A')
            center2 = wcoord.all_world2pix(crval[0], crval[1], 0)

            # Determine shift between preceding and following image
            xshift = center2[0] - center[0]
            yshift = center2[1] - center[1]
            
            shiftarr = [yshift, xshift]

            print(shiftarr)

            shifted = shift(data[i].copy(), shiftarr, mode='nearest')
            diff = np.float32(data[i+1].copy()-shifted)
            diff2 = np.float32(data[i+1].copy()-data[i])
            diff = cv2.medianBlur(diff, 3)
            diff2 = cv2.medianBlur(diff2, 3)


            fig,ax = plt.subplots(1,4)
            ax[0].imshow(data[i])
            ax[1].imshow(data[i+1])
            ax[2].imshow(diff,vmin=np.nanmedian(diff)-5.0*np.std(diff),vmax =np.nanmedian(diff)+5.0*np.std(diff),cmap="gray")
            ax[3].imshow(diff2,vmin=np.nanmedian(diff2)-np.std(diff2),vmax =np.nanmedian(diff2)+np.std(diff2),cmap="gray")
            plt.show()
                        # 
            # ndat = np.float32(data[i] - ims[j])