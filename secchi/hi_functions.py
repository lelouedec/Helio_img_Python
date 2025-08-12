from datetime import datetime,timedelta
from .secchi_functions import *
import os 



def hi_fix_beacon_date(header):
    """
    Fix wrong date in STEREO-HI beacon data. Number of summed images in beacon header is incorrect, so dates calculated using exposure times are wrong.
    @param header: Header of .fits file
    """
    
    n_im = header['IMGSEQ'] + 1

    if header['N_IMAGES'] == 1:
        header['EXPTIME'] = header['EXPTIME']*n_im

        if n_im == 30:
            header['DATE-OBS'] = (datetime.strptime(header['DATE-END'], '%Y-%m-%dT%H:%M:%S.%f') - timedelta(minutes=29,seconds=40)).strftime('%Y-%m-%dT%H:%M:%S.%f')
            header['DATE-CMD'] = header['DATE-OBS'] 
            header['DATE-AVG'] = (datetime.strptime(header['DATE-END'], '%Y-%m-%dT%H:%M:%S.%f') - timedelta(minutes=14,seconds=50)).strftime('%Y-%m-%dT%H:%M:%S.%f')
        if n_im == 99:
            header['DATE-OBS'] = (datetime.strptime(header['DATE-END'], '%Y-%m-%dT%H:%M:%S.%f') - timedelta(hours=1,minutes=38,seconds=50)).strftime('%Y-%m-%dT%H:%M:%S.%f')
            header['DATE-CMD'] = header['DATE-OBS'] 
            header['DATE-AVG'] = (datetime.strptime(header['DATE-END'], '%Y-%m-%dT%H:%M:%S.%f') - timedelta(minutes=49,seconds=25)).strftime('%Y-%m-%dT%H:%M:%S.%f')


        header['N_IMAGES'] = n_im


def hi_remove_saturation(data, header, saturation_limit=14000, nsaturated=5):
    """Direct conversion of hi_remove_saturation.pro for IDL.
    Detects and masks saturated pixels with nan. Takes image data and header as input. Returns fixed image.
    @param data: Data of .fits file
    @param header: Header of .fits file
    @param saturation_limit: Threshold value before pixel is considered saturated
    @param nsaturated: Number of pixels in a column before column is considered saturated
    @return: Data with oversaturated columns removed"""

    info="$Id: hi_remove_saturation.pro,v 1.3 2009/06/08 11:03:38 crothers Exp $"
    
    n_im = header['imgseq'] + 1
    imsum = header['summed']

    dsatval = saturation_limit * n_im * (2 ** (imsum - 1)) ** 2

    ind = np.where(data > dsatval)

    # if a pixel has a value greater than the dsatval, begin check to test if column is saturated

    if ind[0].size > 0:

        # all pixels are set to zero, except ones exceeding the saturation limit

        mask = np.zeros(np.shape(data))
        ans = data.copy()
        mask[ind] = 1

        # pixels are summed up column-wise
        # where nsaturated is exceeded, values are replaced by nan

        colmask = np.nansum(mask, 0)
        ii = np.array(np.where(colmask > nsaturated))
        # print('nsaturated:', nsaturated)
        # print('colmask:', colmask)
        if len(ii) > 0:
            ans[:, ii] = np.nan #np.nanmedian(data)
        else:
            ans = data.copy()

    else:
        ans = data.copy()

    return ans

def hi_exposure_wt(hdr, silent=True):

    if 'DETECTOR' not in hdr or hdr['DETECTOR'] not in ['HI1', 'HI2']:
        raise ValueError('for HI DETECTOR only')

    clearest = 0.70
    exp_eff = hdr['EXPTIME'] + hdr['n_images'] * (clearest - hdr['CLEARTIM'] + hdr['RO_DELAY'])

    dataWeight = hdr['n_images'] * (2 ** (hdr['ipsum'] - 1))

    wt0 = np.arange(hdr['naxis2'])
    wt1 = np.reshape(wt0, (1, hdr['naxis2']))
    wt2 = np.reshape(wt0[::-1], (1, hdr['naxis2']))

    if hdr['RECTIFY'] == True and hdr['OBSRVTRY'] == 'STEREO_B':
        if not silent:
            print("rectified")
        wt = exp_eff + wt2 * hdr['line_ro'] + wt1 * hdr['line_clr']
    else:
        if not silent:
            print("normal")
        wt = exp_eff + wt1 * hdr['line_clr'] + wt2 * hdr['line_ro']

    wt =rebin(wt, (hdr['naxis1'], hdr['naxis2']))

    return wt


#@numba.njit()
def hi_desmear(im, hdr, post_conj, silent=True):
    """
    Conversion of hi_desmear.pro for IDL. Removes smear caused by no shutter. First compute the effective exposure time
    [time the ccd was static, generate matrix with this in diagonal, the clear time below and the read time above;
    invert and multiply by the image.

    @param im: Data of .fits file
    @param hdr: Header of .fits file
    @param post_conj: Indicates whether spacecraft is pre or post conjecture
    @param silent: Run in silent mode
    @return: Array corrected for shutterless camera
    """

    version='Applied hi_desmear.pro,v 1.11 2023/08/15 16:22:32'
    hdr['HISTORY'] = version

    # Check valid values in header
    if hdr['CLEARTIM'] < 0:
        raise ValueError('CLEARTIM invalid')
    if hdr['RO_DELAY'] < 0:
        raise ValueError('RO_DELAY invalid')
    if hdr['LINE_CLR'] < 0:
        raise ValueError('LINE_CLR invalid')
    if hdr['LINE_RO'] < 0:
        raise ValueError('LINE_RO invalid')
    
    img = im.astype(float)

    # Extract image array if underscan present
    ## CHANGE fixed messed up indexing
    if hdr['dstart1'] <= 1 or hdr['naxis1'] == hdr['naxis2']:
        image = img
    else:
        image = img[hdr['dstart2']-1:hdr['dstop2'],hdr['dstart1']-1:hdr['dstop1']]

    clearest = 0.70
    exp_eff = hdr['EXPTIME'] + hdr['n_images'] * (clearest - hdr['CLEARTIM'] + hdr['RO_DELAY'])

    dataWeight = hdr['n_images'] * (2 ** (hdr['ipsum'] - 1))

    inverted = 0

    if hdr['RECTIFY'] == True:
        if hdr['OBSRVTRY'] == 'STEREO_B':
            if post_conj == 0:
                inverted = 1
            else:
                print('hi_desmear not implemented for STEREO-B with post_conj=True.')
                sys.exit()
        if hdr['OBSRVTRY'] == 'STEREO_A':
            if post_conj == 1:
                inverted = 1
            else:
                inverted = 0
    

    
    if inverted == 1:
        fixup = sc_inverse(hdr['naxis2'], exp_eff, dataWeight*hdr['line_clr'], dataWeight*hdr['line_ro'])
        
    
    else:
        fixup = sc_inverse(hdr['naxis2'], exp_eff, dataWeight*hdr['line_ro'], dataWeight*hdr['line_clr'])


    image =  fixup @ image
    

    if hdr['dstart1'] <= 1 or (hdr['naxis1'] == hdr['naxis2']):
        img = image.copy()

    else:
        img = image[hdr['dstart2'] - 1:hdr['dstop2'], hdr['dstart1'] - 1:hdr['dstop1']]

    return img


def hi_fill_missing(data, header, silent=True):
    """
    Conversion of fill_missing.pro for IDL. Set missing block values sensibly.

    @param data: Data from .fits file
    @param header:Header of .fits file
    @return: Corrected image
    """
    if header['NMISSING'] > 0:
        if len(header['MISSLIST']) < 1:
            if not silent:
                print('Mismatch between nmissing and misslist.')
        else:
            fields = scc_get_missing(header)
            shp = np.shape(data)
            data = data.flatten()
            data[fields] = np.nan
            data = data.reshape(shp)

    return data


def hi_correction(im, hdr, post_conj, calpath, sebip_off=False, calimg_off=False, desmear_off=False,
                  calfac_off=False, exptime_off=False, silent=True,
                  saturation_limit=None, nsaturated=None, bias_off=False, **kw_args):
    
    version = "Applied hi_correction.pro,v 1.20 2015/02/09 14:43:14 crothers"
    
    hdr['HISTORY'] = version
    
    # Correct for SEB IP (ON)
    if not sebip_off:
        im, hdr = scc_sebip(im, hdr, silent=silent)

    # Bias Subtraction (ON)
    if bias_off:
        biasmean = 0.0
        
    else:
        biasmean = get_biasmean(hdr, silent=silent)

        if biasmean != 0.0:
            hdr['HISTORY'] = 'Bias Subtracted ' + str(biasmean)
            hdr['OFFSETCR'] = biasmean
            im -= biasmean

            if not silent:
                print(f"Subtracted BIAS={biasmean}")
    # Extract and correct for cosmic ray reports

    ### hi_cosmics modifies the images as reference! 
    cosmics = hi_cosmics(hdr, im, post_conj, silent=silent)
    im = hi_remove_saturation(im, hdr)

    if not exptime_off:
        if desmear_off:
            im /= hi_exposure_wt(hdr)

            if hdr['NMISSING'] > 0:
                im = hi_fill_missing(im, hdr, silent=silent)

            hdr['HISTORY'] = 'Applied exposure weighting'
            hdr['BUNIT'] = 'DN/s'

            if not silent:
                print("Exposure Normalized to 1 Second, exposure weighting method")

        else:
            im = hi_desmear(im, hdr, post_conj, silent=silent)
            
            if hdr['NMISSING'] > 0:
                im = hi_fill_missing(im, hdr, silent=silent)

            hdr['BUNIT'] = 'DN/s'

            if not silent:
                print("Exposure Normalized to 1 Second, desmearing method")
    
    ipkeep = hdr['IPSUM']
    # # Apply calibration factor
    if calfac_off:
        calfac = 1.0
    else:
        calfac, hdr = get_calfac(hdr, silent=silent)
    
    calfac = 1.0
    diffuse = 1.0
    
    if calfac != 1.0:
        hdr['HISTORY'] = 'Applied calibration factor ' + str(calfac)

        if not silent:
            print(f"Applied calibration factor {calfac}")

        if not calimg_off:
            diffuse = scc_hi_diffuse(hdr, ipsum=ipkeep)
            hdr['HISTORY'] = 'Applied diffuse source correction'

            if not silent:
                print("Applied diffuse source correction")
    else:
        calfac_off = True

    calimg = 1.0
    # Correction for flat field and vignetting (ON)
    if calimg_off:
        calimg = 1.0
    else:
        calimg, fn = get_calimg(hdr, calpath, post_conj)
        
        if calimg.shape[0] > 1:
            hdr['HISTORY'] = f'Applied Flat Field {fn}'
    
    # Apply Correction
    im = im * calfac * diffuse * calimg
    
    return im, hdr

def hi_cosmics(hdr, img, post_conj, silent=True):
    """
    Extracts cosmic ray scrub reports from HI images.
    
    Args:
        hdr: Image header, either FITS or SECCHI structure
        img: Level 0.5 image in DN (long)
    
    Returns:
        Cosmic ray scrub count
    """

    cosmics = -1

    if 's4h' not in hdr['filename']:
        cosmics = hdr['cosmics']
    elif hdr['n_images'] <= 1 and hdr['imgseq'] <= 1:
        cosmics = hdr['cosmics']
    else:
        count = hdr['imgseq'] + 1

        inverted = 0

        if hdr['RECTIFY'] == True:
            if post_conj:
                if hdr['OBSRVTRY'] == 'STEREO_A':
                    inverted = 1
                if hdr['OBSRVTRY'] == 'STEREO_B':
                    inverted = 0
            else:
                if hdr['OBSRVTRY'] == 'STEREO_A':
                    inverted = 0
                if hdr['OBSRVTRY'] == 'STEREO_B':
                    inverted = 1

        if inverted:
            cosmic_counter = img[0,count]

            if cosmic_counter == count:
                cosmics = np.flip(img[0, :count])
                img[0, :count+1] = img[1, :count+1]

            else:
                seek = np.arange(count)
                q = np.where(seek == img[0, :count])[0]

                ctr = q.size

                if ctr > 0:
                    count = q[ctr-1]

                    if count > 1:
                        cosmics = np.flip(img[0, :count])
                        img[0, :count+1] = img[1, :count+1]
                        
                        if not silent:
                            if ctr == 1:
                                print('cosmic ray counter recovered')
                            else:
                                print('cosmic ray counter possibly recovered')

                    else:
                        if hdr['nmissing'] > 0:
                            try:
                                miss = scc_get_missing(hdr, silent=True)
                            except Exception:
                                miss = []
                            if (miss.size > 0) and np.sum(np.array(miss) == hdr['imgseq'] + 1) > 0:
                                if not silent:
                                    print('cosmic ray report is missing')
                            else:
                                if not silent:
                                    print('cosmic ray report implies no images [missing blks?]')
                        else:
                            if not silent:
                                print('cosmic ray report implies no images')
                        cosmics = -1

                else:
                    if not silent:
                        print('cosmic ray counter not recovered')
                    cosmics = -1
        else:

            naxis1 = hdr['naxis1']
            naxis2 = hdr['naxis2']
            cosmic_counter = img[naxis2 - 1, naxis1 - count - 1]

            if cosmic_counter == count:
                cosmics = img[naxis2 - 1, naxis1 - count:naxis1]
                img[naxis2-1,naxis1-count-1:naxis1] = img[naxis2-2, naxis1-count-1:naxis1]

            else:
                seek = np.flip(np.arange(count))
                q = np.where(seek == img[naxis2 - 1, naxis1 - count:naxis1])[0]

                if q.size > 0:
                    count = seek[q[0]]

                    if count > 1:
                        cosmics = img[naxis2-1,naxis1-count:naxis1]
                        img[naxis2-1,naxis1-count-1:naxis1] = img[naxis2 - 2, naxis1-count-1:naxis1]

                        if q.size == 1:
                            if not silent:
                                print('cosmic ray counter recovered')
                        else:
                            if not silent:
                                print('cosmic ray counter possibly recovered')
                    
                    else:
                        if hdr['nmissing'] > 0:
                            try:
                                miss = scc_get_missing(hdr, silent=True)
                            except Exception:
                                miss = []

                            if (miss.size > 0) and np.sum(naxis1 * naxis2 - 1 - np.array(miss) == hdr['imgseq'] + 1) > 0:
                                if not silent:
                                    print('cosmic ray report is missing')
                            else:
                                if not silent:
                                    print('cosmic ray report implies no images [missing blks?]')
                        else:
                            if not silent:
                                print('cosmic ray report implies no images')
                        cosmics = -1

                else:
                    if not silent:
                        print('cosmic ray counter not recovered')
                    cosmics = -1

    return cosmics



def hi_fix_pointing(header, point_path, post_conj, ravg=5, silent=True):
    """
    Conversion of fix_pointing.pro for IDL. To read in the pointing information from the appropriate  pnt_HI??_yyyy-mm-dd_fix_mu_fov.fts file and update the
    supplied HI index with the best fit pointing information and optical parameters calculated by the minimisation
    process of Brown, Bewsher and Eyles (2008).

    @param header: Header of .fits file
    @param point_path: Path of pointing calibration files
    @param ftpsc: STEREO Spacecraft (A/B)
    @param post_conj: Is the spacecraft pre- or post conjunction (2014)
    @param ravg: Set ravg wuality parameter
    @param silent: Run in silent mode
    """

    ## CHANGE From 1 to 0 to reflect default IDL behaviour
    hi_nominal = 0

    hdr_date = header['DATE-AVG']
    hdr_date = hdr_date[0:10]

    point_file = 'pnt_' + header['DETECTOR'] + header['OBSRVTRY'][7] + '_' + hdr_date + '_' + 'fix_mu_fov.fts'
    fle = point_path + point_file
    
    if os.path.isfile(fle):

        if not silent:
            print(('Reading {}...').format(point_file))
    
        hdul_point = fits.open(fle)

        for i in range(1, len(hdul_point)):
            extdate = hdul_point[i].header['extname']
            fledate = hdul_point[i].header['filename'][0:13]

            if (header['DATE-AVG'] == extdate) or (datetime.datetime.strptime(header['DATE-OBS'], '%Y-%m-%dT%H:%M:%S.%f').strftime('%Y%m%d_%H%M') == fledate):
                ec = i
                break

        if (header['DATE-AVG'] == extdate) or (datetime.datetime.strptime(header['DATE-OBS'], '%Y-%m-%dT%H:%M:%S.%f').strftime('%Y%m%d_%H%M') == fledate):

            stcravg = hdul_point[ec].header['ravg']
            stcnst1 = hdul_point[ec].header['nst1']

            if header['naxis1'] != 0:
                sumdif = np.round(header['cdelt1'] / hdul_point[ec].header['cdelt1'])
            else:
                sumdif = 1

            if stcnst1 < 20:
                if not silent:
                    print('Subfield presumed')
                    print('Using calibrated fixed instrument offsets')

                hi_calib_point(header, post_conj, hi_nominal)
                header['ravg'] = -894.

            else:
                ## CHANGE < to <=, > to >=

                if (stcravg <= ravg) & (stcravg >= 0.):
                    header['crval1a'] = hdul_point[ec].header['crval1a']
                    header['crval2a'] = hdul_point[ec].header['crval2a']
                    header['pc1_1a'] = hdul_point[ec].header['pc1_1a']
                    header['pc1_2a'] = hdul_point[ec].header['pc1_2a']
                    header['pc2_1a'] = hdul_point[ec].header['pc2_1a']
                    header['pc2_2a'] = hdul_point[ec].header['pc2_2a']
                    header['cdelt1a'] = hdul_point[ec].header['cdelt1a'] * sumdif
                    header['cdelt2a'] = hdul_point[ec].header['cdelt2a'] * sumdif
                    header['pv2_1a'] = hdul_point[ec].header['pv2_1a']
                    header['crval1'] = hdul_point[ec].header['crval1']
                    header['crval2'] = hdul_point[ec].header['crval2']
                    header['pc1_1'] = hdul_point[ec].header['pc1_1']
                    header['pc1_2'] = hdul_point[ec].header['pc1_2']
                    header['pc2_1'] = hdul_point[ec].header['pc2_1']
                    header['pc2_2'] = hdul_point[ec].header['pc2_2']
                    header['cdelt1'] = hdul_point[ec].header['cdelt1'] * sumdif
                    header['cdelt2'] = hdul_point[ec].header['cdelt2'] * sumdif
                    header['pv2_1'] = hdul_point[ec].header['pv2_1']
                    header['xcen'] = hdul_point[ec].header['xcen']
                    header['ycen'] = hdul_point[ec].header['ycen']
                    header['crota'] = hdul_point[ec].header['crota']
                    header['ins_x0'] = hdul_point[ec].header['ins_x0']
                    header['ins_y0'] = hdul_point[ec].header['ins_y0']
                    header['ins_r0'] = hdul_point[ec].header['ins_r0']
                    header['ravg'] = hdul_point[ec].header['ravg']

                else:
                    if not silent:
                        print('R_avg does not meet criteria')
                        print('Using calibrated fixed instrument offsets')

                    hi_calib_point(header, post_conj, hi_nominal)
                    header['ravg'] = -883.

        else:
            if not silent:
                print(('No pointing calibration file found for file {}').format(point_file))
                print('Using calibrated fixed instrument offsets')

            hi_calib_point(header, post_conj, hi_nominal)
            header['ravg'] = -882.

    if not os.path.isfile(fle):
        if not silent:
            print(('No pointing calibration file found for file {}').format(point_file))
            print('Using calibrated fixed instrument offsets')

        hi_calib_point(header, post_conj, hi_nominal)
        header['ravg'] = -881.
    
    return header
#######################################################################################################################################
def hi_calib_roll(header, system, post_conj, hi_nominal):
    """
    Conversion of hi_calib_roll.pro for IDL. Calculate the total roll angle of the HI image including
    contributions from the pitch and roll of the spacecraft. The total HI roll is a non-straighforward combination
    of the individual rolls of the s/c and HI, along with the pitch of the s/c and the offsets of HI. This
    routine calculates the total roll by taking 2 test points in the HI fov, transforms them to the
    appropriate frame of reference (given by the system keyword) and calculates the angle they make in this frame.

    @param header: Header of .fits file
    @param system: Which coordinate system to work in 'hpc' or 'gei'
    @param post_conj: Is the spacecraft pre- or post conjunction (2014)
    @param hi_nominal: Retrieve nominal pointing values at launch (propagated to get_hi_params)
    @return: Total roll of the spacecraft
    """

    if 'summed' in header:
        naxis1 = 2048 / 2 ** (header['summed'] - 1)
        naxis2 = naxis1

    else:
        naxis1 = header['naxis1']
        naxis2 = header['naxis2']

    if naxis1 <= 0:
        naxis1 = 1024.

    if naxis2 <= 0:
        naxis2 = 1024.

    cpix = np.array([naxis1, naxis2]) / 2. - 0.5
    xv = cpix[0] + [0., 512.]
    yv = np.full_like(cpix, cpix[1])

    xy = fov2pos(xv, yv, header, system, hi_nominal)

    z = xy[2, 1] - xy[2, 0]
    x = -(xy[1, 1] - xy[1, 0])
    y = xy[0, 1] - xy[0, 0]

    tx = xy[0, 0]
    ty = xy[1, 0]
    tz = 0.0

    a = np.sqrt(tx ** 2 + ty ** 2)
    b = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    ab = x * tx + y * ty

    val = np.nanmin([np.nanmax([ab / (a * b), -1.0]), 1.0])

    if z >= 0.0:
        oroll = np.arccos(val) * 180. / np.pi

    else:
        oroll = -np.arccos(val) * 180. / np.pi

    if post_conj:
        oroll = oroll - 180.

    return oroll

def get_hi_params(header, hi_nominal):
    """
    Conversion of get_hi_params.pro for IDL. To detect which HI telescope is being used and return the
    instrument offsets relative to the spacecraft along with the mu parameter and the fov in degrees.
    As 'best pointings' may change as further calibration is done, it was thought more useful if there was one
    central routine to provide this data, rather than having to make the same changes in many different
    codes. Note, if you set one of the output variables to some value, then it will retain that value.

    @param header: Header of .fits file
    @param hi_nominal: Retrieve nominal pointing values at launch (propagated to get_hi_params)
    @return: HI yaw offset, HI pitch offfset, HI roll, HI distortion parameter, HI fov
    """

    if hi_nominal:
        if ((header['obsrvtry'] == 'STEREO_A') and (header['detector'] == 'HI1')):
            pitch_hi = 0.0
            offset_hi = -13.98
            roll_hi = 0.0
            mu = 0.16677
            d = 20.2663

        if ((header['obsrvtry'] == 'STEREO_A') and (header['detector'] == 'HI2')):
            offset_hi = -53.68
            pitch_hi = 0.
            roll_hi = 0.
            mu = 0.83329
            d = 70.8002

        if ((header['obsrvtry'] == 'STEREO_B') and (header['detector'] == 'HI1')):
            offset_hi = 13.98
            pitch_hi = 0.
            roll_hi = 0.
            mu = 0.10001
            d = 20.2201

        if ((header['obsrvtry'] == 'STEREO_B') and (header['detector'] == 'HI2')):
            offset_hi = 53.68
            pitch_hi = 0.
            roll_hi = 0.
            mu = 0.65062
            d = 69.8352

    else:
        if ((header['obsrvtry'] == 'STEREO_A') and (header['detector'] == 'HI1')):
            pitch_hi = 0.1159
            offset_hi = -14.0037
            roll_hi = 1.0215
            mu = 0.102422
            d = 20.27528

        if ((header['obsrvtry'] == 'STEREO_A') and (header['detector'] == 'HI2')):
            offset_hi = -53.4075
            pitch_hi = 0.0662
            roll_hi = 0.1175
            mu = 0.785486
            d = 70.73507

        if ((header['obsrvtry'] == 'STEREO_B') and (header['detector'] == 'HI1')):
            offset_hi = 14.10
            pitch_hi = 0.022
            roll_hi = 0.37
            mu = 0.09509
            d = 20.23791

        if ((header['obsrvtry'] == 'STEREO_B') and (header['detector'] == 'HI2')):
            offset_hi = 53.690
            pitch_hi = 0.213
            roll_hi = -0.052
            mu = 0.68886
            d = 70.20152

    return pitch_hi, offset_hi, roll_hi, mu, d




def fov2pos(xv, yv, header, system, hi_nominal):
    """
    Conversion of fov2pos.pro for IDL. To convert HI pixel positions to solar plane of sky
    coordinates in units of AU (actually, not quite, 1 is defined as the distance from the S/C to the Sun) and
    the Sun at (0,0). HI pixel position is converted to general AZP, then to Cartesian coordinates in the HI frame of
    reference. This is then rotated to be in the S/C frame and finally to the end reference frame. The final frame
    is a left-handed Cartesian system with x into the screen (towards the reference point), and z pointing up.

    @param xv: Array of x-pixel positions to be converted
    @param yv: Array of y-pixel positions to be converted
    @param header: Header of .fits file
    @param system: Which coordinate system to work in 'hpc' or 'gei'
    @param hi_nominal: Retrieve nominal pointing values at launch (propagated to get_hi_params)
    @return: An array of (x,y,z,w) quadruplets, where x,y,z have the meaning described above and w is a scale
    factor (see '3D computer graphics' by Alan Watt for further details). The wcolumn can be discounted
    for almost all user applications.
    """

    if system == 'hpc':
        yaw = header['sc_yaw']
        pitch = header['sc_pitch']
        roll = -header['sc_roll']

    else:
        yaw = header['sc_yawa']
        pitch = header['sc_pita']
        roll = header['sc_rolla']

    ccdosx = 0.
    ccdosy = 0.

    naxis = np.array([header['naxis1'], header['naxis2']])

    ## CHANGE From if any then set all 1024
    naxis = np.where(naxis == 0, 1024, naxis)

    pmult = naxis / 2.0 - 0.5

    pitch_hi, offset_hi, roll_hi, mu, d = get_hi_params(header, hi_nominal)

    ang = (90. - 0.5 * d) * np.pi / 180.
    rng = (1.0 + mu) * np.cos(ang) / (np.sin(ang) + mu)

    vfov = np.array([xv, yv])
    nst = len(vfov[0])

    vv4 = np.zeros((4, nst))

    vv4[0][:] = ((vfov[0] - ccdosx) / pmult[0] - 1.0) * rng
    vv4[1][:] = ((vfov[1] - ccdosy) / pmult[1] - 1.0) * rng
    vv4[2][:] = 1.0
    vv4[3][:] = 1.0

    vv3 = azp2cart(vv4, mu)
    vv3[2, :] = 1.0

    vv2 = hi2sc(vv3, roll_hi, pitch_hi, offset_hi)
    vv = sc2cart(vv2, roll, pitch, yaw)

    return vv

def hi2sc(vec, roll_hi_deg, pitch_hi_deg, offset_hi_deg):
    """
    Conversion of hi2sec.pro for IDL. To transform the given position from the HI frame of
    reference to the spacecraft frame of reference. Note, this is a low level code, and would usually not be
    called directly. For the transformation we use 4x4 transformation
    matrices discussed in e.g. '3D computer graphics' by Alan Watt

    @param vec: Array of vector postions to transform
    @param roll_hi_deg: HI roll angle relative to spacecraft (in degrees)
    @param pitch_hi_deg: HI pitch angle relative to spacecraft (in degrees)
    @param offset_hi_deg: HI yaw angle relative to spacecraft (in degrees)
    @return: An array of transformed vector positions
    """
    npts = len(vec[0, :])

    theta = (90 - pitch_hi_deg) * np.pi / 180.
    phi = offset_hi_deg * np.pi / 180.
    roll = roll_hi_deg * np.pi / 180.

    normz = np.sin(theta) * np.cos(phi)
    normx = np.sin(theta) * np.sin(phi)
    normy = np.cos(theta)

    vdx = 0.
    vdy = 1.
    vdz = 0.

    vd_norm = vdx * normx + vdy * normy + vdz * normz

    vxtmp = vdx - vd_norm * normx
    vytmp = vdy - vd_norm * normy
    vztmp = vdz - vd_norm * normz

    ndiv = np.sqrt(vxtmp ** 2 + vytmp ** 2 + vztmp ** 2)
    vx = vxtmp / ndiv
    vy = vytmp / ndiv
    vz = vztmp / ndiv

    ux = -(normy * vz - normz * vy)
    uy = -(normz * vx - normx * vz)
    uz = -(normx * vy - normy * vx)

    cx = 0.
    cy = 0.
    cz = 0.

    tmat = np.array([[1., 0., 0., -cx], [0., 1., 0., -cy], [0., 0., 1., -cz], [0., 0., 0., 1.]])
    rmat = np.array([[ux, uy, uz, 0.], [vx, vy, vz, 0.], [normx, normy, normz, 0.], [0., 0., 0., 1.]])

    rollmat = np.array(
        [[np.cos(roll), -np.sin(roll), 0., 0.], [np.sin(roll), np.cos(roll), 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]])

    tview = rollmat @ (rmat @ tmat)

    itview = np.linalg.inv(tview)

    vout = np.zeros((4, npts))

    for i in range(npts):
        vout[:, i] = np.transpose(itview @ np.transpose(vec[:, i]))

    return vout

def fov2radec(xv, yv, header, system, hi_nominal):
    """
    Conversion of fov2radec for IDL. To convert HI pixel positions to RA-Dec pairs.
    HI pixel positions are converted into a general AZP form, which is converted to cartesian coordintes.
    These are then rotated from HI pointing, to S/C pointing then finally aligned with the RA-Dec frame.
    The Cartesian coordinates are finally converted to RA-Dec.

    @param xv: An array of x pixel positions
    @param yv: An array of y pixel positions
    @param header: Header of .fits file
    @param system: Which coordinate system to work in 'hpc' or 'gei'
    @param hi_nominal: Retrieve nominal pointing values at launch (propagated to get_hi_params)
    @return: An array of transformed vector positions
    """
    if system == 'gei':
        yaw = header['sc_yawa']
        pitch = header['sc_pita']
        roll = header['sc_rolla']

    else:
        yaw = header['sc_yaw']
        pitch = header['sc_pitch']
        roll = -header['sc_roll']

    ccdosx = 0.
    ccdosy = 0.
    pmult = header['naxis1'] / 2.0

    pitch_hi, offset_hi, roll_hi, mu, d = get_hi_params(header, hi_nominal)

    ang = (90. - 0.5 * d) * np.pi / 180.
    rng = (1.0 + mu) * np.cos(ang) / (np.sin(ang) + mu)

    vfov = np.array([xv, yv])

    nst = len(vfov[0])

    vv4 = np.zeros((4, nst))

    vv4[0, :] = ((vfov[0] - ccdosx) / pmult - 1.0) * rng
    vv4[1, :] = ((vfov[1] - ccdosy) / pmult - 1.0) * rng
    vv4[3, :] = 1.0

    vv3 = azp2cart(vv4, mu)
    vv2 = hi2sc(vv3, roll_hi, pitch_hi, offset_hi)
    vv = sc2cart(vv2, roll, pitch, yaw)

    radec = np.zeros((2, nst))

    rd = 180 / np.pi

    for i in range(nst):

        th = np.arccos(vv[2, i])
        radec[1, i] = 90. - th * rd

        cphi = vv[0, i] / np.sin(th)
        sphi = vv[1, i] / np.sin(th)

        th1 = np.arccos(cphi) * rd
        th2 = np.arcsin(sphi) * rd

        if (th2 > 0):
            ra = th1
        else:
            ra = -th1

        radec[0, i] = ra

    return radec



def hi_calib_point(header, post_conj, hi_nominal):
    """
    Conversion of hi_calib_point.pro for IDL.

    @param header: Header of .fits file
    @param post_conj: Is the spacecraft pre- or post conjunction (2014)
    @param hi_nominal: Retrieve nominal pointing values at launch (propagated to get_hi_params)
    """

    roll = hi_calib_roll(header, 'gei', post_conj, hi_nominal)

    header['pc1_1a'] = np.cos(roll * np.pi / 180.)
    header['pc1_2a'] = -np.sin(roll * np.pi / 180.)
    header['pc2_1a'] = np.sin(roll * np.pi / 180.)
    header['pc2_2a'] = np.cos(roll * np.pi / 180.)

    roll = hi_calib_roll(header, 'hpc', post_conj, hi_nominal)

    header['crota'] = -roll
    header['pc1_1'] = np.cos(roll * np.pi / 180.)
    header['pc1_2'] = -np.sin(roll * np.pi / 180.)
    header['pc2_1'] = np.sin(roll * np.pi / 180.)
    header['pc2_2'] = np.cos(roll * np.pi / 180.)

    if 'summed' in header:

        naxis1 = 2048 / 2 ** (header['summed'] - 1)
        naxis2 = naxis1

    else:
        naxis1 = header['naxis1']
        naxis2 = header['naxis2']

    if naxis1 <= 0:
        naxis1 = 1024.

    if naxis2 <= 0:
        naxis2 = 1024.

    xv = [0.5 * naxis1, naxis1]
    yv = [0.5 * naxis2, 0.5 * naxis2]

    radec = fov2radec(xv, yv, header, 'gei', hi_nominal)

    header['crval1a'] = radec[0, 0]
    header['crval2a'] = radec[1, 0]

    radec = fov2radec(xv, yv, header, 'hpc', hi_nominal)
    header['crval1'] = -radec[0, 0]
    header['crval2'] = radec[1, 0]

    pitch_hi, offset_hi, roll_hi, mu, d = get_hi_params(header, hi_nominal)
    header['pv2_1a'] = mu
    header['pv2_1'] = mu

    fp, fp_mm, plate = fparaxial(d, mu, header['naxis1'], header['naxis2'])

    if header['cunit1a'] == 'deg':
        xsize = plate / 3600.

    if header['cunit2a'] == 'deg':
        ysize = plate / 3600.

    if header['cunit1a'] == 'arcsec':
        xsize = plate

    if header['cunit2a'] == 'arcsec':
        ysize = plate

    header['cdelt1a'] = -xsize
    header['cdelt2a'] = ysize

    if header['cunit1'] == 'deg':
        xsize = plate / 3600.

    if header['cunit2'] == 'deg':
        ysize = plate / 3600.

    if header['cunit1'] == 'arcsec':
        xsize = plate

    if header['cunit2'] == 'arcsec':
        ysize = plate

    header['cdelt1'] = xsize
    header['cdelt2'] = ysize

    header['ins_x0'] = -offset_hi
    header['ins_y0'] = pitch_hi
    header['ins_r0'] = -roll_hi


def fparaxial(fov, mu, naxis1, naxis2):
    """
    Conversion of fparaxial.pro for IDL. Calculate paraxial platescale of HI.
    This routine calculates the paraxial platescale from the calibrated fov and distortion parameter.

    @param fov: Calibrated field of view
    @param mu: HI distortion parameter
    @param naxis1: NAXIS1 from .fits header
    @param naxis2: NAXIS2 from .fits header
    @return: Paraxial platescale in mm
    """
    theta = (90. - (fov / 2.)) * np.pi / 180.

    tmp1 = (np.sin(theta) + mu) / ((1 + mu) * np.cos(theta))

    fp = (naxis1 / 2.) * tmp1

    widthmm = 0.0135 * 2048.
    fp_mm = widthmm * tmp1 / 2.

    plate = (0.0135 / fp_mm) * (180 / np.pi) * 3600.

    plate = plate * 2048. / naxis1

    return fp, fp_mm, plate

def hi_prep(im, hdr, post_conj, calpath, pointpath, calibrate_on=True, smask_on=False, fill_mean=True, fill_value=None, update_hdr_on=True, silent=True, **kw_args):
    """
    Conversion of hi_prep.pro for IDL. Processes the image with various corrections and updates based on the header information and flags.

    Parameters:
    -----------
    im : numpy.ndarray
        Image data to be processed.
    hdr : dict
        Header information associated with the image.
    calibrate_on : bool, optional
        If False, disables calibration corrections.
    smask_on : bool, optional
        If True, apply smoothing mask (only for HI2 detector).
    fill_mean : bool, optional
        If True, fill mask regions with mean image value.
    fill_value : float, optional
        Specific value to fill mask regions.
    update_hdr_on : bool, optional
        If False, disables updating header to Level 1 values.
    silent : bool, optional
        If True, suppress informational messages.
    corr_kw : dict, optional
        Dictionary of correction keywords passed to hi_correction().
    """
    # Update IMGSEQ for hi-res images if imgseq is not 0

    if hdr['NAXIS1'] > 1024 and hdr['IMGSEQ'] != 0 and hdr['N_IMAGES'] == 1:
        hdr['imgseq'] = 0

    # Calibration corrections
    if calibrate_on:
        im, hdr = hi_correction(im, hdr, post_conj, calpath, **kw_args)
        # hdr = hi_fix_pointing(hdr, pointpath, post_conj, silent=silent)
        if(im.shape[0]==1024):
            hdr = hi_fix_pointing(hdr, pointpath, post_conj, silent=silent)
        else:
            hi_calib_point(hdr, post_conj, 0)
            hdr['ravg'] = -881.0
    else:
        cosmics = -1

    # Smooth Mask (only for HI2 detector)
    if smask_on and calibrate_on and hdr['DETECTOR'] == 'HI2':
        mask = get_smask(hdr, calpath, post_conj, silent=True)
        m_dex = np.where(mask == 0)
        if fill_mean:
            im[m_dex] = np.mean(im)
        elif fill_value is not None:
            im[m_dex] = fill_value
        else:
            im *= mask
        if not silent:
            print('Mask applied to HI2 image')

    if kw_args['calfac_off'] and kw_args['nocalfac_butcorrforipsum']:
        sumcount = hdr['ipsum'] - 1
        divfactor = (2 ** sumcount) ** 2
        im = im / divfactor

        if hdr['ipsum'] > 1:
            if not silent:
                print(f'Divided image by {divfactor} to account for IPSUM')
                print('IPSUM changed to 1 in header.')
            
            hdr['history'] = f'image Divided by {divfactor} to account for IPSUM'
        
        hdr['ipsum'] = 1
        hdr['bunit'] = hdr['bunit'] + '/CCDPIX'

    # Update Header to Level 1 values
    if update_hdr_on:
        hdr = scc_update_hdr(im, hdr)

    calfac,hdr = get_calfac(hdr,'MSB')
    calfac = calfac*2.223e15

    cdelt=35.96382/3600
    summing=int(np.log(hdr["CDELT1"]/cdelt)/np.log(2.))+1
    diffuse = scc_hi_diffuse(hdr,summing)

    im = im * calfac * diffuse 

    #if we want to modify it there
    # hdr['bunit'] = 'S10'


    return im, hdr


def scc_hi_diffuse(header, ipsum=None):
    """
    Conversion of scc_hi_diffuse.pro for IDL. Compute correction for diffuse sources arrising from changes
    in the solid angle in the optics. In the mapping of the optics the area of sky viewed is not equal off axis.

    @param header: Header of .fits file
    @param ipsum: Allows override of header ipsum value for use in L1 and beyond images
    @return: Correction factor for given image
    """

    if ipsum is None:
        ipsum = header['ipsum']
    summing = 2 ** (ipsum - 1)

    ##CHANGE changed if-else logic
    
    try:
        ravg = header['ravg']

    except KeyError:
        ravg = 0

    if ravg >= 0:
        mu = header['pv2_1']
        cdelt = header['cdelt1'] * np.pi / 180

    else:
        
        if header['detector'] == 'HI1':

            if header['OBSRVTRY'] == 'STEREO_A':
                mu = 0.102422
                cdelt = 35.96382 / 3600 * np.pi / 180 * summing

            elif header['OBSRVTRY'] == 'STEREO_B':
                mu = 0.095092
                cdelt = 35.89977 / 3600 * np.pi / 180 * summing

        elif header['detector'] == 'HI2':

            if header['OBSRVTRY'] == 'STEREO_A':
                mu = 0.785486
                cdelt = 130.03175 / 3600 * np.pi / 180 * summing

            if header['OBSRVTRY'] == 'STEREO_B':
                mu = 0.68886
                cdelt = 129.80319 / 3600 * np.pi / 180 * summing

    pixelSize = 0.0135 * summing
    fp = pixelSize / cdelt

    x = np.arange(header['naxis1']) - header['crpix1'] + header['dstart1']
    x = x[:,None].repeat(header['naxis2'],1)
    # x = np.array([x for i in range(header['naxis1'])])

    y = np.arange(header['naxis2']) - header['crpix2'] + header['dstart2']
    # y = np.transpose(y)
    # y = np.array([y for i in range(header['naxis1'])])
    y = y[None,:].repeat(header['naxis1'],0)

    r = np.sqrt(x * x + y * y) * pixelSize

    gamma = fp * (mu + 1.0) / r
    cosalpha1 = (-1.0 * mu + gamma * np.sqrt(1.0 - mu * mu + gamma * gamma)) / (1.0 + gamma * gamma)

    correct = ((mu + 1.0) ** 2 * (mu * cosalpha1 + 1.0)) / ((mu + cosalpha1) ** 3)

    return correct
