
from astropy import wcs
import sys
import numpy as np 
from astropy.io import fits
from scipy.ndimage import zoom
from datetime import datetime,timedelta
import matplotlib.pyplot as plt 
from natsort import natsorted
import glob 
from skimage.transform import resize
from PIL import Image



def get_calfac(hdr, conv='MSB', silent=True):

    hdr_dateavg = datetime.strptime(hdr['date-avg'], '%Y-%m-%dT%H:%M:%S.%f')

    if hdr['DETECTOR'] == 'COR1':

        if hdr['OBSRVTRY'] == 'STEREO_A':
            calfac = 6.578E-11
            tai0 = datetime.strptime('2007-12-01T03:41:48.174', '%Y-%m-%dT%H:%M:%S.%f')
            rate = 0.00648

        elif hdr['OBSRVTRY'] == 'STEREO_B':
            calfac = 7.080E-11
            tai0 = datetime.strptime('2008-01-17T02:20:15.717', '%Y-%m-%dT%H:%M:%S.%f')
            rate = 0.00258
        
        years = (hdr_dateavg - tai0).total_seconds() / (3600. * 24 * 365.25)
        calfac = calfac / (1 - rate * years)
    
    elif hdr['DETECTOR'] == 'COR2':

        if hdr['OBSRVTRY'] == 'STEREO_A':
            calfac = 2.7E-12 * 0.5

        elif hdr['OBSRVTRY'] == 'STEREO_B':
            calfac = 2.8E-12 * 0.5
    
    elif hdr['DETECTOR'] == 'EUVI':
        gain = 15.0
        calfac = gain * (3.65 * hdr['WAVELNTH']) / (13.6 * 911)
    
    elif hdr['DETECTOR'] == 'HI1':

        if hdr['OBSRVTRY'] == 'STEREO_A':
            years = (hdr_dateavg - datetime.strptime('2011-06-27T00:00:00.000', '%Y-%m-%dT%H:%M:%S.%f')).total_seconds() / (3600 * 24 * 365.25)
            
            ## commented this, otherwise why would it be used in paper 
            if years < 0:
                years = 0

            if conv == 's10':
                calfac = 763.2 + 1.315 * years
                # calfac = (3.453e-13 + 5.914e-16 * years )
            else:
                calfac = 3.453e-13 + 5.914e-16 * years
                # calfac = 3.453E-13 * 2.223e15

            hdr['HISTORY'] = 'revised calibration Tappin et al Solar Physics 2022 DOI 10.1007/s11207-022-01966-x'

        elif hdr['OBSRVTRY'] == 'STEREO_B':
            years = (hdr_dateavg - datetime.strptime('2007-01-01T00:00:00.000', '%Y-%m-%dT%H:%M:%S.%f')).total_seconds() / (3600 * 24 * 365.25)

            if years < 0:
                years = 0

            if conv == 's10':
                calfac = 790.0 + 0.001503*years
                ## TODO where is the year factor for HI1 STEREO B?
            else:
                annualchange = 0.001503
                calfac = 3.55E-13
                calfac = calfac / (1 - annualchange * years)


            hdr['HISTORY'] = 'revised calibration Tappin et al Solar Physics 2017 DOI 10.1007/s11207-017-1052-0'
    
    elif hdr['DETECTOR'] == 'HI2':

        if hdr['OBSRVTRY'] == 'STEREO_A':

            years = (hdr_dateavg - datetime.strptime('2015-01-01T00:00:00.000', '%Y-%m-%dT%H:%M:%S.%f')).total_seconds() / (3600 * 24 * 365.25)
            
            if years < 0:
                if conv == 's10':
                    calfac = 99.5 + 0.1225 * years
                else:
                    calfac = 4.476E-14 + 5.511E-17 * years
            else:
                if conv == 's10':
                    calfac = 100.3 + 0.1580 * years
                else:
                    calfac = 4.512E-14 + 7.107E-17 * years

            hdr['HISTORY'] = 'revised calibration Tappin et al Solar Physics 2022 DOI 10.1007/s11207-022-01966-x'

        elif hdr['OBSRVTRY'] == 'STEREO_B':
            years = (hdr_dateavg - datetime.strptime('2000-12-31T00:00:00.000', '%Y-%m-%dT%H:%M:%S.%f')).total_seconds() / (3600 * 24 * 365.25)
            if conv == 's10':
                calfac = 95.424 + 0.067 * years
            else:
                calfac = 4.293E-14 + 3.014E-17 * years
    
    hdr['calfac'] = calfac
    if 'ipsum' in hdr and hdr['ipsum'] > 1 and calfac != 1.0:
        divfactor = (2 ** (hdr['ipsum'] - 1)) ** 2
        hdr['ipsum'] = 1
        calfac = calfac / divfactor

        if not silent:
            print(f'Divided calfac by {divfactor} to account for IPSUM')
            print('IPSUM changed to 1 in header.')

        hdr['HISTORY'] =  f'get_calfac Divided calfac by {divfactor} to account for IPSUM'

    if 'polar' in hdr and hdr['polar'] == 1001 and hdr.get('seb_prog') != 'DOUBLE':
        calfac *= 2

        if not silent:
            print('Applied factor of 2 for total brightness')

        hdr['HISTORY'] =  'get_calfac Applied factor of 2 for total brightness'

    return calfac, hdr


def scc_icerdiv2(i, d, pipeline=False, silent=True):
    """
    Correct for conditional DIV2 by on-board IP prior to ICER.

    Parameters:
    i (dict): Header index structure containing necessary tags.
    d (np.array): Image data array, replaced by corrected data array.
    pipeline (bool): If True, pipeline is set. Default is False.
    silent (bool): If True, suppress print statements. Default is True.

    Returns:
    dict: Updated header index structure.
    np.array: Updated image data array.
    str: Updated ICER message.
    """
    # Info for logging
    info = "$Id: scc_icerdiv2.pro,v 1.19 2011/09/15 21:57:35 nathan Exp $"
    histinfo = info[1:-2]

    # Get the IP commands
    ip = i['IP_00_19']
    if len(ip) < 60:
        ip = ' ' + ip
    if len(ip) < 60:
        ip = ' ' + ip
    ip = np.array([np.int8(x) for x in ip], dtype=np.int8).reshape(3, 20)

    if not silent:
        print('IP_00_19:', ip)

    w = np.where(ip != 0)
    nip = len(w[0])

    icradiv2 = 0
    idecdiv2 = 0
    icramsg = ''
    datap01 = i['DATAP01']
    biasmean = i['BIASMEAN']

    if pipeline:
        print('Pipeline = True not implemented yet - should not be used on L0.5 data.')
        sys.exit()

    # Calculate various conditions
    icer = 90 <= ip[nip-1] <= 102
    div2 = ip[nip-2] == 1
    noticfilt = ip[nip-2] < 106 or ip[nip-2] > 112
    nosubbias = np.where(ip == 103)[0].size == 0
    biasmp01 = (biasmean / 2) - datap01
    p01ltbias = abs(biasmp01) < 0.02 * (biasmean / 2)

    # Logic to determine whether data was most likely divided by 2
    domul2 = icradiv2 or idecdiv2 or (icer and noticfilt and nosubbias and p01ltbias)

    if not silent:
        print(f'{icradiv2}=icradiv2, {idecdiv2}=idecdiv2, {icer}=icer, {noticfilt}=noticfilt, {nosubbias}=nosubbias, {p01ltbias}=p01ltbias')

    if pipeline:
        print('Pipeline = True not implemented yet - should not be used on L0.5 data.')
        sys.exit()
        
    # Apply correction
    if domul2:
        m2 = np.array(2, dtype=d.dtype)
        d *= m2
        for key in ['DATAP01', 'DATAMIN', 'DATAMAX', 'DATAAVG', 'DATAP10', 'DATAP25', 'DATAP75', 'DATAP90', 'DATAP95', 'DATAP98', 'DATAP99']:
            i[key] *= 2
        i['DIV2CORR'] = 'T'

        if idecdiv2 and icradiv2:
            i['DIV2CORR'] = 'F'

        if not silent:
            print('Image corrected by icerdiv2')

        icramsg = 'Corrected for icerdiv2 because: ' + icramsg

    else:
        icramsg = 'No div2 correction: ' + icramsg

    if not silent:
        print(icramsg)

    if pipeline:
        print('Pipeline = True not implemented yet - should not be used on L0.5 data.')
        sys.exit()

    if datap01 < 0.75 * biasmean:
        if not silent:
            print('datap01:', datap01, 'biasmean:', biasmean)
            print(0.02*(biasmean/2))

    h_dex = 20 - np.sum([x == '' for x in i['HISTORY']])
    i['HISTORY'][h_dex] = histinfo

    return i, d



def precommcorrect(im, hdr, extra = None, silent=True):
    """
    Apply corrections to images taken before all commissioning data is reprocessed.
    
    Args:
    im (np.ndarray): level 0.5 image from sccreadfits
    hdr (dict): level 0.5 header from sccreadfits, SECCHI structure
    extra (dict): Extra information for pointing correction (COR, EUVI only)
    silent (bool, optional): Suppress print statements if True
    """
    
    # Apply IcerDiv2 correction (Bug 49)
    if 89 < hdr['comprssn'] < 102:
        if hdr['DIV2CORR'] == 'F':
            im, hdr = scc_icerdiv2(hdr, im)
        else:
            biasmean = hdr['biasmean']
            p01mbias = hdr['datap01'] - biasmean
            if not silent:
                print(f'p01mbias: {p01mbias}, biasmean: {biasmean}')
            if p01mbias > 0.8 * hdr['biasmean']:
                im = im/2
                for key in ['datap01', 'datamin', 'datamax', 'dataavg', 'datap10', 'datap25', 'datap75', 'datap90', 'datap95', 'datap98', 'datap99']:
                    hdr[key] /= 2
                hdr['div2corr'] = 'F'

                if not silent:
                    print('Image corrected for incorrect icerdiv2', info=True)

    hdr['mask_tbl'] = 'NONE'

    # Correct Image Center
    if hdr['DETECTOR'] == 'EUVI':
        print('Precommcorrect not implemented for EUVI')
        sys.exit()
        #euvi_point(hdr, quiet=silent)
    elif hdr['DETECTOR'] == 'COR1':
        print('Precommcorrect not implemented for COR1')
        sys.exit()
        #cor1_point(hdr, SILENT=silent, **ex)
    elif hdr['DETECTOR'] == 'COR2':
        print('Precommcorrect not implemented for COR2')
        sys.exit()
        #cor2_point(hdr, SILENT=silent, **ex)

    # Add DSTART(STOP)1(2)
    if hdr['DSTOP1'] < 1 or hdr['DSTOP1'] > hdr['NAXIS1'] or hdr['DSTOP2'] > hdr['NAXIS2']:
        x1 = max(0, 51 - hdr['P1COL'])
        x2 = min(2048 + x1 - 1, hdr['P2COL'] - hdr['P1COL'])
        y1 = max(0, 1 - hdr['P1ROW'])
        y2 = min(2048 + y1 - 1, hdr['P2ROW'] - hdr['P1ROW'])

        if hdr['P1COL'] < 51:
            hdr['P1COL'] = 51

        hdr['P2COL'] = hdr['P1COL'] + (x2 - x1)

        if hdr['P1ROW'] < 1:
            hdr['P1ROW'] = 1

        hdr['P2ROW'] = hdr['P1ROW'] + (y2 - y1)

        x1 = int(x1 / 2 ** (hdr['summed'] - 1))
        x2 = int((hdr['P2COL'] - hdr['P1COL'] + 1) / 2 ** (hdr['summed'] - 1)) + x1 - 1
        y1 = int(y1 / 2 ** (hdr['summed'] - 1))
        y2 = int((hdr['P2ROW'] - hdr['P1ROW'] + 1) / 2 ** (hdr['summed'] - 1)) + y1 - 1

        if hdr['RECTIFY'] == True:
            if hdr['OBSRVTRY'] == 'STEREO_A':
                if hdr['DETECTOR'] == 'EUVI':
                    rx1 = hdr['naxis1'] - y2 - 1
                    rx2 = hdr['naxis1'] - y1 - 1
                    ry1 = hdr['naxis2'] - x2 - 1
                    ry2 = hdr['naxis2'] - x1 - 1
                    hdr['R1COL'] = 2176 - hdr['P2ROW'] + 1
                    hdr['R2COL'] = 2176 - hdr['P1ROW'] + 1
                    hdr['R1ROW'] = 2176 - hdr['P2COL'] + 1
                    hdr['R2ROW'] = 2176 - hdr['P1COL'] + 1

                    print('Precommcorrect not implemented for EUVI')
                    sys.exit()

                elif hdr['DETECTOR'] == 'COR1':
                    rx1 = y1
                    rx2 = y2
                    ry1 = hdr['naxis2'] - x2 - 1
                    ry2 = hdr['naxis2'] - x1 - 1
                    hdr['R1COL'] = hdr['P1ROW']
                    hdr['R2COL'] = hdr['P2ROW']
                    hdr['R1ROW'] = 2176 - hdr['P2COL'] + 1
                    hdr['R2ROW'] = 2176 - hdr['P1COL'] + 1

                    print('Precommcorrect not implemented for COR1')
                    sys.exit()

                elif hdr['DETECTOR'] == 'COR2':
                    rx1 = hdr['naxis1'] - y2 - 1
                    rx2 = hdr['naxis1'] - y1 - 1
                    ry1 = x1
                    ry2 = x2
                    hdr['R1COL'] = 2176 - hdr['P2ROW'] + 1
                    hdr['R2COL'] = 2176 - hdr['P1ROW'] + 1
                    hdr['R1ROW'] = hdr['P1COL']
                    hdr['R2ROW'] = hdr['P2COL']
                    print('Precommcorrect not implemented for COR2')
                    sys.exit()

                elif hdr['DETECTOR'] == 'HI1':
                    rx1 = x1
                    rx2 = x2
                    ry1 = y1
                    ry2 = y2
                    hdr['R1COL'] = hdr['P1COL']
                    hdr['R2COL'] = hdr['P2COL']
                    hdr['R1ROW'] = hdr['P1ROW']
                    hdr['R2ROW'] = hdr['P2ROW']

                elif hdr['DETECTOR'] == 'HI2':
                    rx1 = x1
                    rx2 = x2
                    ry1 = y1
                    ry2 = y2
                    hdr['R1COL'] = hdr['P1COL']
                    hdr['R2COL'] = hdr['P2COL']
                    hdr['R1ROW'] = hdr['P1ROW']
                    hdr['R2ROW'] = hdr['P2ROW']

            elif hdr['OBSRVTRY'] == 'STEREO_B':
                if hdr['DETECTOR'] == 'EUVI':
                    rx1 = y1
                    rx2 = y2
                    ry1 = hdr['naxis2'] - x2 - 1
                    ry2 = hdr['naxis2'] - x1 - 1
                    hdr['R1COL'] = hdr['P1ROW']
                    hdr['R2COL'] = hdr['P2ROW']
                    hdr['R1ROW'] = 2176 - hdr['P2COL'] + 1
                    hdr['R2ROW'] = 2176 - hdr['P1COL'] + 1
                    print('Precommcorrect not implemented for EUVI')
                    sys.exit()

                elif hdr['DETECTOR'] == 'COR1':
                    rx1 = hdr['naxis1'] - y2 - 1
                    rx2 = hdr['naxis1'] - y1 - 1
                    ry1 = x1
                    ry2 = x2
                    hdr['R1COL'] = 2176 - hdr['P2ROW'] + 1
                    hdr['R2COL'] = 2176 - hdr['P1ROW'] + 1
                    hdr['R1ROW'] = hdr['P1COL']
                    hdr['R2ROW'] = hdr['P2COL']
                    print('Precommcorrect not implemented for COR1')
                    sys.exit()

                elif hdr['DETECTOR'] == 'COR2':
                    rx1 = y1
                    rx2 = y2
                    ry1 = hdr['naxis2'] - x2 - 1
                    ry2 = hdr['naxis2'] - x1 - 1
                    hdr['R1COL'] = hdr['P1ROW']
                    hdr['R2COL'] = hdr['P2ROW']
                    hdr['R1ROW'] = 2176 - hdr['P2COL'] + 1
                    hdr['R2ROW'] = 2176 - hdr['P1COL'] + 1
                    print('Precommcorrect not implemented for COR2')
                    sys.exit()

                elif hdr['DETECTOR'] == 'HI1':
                    rx1 = hdr['naxis1'] - x2 - 1
                    rx2 = hdr['naxis1'] - x1 - 1
                    ry1 = hdr['naxis2'] - y2 - 1
                    ry2 = hdr['naxis2'] - y1 - 1
                    hdr['R1COL'] = 2176 - hdr['P2ROW']
                    hdr['R2COL'] = 2176 - hdr['P1ROW']
                    hdr['R1ROW'] = 2176 - hdr['P2COL']
                    hdr['R2ROW'] = 2176 - hdr['P1COL']

                elif hdr['DETECTOR'] == 'HI2':
                    rx1 = hdr['naxis1'] - x2 - 1
                    rx2 = hdr['naxis1'] - x1 - 1
                    ry1 = hdr['naxis2'] - y2 - 1
                    ry2 = hdr['naxis2'] - y1 - 1
                    hdr['R1COL'] = 2176 - hdr['P2ROW']
                    hdr['R2COL'] = 2176 - hdr['P1ROW']
                    hdr['R1ROW'] = 2176 - hdr['P2COL']
                    hdr['R2ROW'] = 2176 - hdr['P1COL']

            x1 = rx1
            x2 = rx2
            y1 = ry1
            y2 = ry2

        hdr['DSTART1'] = x1+1
        hdr['DSTART2'] = y1+1
        hdr['DSTOP1'] = x2+1
        hdr['DSTOP2'] = y2+1

    return im, hdr
def secchi_rectify(a, scch, hdr=None, norotate=False, silent=True):

    ## CHANGE Added history, changed dstart1, dstart2 (< function behaves differently in IDL)

    info = "$Id: secchi_rectify.pro,v 1.29 2023/08/14 17:50:07 secchia Exp $"
    histinfo = info[1:-2]

    if scch['rectify'] == True:
        if not silent:
            print('RECTIFY=T -- Returning with no changes')
        return a

    crval1 = scch['crval1']

    if scch['OBSRVTRY'] == 'STEREO_A':    
        post_conj = int(np.sign(crval1))

    if scch['OBSRVTRY'] == 'STEREO_B':    
        post_conj = int(-1*np.sign(crval1))

    if post_conj == -1:
        post_conj = False
    if post_conj == 1:
        post_conj = True
        
    stch = scch.copy()
    
    ## TODO implement other detectors

    if not norotate:
        stch['rectify'] = True

        if scch['OBSRVTRY'] == 'STEREO_A' and post_conj == 0:
            if scch['detector'] == 'EUVI':
                # b = np.rot90(a.T, 2)
                # stch['r1row'] = 2176 - scch['p2col'] + 1
                # stch['r2row'] = 2176 - scch['p1col'] + 1
                # stch['r1col'] = 2176 - scch['p2row'] + 1
                # stch['r2col'] = 2176 - scch['p1row'] + 1
                # stch['crpix1'] = scch['naxis2'] - scch['crpix2'] + 1
                # stch['crpix2'] = scch['naxis1'] - scch['crpix1'] + 1
                # stch['naxis1'], stch['naxis2'] = scch['naxis2'], scch['naxis1']
                # stch['sumrow'] = scch['sumcol']
                # stch['sumcol'] = scch['sumrow']
                # stch['rectrota'] = 6
                # rotcmt = 'transpose and rotate 180 deg CCW'
                # stch['dstart1'] = max(1, 129 - stch['r1col'] + 1)
                # stch['dstop1'] = stch['dstart1'] - 1 + min((stch['r2col'] - stch['r1col'] + 1), 2048)
                # stch['dstart2'] = max(1, 79 - stch['r1row'] + 1)
                # stch['dstop2'] = stch['dstart2'] - 1 + min((stch['r2row'] - stch['r1row'] + 1), 2048)
                print('Rectify not implemented for EUVI')
                sys.exit()

            elif scch['detector'] == 'COR1':
                # b = np.rot90(a, 3)
                # stch['r1row'] = 2176 - scch['p2col'] + 1
                # stch['r2row'] = 2176 - scch['p1col'] + 1
                # stch['r1col'] = scch['p1row']
                # stch['r2col'] = scch['p2row']
                # stch['crpix1'] = scch['crpix2']
                # stch['crpix2'] = scch['naxis1'] - scch['crpix1'] + 1
                # stch['naxis1'], stch['naxis2'] = scch['naxis2'], scch['naxis1']
                # stch['sumrow'] = scch['sumcol']
                # stch['sumcol'] = scch['sumrow']
                # stch['rectrota'] = 3
                # rotcmt = 'rotate 270 deg CCW'
                # stch['dstart1'] = 1
                # stch['dstop1'] = stch['dstart1'] - 1 + min((stch['r2col'] - stch['r1col'] + 1), 2048)
                # stch['dstart2'] = max(1, 79 - stch['r1row'] + 1)
                # stch['dstop2'] = stch['dstart2'] - 1 + min((stch['r2row'] - stch['r1row'] + 1), 2048)
                print('Rectify not implemented for COR1')
                sys.exit()

            elif scch['detector'] == 'COR2':
                b = np.rot90(a, 1)
                stch['r1row'] = scch['p1col']
                stch['r2row'] = scch['p2col']
                stch['r1col'] = 2176 - scch['p2row'] + 1
                stch['r2col'] = 2176 - scch['p1row'] + 1
                stch['crpix1'] = scch['naxis2'] - scch['crpix2'] + 1
                stch['crpix2'] = scch['crpix1']
                stch['naxis1'], stch['naxis2'] = scch['naxis2'], scch['naxis1']
                stch['sumrow'] = scch['sumcol']
                stch['sumcol'] = scch['sumrow']
                stch['rectrota'] = 1
                rotcmt = 'rotate 90 deg CCW'
                stch['dstart1'] = max(1, 129 - stch['r1col'] + 1)
                stch['dstop1'] = stch['dstart1'] - 1 + min((stch['r2col'] - stch['r1col'] + 1), 2048)
                stch['dstart2'] = max(1, 51 - stch['r1row'] + 1)
                stch['dstop2'] = stch['dstart2'] - 1 + min((stch['r2row'] - stch['r1row'] + 1), 2048)
                print("--------")
            elif scch['detector'] in ['HI1', 'HI2']:
                b = a  # no change
                stch['r1row'] = scch['p1row']
                stch['r2row'] = scch['p2row']
                stch['r1col'] = scch['p1col']
                stch['r2col'] = scch['p2col']
                stch['rectrota'] = 0
                rotcmt = 'no rotation necessary'

        elif scch['OBSRVTRY'] == 'STEREO_B' and post_conj == 0:
            if scch['detector'] == 'EUVI':
                # b = np.rot90(a, 3)
                # stch['r1row'] = 2176 - scch['p2col'] + 1
                # stch['r2row'] = 2176 - scch['p1col'] + 1
                # stch['r1col'] = scch['p1row']
                # stch['r2col'] = scch['p2row']
                # stch['crpix1'] = scch['crpix2']
                # stch['crpix2'] = scch['naxis1'] - scch['crpix1'] + 1
                # stch['naxis1'], stch['naxis2'] = scch['naxis2'], scch['naxis1']
                # stch['sumrow'] = scch['sumcol']
                # stch['sumcol'] = scch['sumrow']
                # stch['rectrota'] = 3
                # rotcmt = 'rotate 270 deg CCW'
                # stch['dstart1'] = 1
                # stch['dstop1'] = stch['dstart1'] - 1 + min((stch['r2col'] - stch['r1col'] + 1), 2048)
                # stch['dstart2'] = max(1, 79 - stch['r1row'] + 1)
                # stch['dstop2'] = stch['dstart2'] - 1 + min((stch['r2row'] - stch['r1row'] + 1), 2048)
                print('Rectify not implemented for EUVI')
                sys.exit()

            elif scch['detector'] == 'COR1':
                # b = np.rot90(a, 1)
                # stch['r1row'] = scch['p1col']
                # stch['r2row'] = scch['p2col']
                # stch['r1col'] = 2176 - scch['p2row'] + 1
                # stch['r2col'] = 2176 - scch['p1row'] + 1
                # stch['crpix1'] = scch['naxis2'] - scch['crpix2'] + 1
                # stch['crpix2'] = scch['crpix1']
                # stch['naxis1'], stch['naxis2'] = scch['naxis2'], scch['naxis1']
                # stch['sumrow'] = scch['sumcol']
                # stch['sumcol'] = scch['sumrow']
                # stch['rectrota'] = 1
                # rotcmt = 'rotate 90 deg CCW'
                # stch['dstart1'] = max(1, 51 - stch['r1col'] + 1)
                # stch['dstop1'] = stch['dstart1'] - 1 + min((stch['r2col'] - stch['r1col'] + 1), 2048)
                # stch['dstart2'] = max(1, 129 - stch['r1row'] + 1)
                # stch['dstop2'] = stch['dstart2'] - 1 + min((stch['r2row'] - stch['r1row'] + 1), 2048)

                print('Rectify not implemented for COR1')
                sys.exit()

            elif scch['detector'] == 'COR2':
                # b = np.rot90(a, 3)
                # stch['r1row'] = 2176 - scch['p2col'] + 1
                # stch['r2row'] = 2176 - scch['p1col'] + 1
                # stch['r1col'] = 2176 - scch['p2row'] + 1
                # stch['r2col'] = 2176 - scch['p1row'] + 1
                # stch['crpix1'] = scch['naxis2'] - scch['crpix2'] + 1
                # stch['crpix2'] = scch['naxis1'] - scch['crpix1'] + 1
                # stch['naxis1'], stch['naxis2'] = scch['naxis2'], scch['naxis1']
                # stch['sumrow'] = scch['sumcol']
                # stch['sumcol'] = scch['sumrow']
                # stch['rectrota'] = 3
                # rotcmt = 'rotate 270 deg CCW'
                # stch['dstart1'] = max(1, 129 - stch['r1col'] + 1)
                # stch['dstop1'] = stch['dstart1'] - 1 + min((stch['r2col'] - stch['r1col'] + 1), 2048)
                # stch['dstart2'] = max(1, 79 - stch['r1row'] + 1)
                # stch['dstop2'] = stch['dstart2'] - 1 + min((stch['r2row'] - stch['r1row'] + 1), 2048)

                print('Rectify not implemented for COR2')
                sys.exit()

            elif scch['detector'] in ['HI1', 'HI2']:

                b = np.rot90(a, 2)
                stch['r1row'] = 2176 - scch['p2row'] + 1
                stch['r2row'] = 2176 - scch['p1row'] + 1
                stch['r1col'] = 2176 - scch['p2col'] + 1
                stch['r2col'] = 2176 - scch['p1col'] + 1

                stch['crpix1'] = scch['naxis1'] - scch['crpix1'] + 1
                stch['crpix2'] = scch['naxis2'] - scch['crpix2'] + 1
                stch['naxis1'] = scch['naxis1']
                stch['naxis2'] = scch['naxis2']

                stch['rectrota'] = 2
                rotcmt = 'rotate 180 deg CCW'

                stch['dstart1'] = max(1, 79 - stch['r1col'] + 1)
                stch['dstop1'] = stch['dstart1'] - 1 + min((stch['r2col'] - stch['r1col'] + 1), 2048)
                stch['dstart2'] = max(1, 129 - stch['r1row'] + 1)
                stch['dstop2'] = stch['dstart2'] - 1 + min((stch['r2row'] - stch['r1row'] + 1), 2048)
        
        elif scch['OBSRVTRY'] == 'STEREO_A' and post_conj == 1:

            if scch['detector'] == 'EUVI':

                # b = a.T
                # stch['r1row'] = scch['p1col']
                # stch['r2row'] = scch['p2col']
                # stch['r1col'] = scch['p1row']
                # stch['r2col'] = scch['p2row']
                # stch['crpix1'] = scch['naxis1'] - scch['crpix2'] + 1
                # stch['crpix2'] = scch['naxis2'] - scch['crpix1'] + 1
                # stch['naxis1'] = scch['naxis2']
                # stch['naxis2'] = scch['naxis1']
                # stch['sumrow'] = scch['sumcol']
                # stch['sumcol'] = scch['sumrow']
                # stch['rectrota'] = 4
                # rotcmt = 'transpose'
                # stch['dstart1'] = max(1, 129 - stch['r1col'] + 1)
                # stch['dstop1'] = stch['dstart1'] - 1 + min((stch['r2col'] - stch['r1col'] + 1), 2048)
                # stch['dstart2'] = max(1, 79 - stch['r1row'] + 1)
                # stch['dstop2'] = stch['dstart2'] - 1 + min((stch['r2row'] - stch['r1row'] + 1), 2048)

                print('Rectify not implemented for EUVI')
                sys.exit()

            elif scch['detector'] == 'COR1':

                # b = np.rot90(a, 1)
                # stch['r1row'] = scch['p1col']
                # stch['r2row'] = scch['p2col']
                # stch['r1col'] = 2176 - scch['p2row'] + 1
                # stch['r2col'] = 2176 - scch['p1row'] + 1
                # stch['crpix1'] = scch['naxis2'] - scch['crpix2'] + 1
                # stch['crpix2'] = stch['crpix1']
                # stch['naxis1'] = scch['naxis2']
                # stch['naxis2'] = scch['naxis1']
                # stch['sumrow'] = scch['sumcol']
                # stch['sumcol'] = scch['sumrow']
                # stch['rectrota'] = 1
                # rotcmt = 'rotate 90 deg CCW'
                # stch['dstart1'] = max(1, 129 - stch['r1col'] + 1)
                # stch['dstop1'] = stch['dstart1'] - 1 + min((stch['r2col'] - stch['r1col'] + 1), 2048)
                # stch['dstart2'] = max(1, 51 - stch['r1row'] + 1)
                # stch['dstop2'] = stch['dstart2'] - 1 + min((stch['r2row'] - stch['r1row'] + 1), 2048)

                print('Rectify not implemented for COR1')
                sys.exit()

            elif scch['detector'] == 'COR2':
                
                # b = np.rot90(a, 3)
                # stch['r1row'] = 2176 - scch['p2col'] + 1
                # stch['r2row'] = 2176 - scch['p1col'] + 1
                # stch['r1col'] = scch['p1row']
                # stch['r2col'] = scch['p2row']
                # stch['crpix1'] = scch['crpix2']
                # stch['crpix2'] = scch['naxis1'] - scch['crpix1'] + 1
                # stch['naxis1'] = scch['naxis2']
                # stch['naxis2'] = scch['naxis1']
                # stch['sumrow'] = scch['sumcol']
                # stch['sumcol'] = scch['sumrow']
                # stch['rectrota'] = 3
                # rotcmt = 'rotate 270 deg CCW'
                # stch['dstart1'] = 1
                # stch['dstop1'] = stch['dstart1'] - 1 + min((stch['r2col'] - stch['r1col'] + 1), 2048)
                # stch['dstart2'] = max(1, 79 - stch['r1row'] + 1)
                # stch['dstop2'] = stch['dstart2'] - 1 + min((stch['r2row'] - stch['r1row'] + 1), 2048)

                print('Rectify not implemented for COR2')
                sys.exit()

            elif scch['detector'] in ['HI1', 'HI2']:

                b = np.rot90(a, 2)
                stch['r1row'] = 2176 - scch['p2row'] + 1
                stch['r2row'] = 2176 - scch['p1row'] + 1
                stch['r1col'] = 2176 - scch['p2col'] + 1
                stch['r2col'] = 2176 - scch['p1col'] + 1
                stch['crpix1'] = scch['naxis1'] - scch['crpix1'] + 1
                stch['crpix2'] = scch['naxis2'] - scch['crpix2'] + 1
                stch['naxis1'] = scch['naxis1']
                stch['naxis2'] = scch['naxis2']
                stch['rectrota'] = 2
                rotcmt = 'rotate 180 deg CCW'
                stch['dstart1'] = max(1, 79 - stch['r1col'] + 1)
                stch['dstop1'] = stch['dstart1'] - 1 + min((stch['r2col'] - stch['r1col'] + 1), 2048)
                stch['dstart2'] = max(1, 129 - stch['r1row'] + 1)
                stch['dstop2'] = stch['dstart2'] - 1 + min((stch['r2row'] - stch['r1row'] + 1), 2048)

            else:
                b = a  # If detector is not recognized, return the original image
                rotcmt = None

        elif scch['OBSRVTRY'] == 'STEREO_B' and post_conj == 1:
            print('Case of ST-B with post_conj = True not implemented. Exiting...')
            sys.exit()

        
    else:

        stch['rectify'] = False
        b = a  # no rotation performed

        stch['r1row'] = scch['p1row']
        stch['r2row'] = scch['p2row']
        stch['r1col'] = scch['p1col']
        stch['r2col'] = scch['p2col']
        stch['rectrota'] = 0
        rotcmt = 'no rotation necessary'

    if stch['r1col'] < 1:
        stch['r2col'] += np.abs(stch['r1col']) + 1
        stch['r1col'] = 1
        
    if stch['r1row'] < 1:
        stch['r2row'] += np.abs(stch['r1row']) + 1
        stch['r1row'] = 1

    xden = 2 ** (scch['ipsum'] + scch['sumcol'] - 2)
    yden = 2 ** (scch['ipsum'] + scch['sumrow'] - 2)

    stch['dstart1'] = max(int(np.ceil(float(stch['dstart1']) / xden)), 1)
    stch['dstart2'] = max(int(np.ceil(float(stch['dstart2']) / yden)), 1)
    stch['dstop1'] = int(float(stch['dstop1']) / xden)
    stch['dstop2'] = int(float(stch['dstop2']) / yden)


    if stch['NAXIS1'] > 0 and stch['NAXIS2'] > 0:
         
        try:
            wcoord = wcs.WCS(stch)
            xycen = wcoord.all_pix2world((stch['naxis1'] - 1.) / 2., (stch['naxis2'] - 1.) / 2., 0)

            stch['xcen'] = float(xycen[0])
            stch['ycen'] = float(xycen[1])

        except wcs.SingularMatrixError:

            stch['xcen'] = 9999.0
            stch['ycen'] = 9999.0

    if hdr is not None:

        hdr['NAXIS1'] = stch['naxis1']
        hdr['NAXIS2'] = stch['naxis2']
        hdr['R1COL'] = stch['r1col']
        hdr['R2COL'] = stch['r2col']
        hdr['R1ROW'] = stch['r1row']
        hdr['R2ROW'] = stch['r2row']
        hdr['SUMROW'] = stch['sumrow']
        hdr['SUMCOL'] = stch['sumcol']
        hdr['RECTIFY'] = stch['rectify']
        hdr['CRPIX1'] = stch['crpix1']
        hdr['CRPIX2'] = stch['crpix2']
        hdr['XCEN'] = stch['xcen']
        hdr['YCEN'] = stch['ycen']
        hdr['CRPIX1A'] = stch['crpix1']
        hdr['CRPIX2A'] = stch['crpix2']
        hdr['DSTART1'] = stch['dstart1']
        hdr['DSTART2'] = stch['dstart2']
        hdr['DSTOP1'] = stch['dstop1']
        hdr['DSTOP2'] = stch['dstop2']
        hdr['HISTORY'] = histinfo  # Assuming histinfo is defined
        hdr['RECTROTA'] = f"{stch['rectrota']} {rotcmt}"  # Assuming rotcmt is defined

    scch = stch

    if norotate:
        if not silent:
            print('norotate set -- Image returned unchanged')
        
        return a, scch
    
    else:
        if not silent:
            print(f'Rectification applied to {scch["filename"]}: {rotcmt}')

        return b, scch
    

def fix_secchi_hdr(hdr):
    
    # Initialize default values
    int_val = 0
    lon_val = 0
    flt_val = 0.0
    fltd_val = 0.0
    uint8_val = 0
    uint16_val = 0
    uint32_val = 0
    int8_val = 0
    int16_val = 0
    int32_val = 0
    str_val = ''
    sta_val = [str_val] * 20

    secchi_hdr = {
        # SIMPLE: 'T', 
        'EXTEND': 'F', 

        # Most-used keywords
        'BITPIX': int_val, 
        'NAXIS': int_val, 
        'NAXIS1': int_val, 
        'NAXIS2': int_val,
        
        'DATE-OBS': str_val, 
        'TIME_OBS': str_val, 
        'FILEORIG': str_val, 
        'SEB_PROG': str_val, 
        'SYNC': str_val, 
        'SPWX': 'F', 
        'EXPCMD': -1.0, 
        'EXPTIME': -1.0, 
        'DSTART1': int_val, 
        'DSTOP1': int_val, 
        'DSTART2': int_val, 
        'DSTOP2': int_val,
        'P1COL': int16_val, 
        'P2COL': int16_val, 
        'P1ROW': int16_val, 
        'P2ROW': int16_val, 
        'R1COL': int16_val, 
        'R2COL': int16_val, 
        'R1ROW': int16_val, 
        'R2ROW': int16_val, 
        'RECTIFY': 'F', 
        'RECTROTA': int_val, 
        'LEDCOLOR': str_val, 
        'LEDPULSE': uint32_val, 
        'OFFSET': 9999, 
        'BIASMEAN': flt_val, 
        'BIASSDEV': -1.0, 
        'GAINCMD': -1, 
        'GAINMODE': str_val, 
        'SUMMED': flt_val, 
        'SUMROW': 1, 
        'SUMCOL': 1, 
        'CEB_T': 999, 
        'TEMP_CCD': 9999.0, 
        'POLAR': -1.0, 
        'ENCODERP': -1, 
        'WAVELNTH': int_val, 
        'ENCODERQ': -1, 
        'FILTER': str_val, 
        'ENCODERF': -1, 
        'FPS_ON': str_val, 
        'OBS_PROG': 'schedule', 
        'DOORSTAT': -1, 
        'SHUTTDIR': str_val, 
        'READ_TBL': -1, 
        'CLR_TBL': -1, 
        'READFILE': str_val, 
        'DATE-CLR': str_val, 
        'DATE-RO': str_val, 
        'READTIME': -1.0, 
        'CLEARTIM': fltd_val, 
        'IP_TIME': -1, 
        'COMPRSSN': int_val, 
        'COMPFACT': flt_val, 
        'NMISSING': -1.0, 
        'MISSLIST': str_val, 
        'SETUPTBL': str_val, 
        'EXPOSTBL': str_val, 
        'MASK_TBL': str_val, 
        'IP_TBL': str_val, 
        'COMMENT': sta_val, 
        'HISTORY': sta_val, 
        'DIV2CORR': 'F', 
        'DISTCORR': 'F', 

        # Less-used keywords
        'TEMPAFT1': 9999.0, 
        'TEMPAFT2': 9999.0, 
        'TEMPMID1': 9999.0, 
        'TEMPMID2': 9999.0, 
        'TEMPFWD1': 9999.0, 
        'TEMPFWD2': 9999.0, 
        'TEMPTHRM': 9999.0, 
        'TEMP_CEB': 9999.0, 
        'ORIGIN': str_val, 
        'DETECTOR': str_val, 
        'IMGCTR': uint16_val, 
        'TIMGCTR': uint16_val, 
        'OBJECT': str_val, 
        'FILENAME': str_val, 
        'DATE': str_val, 
        'INSTRUME': 'SECCHI', 
        'OBSRVTRY': str_val, 
        'TELESCOP': 'STEREO',
        "DATE__OBS": " ",
        "TELESCOP": "STEREO",
        "WAVEFILE": str_val,   # name of waveform table file used by fsw
        "CCDSUM": flt_val,   # (sumrow + sumcol) / 2.0
        "IPSUM": flt_val,    # (sebxsum + sebysum) / 2.0
        "DATE-CMD": str_val,   # originally scheduled observation time
        "DATE-AVG": str_val,   # date of midpoint of the exposure(s) (UTC standard)
        "DATE-END": str_val,   # Date/time of end of (last) exposure
        "OBT_TIME": flt_val, # value of STEREO on-board-time since epoch ???
        "APID": int_val,       # application identifier / how downlinked
        "OBS_ID": int_val,     # observing sequence ID from planniing tool
        "OBSSETID": int_val,   # observing set (=campaign) ID from planning tool
        "IP_PROG0": int_val,   # description of onboard image processing sequence used
        "IP_PROG1": int_val,   # description of onboard image processing sequence used
        "IP_PROG2": int_val,   # description of onboard image processing sequence used
        "IP_PROG3": int_val,   # description of onboard image processing sequence used
        "IP_PROG4": int_val,   # description of onboard image processing sequence used
        "IP_PROG5": int_val,   # description of onboard image processing sequence used
        "IP_PROG6": int_val,   # description of onboard image processing sequence used
        "IP_PROG7": int_val,   # description of onboard image processing sequence used
        "IP_PROG8": int_val,   # description of onboard image processing sequence used
        "IP_PROG9": int_val,   # description of onboard image processing sequence used
        "IP_00_19": str_val,   # numeral char representation of values 0 - 19 in ip.Cmds
        "IMGSEQ": -1,      # number of image in current sequence (usually 0)
        "OBSERVER": str_val,   # Name of operator
        "BUNIT": str_val,      # unit of values in array
        "BLANK": int_val,      # value in array which means no data
        "FPS_CMD": str_val,    # T/F: from useFPS
        "VERSION": str_val,    # Identifier of FSW header version plus (EUVI only) pointing version
        "CEB_STAT": -1,    # CEB-Link-status (enum CAMERA_INTERFACE_STATUS)
        "CAM_STAT": -1,    # CCD-Interface-status (enum CAMERA_PROGRAM_STATE)
        "READPORT": str_val,   # CCD readout port
        "CMDOFFSE": flt_val, # lightTravelOffsetTime/1000.
        "RO_DELAY": -1.0,  # time (sec) between issuing ro command to the CEB and the start of the ro operation
        "LINE_CLR": -1.0,  # time (sec) per line for clear operation
        "LINE_RO": -1.0,   # time (sec) per line for readout operation
        "RAVG": -999.0,    # average error in star position (pixels)
        "BSCALE": 1.0,     # scale factor for FITS
        "BZERO": flt_val,    # value corresponding to zero in array for FITS
        "SCSTATUS": -1,    # spacecraft status message before exposure
        "SCANT_ON": str_val,   # T/F: derived from s/c status before and after
        "SCFP_ON": str_val,    # T/F: from actualSCFinePointMode
        "CADENCE": int_val,    # Number of seconds between exposures/sequences for the current observing program
        "CRITEVT": str_val,    # 0xHHHH (uppercase hex word)
        "EVENT": 'F',      # A flare IP event has (not) been triggered
        "EVCOUNT": str_val,    # count of number of times evtDetect has run ('0'..'127') ... remains a string
        "EVROW": int_val,      # X-coordinate of centroid of triggered event
        "EVCOL": int_val,      # Y-coordinate of centroid of triggered event
        "COSMICS": int_val,    # Number of pixels removed from image by cosmic ray removal algorithm in FSW
        "N_IMAGES": int_val,   # Number of CCD readouts used to compute the image
        "VCHANNEL": int_val,   # Virtual channel of telemetry downlink
        "OFFSETCR": flt_val, # Offset bias subtracted from image.
        "DOWNLINK": str_val,   # How the image came down
        "DATAMIN": -1.0,   # Minimum value of the image, including the bias derived
        "DATAMAX":-1.0,    # Maximum value of the image, including the bias derived
        "DATAZER": -1,     # Number of zero pixels in the image derived
        "DATASAT": -1,     # Number of saturated values in the image derived
        "DSATVAL": -1.0,   # Value used as saturated constant
        "DATAAVG": -1.0,   # Average value of the image derived
        "DATASIG": -1.0,   # Standard deviation in computing the average derived
        "DATAP01": -1.0,   # Intensity of 1st percentile of image derived
        "DATAP10": -1.0,   # Intensity of 10th percentile image derived
        "DATAP25": -1.0,   # Intensity of 25th percentile of image derived
        "DATAP50": -1.0,   # Intensity of 50th percentile of image derived (median)
        "DATAP75": -1.0,   # Intensity of 75th percentile of image derived
        "DATAP90": -1.0,   # Intensity of 90th percentile of image derived
        "DATAP95": -1.0,   # Intensity of 95th percentile of image derived
        "DATAP98": -1.0,   # Intensity of 98th percentile of image derived
        "DATAP99": -1.0,   # Intensity of 99th percentile of image derived
        "CALFAC": 0.0,     # Calibration factor applied, NOT including binning correction
        "CRPIX1": flt_val,   
        "CRPIX2": flt_val,
        "CRPIX1A": flt_val,  
        "CRPIX2A": flt_val,
        "RSUN": flt_val,     
        "CTYPE1": 'HPLN-TAN',
        "CTYPE2": 'HPLT-TAN',
        "CRVAL1": flt_val,
        "CRVAL2": flt_val,
        "CROTA": flt_val,    
        "PC1_1": 1.0,      
        "PC1_2": flt_val,    
        "PC2_1": flt_val,    
        "PC2_2": 1.0,      
        "CUNIT1": str_val,     # ARCSEC or DEG for HI
        "CUNIT2": str_val,     # ARCSEC or DEG for HI
        "CDELT1": flt_val,
        "CDELT2": flt_val,
        "PV2_1": flt_val,    # parameter for AZP projection (HI only)
        "PV2_1A": flt_val,   # parameter for AZP projection (HI only)
        "SC_ROLL": 9999.0, # values from get_stereo_hpc_point: (deg) - HI from scc_sunvec (GT)
        "SC_PITCH": 9999.0,# arcsec, HI deg
        "SC_YAW": 9999.0,  # arcsec, HI deg
        "SC_ROLLA": 9999.0,# RA/Dec values: (deg)
        "SC_PITA": 9999.0, # degrees
        "SC_YAWA": 9999.0, # degrees
        "INS_R0": 0.0,     # applied instrument offset in roll
        "INS_Y0": 0.0,     # applied instrument offset in pitch (Y-axis)
        "INS_X0": 0.0,     # applied instrument offset in yaw (X-axis) from
        "CTYPE1A": 'RA---TAN',
        "CTYPE2A": 'DEC--TAN',
        "CUNIT1A": 'deg',  # DEG
        "CUNIT2A": 'deg',  # DEG
        "CRVAL1A": flt_val,
        "CRVAL2A": flt_val,
        "PC1_1A": 1.0,     
        "PC1_2A": flt_val,   
        "PC2_1A": flt_val,   
        "PC2_2A": 1.0,     
        "CDELT1A": flt_val,
        "CDELT2A": flt_val,
        "CRLN_OBS": flt_val,
        "CRLT_OBS": flt_val,
        "XCEN": 9999.0,
        "YCEN": 9999.0,        
        "EPHEMFIL": str_val,   # ephemeris SPICE kernel
        "ATT_FILE": str_val,   # attitude SPICE kernel
        "DSUN_OBS": flt_val,
        "HCIX_OBS": flt_val,
        "HCIY_OBS": flt_val,
        "HCIZ_OBS": flt_val,
        "HAEX_OBS": flt_val,
        "HAEY_OBS": flt_val,
        "HAEZ_OBS": flt_val,
        "HEEX_OBS": flt_val,
        "HEEY_OBS": flt_val,
        "HEEZ_OBS": flt_val,
        "HEQX_OBS": flt_val,
        "HEQY_OBS": flt_val,
        "HEQZ_OBS": flt_val,
        "LONPOLE": 180,
        "HGLN_OBS": flt_val,
        "HGLT_OBS": flt_val,
        "EAR_TIME": flt_val,
        "SUN_TIME": flt_val,
        # "JITRSDEV": flt_val,   # std deviation of jitter from FPS or GT values
        # "FPSNUMS": 99999,    # Number of FPS samples
        # "FPSOFFY": 0,        # Y offset
        # "FPSOFFZ": 0,        # Z offset
        # "FPSGTSY": 0,        # FPS Y sum
        # "FPSGTSZ": 0,        # FPS Z sum
        # "FPSGTQY": 0,        # FPS Y square
        # "FPSGTQZ": 0,        # FPS Z square
        # "FPSERS1": 0,        # PZT Error sum [0]
        # "FPSERS2": 0,        # PZT Error sum [1]
        # "FPSERS3": 0,        # PZT Error sum [2]
        # "FPSERQ1": 0,        # PZT Error square [0]
        # "FPSERQ2": 0,        # PZT Error square [1]
        # "FPSERQ3": 0,        # PZT Error square [2]
        # "FPSDAS1": 0,        # PZT DAC sum [0]
        # "FPSDAS2": 0,        # PZT DAC sum [1]
        # "FPSDAS3": 0,        # PZT DAC sum [2]
        # "FPSDAQ1": 0,        # PZT DAC square [0]
        # "FPSDAQ2": 0,        # PZT DAC square [1]
        # "FPSDAQ3": 0         # PZT DAC square [2]
}
    
    for key in secchi_hdr.keys():
        if not key in hdr:
            hdr[key] = secchi_hdr[key]

    return hdr



def scc_img_trim(im, header, silent=True):
    """
    Conversion of scc_img_trim.pro for IDL. Returns rectified images with under/over scan areas removed.
    The program returns the imaging area of the CCD. If the image has not been rectified such that ecliptic north
    is up then the image is rectified.

    @param im: Selected image
    @param header: Header of .fits file
    @param silent: Suppress print statements
    @return: Rectified image with under-/overscan removed
    """
    info = "$Id: scc_img_trim.pro,v 2.4 2007/12/13 17:01:13 colaninn Exp $"
    histinfo = info[1:-1]

    if (header['DSTOP1'] < 1) or (header['DSTOP1'] > header['NAXIS1']) or (header['DSTOP2'] > header['NAXIS2']):
        precommcorrect(im, header, silent)

    x1 = header['DSTART1'] - 1
    x2 = header['DSTOP1'] - 1
    y1 = header['DSTART2'] - 1
    y2 = header['DSTOP2'] - 1

    img = im[y1:y2 + 1,x1:x2 + 1]

    s = np.shape(img)

    if (header['NAXIS1'] != s[0]) or (header['NAXIS2'] != s[1]):

        if not silent:
            print('Removing under- and overscan...')

        hdrsum = 2 ** (header['SUMMED'] - 1)

        header['R1COL'] = header['R1COL'] + (x1 * hdrsum)
        header['R2COL'] = header['R1COL'] + (s[0] * hdrsum) - 1
        header['R1ROW'] = header['R1ROW'] + (y1 * hdrsum)
        header['R2ROW'] = header['R1ROW'] + (s[1] * hdrsum) - 1

        header['DSTART1'] = 1
        header['DSTOP1'] = s[0]
        header['DSTART2'] = 1
        header['DSTOP2'] = s[1]

        header['NAXIS1'] = s[0]
        header['NAXIS2'] = s[1]

        header['CRPIX1'] = header['CRPIX1'] - x1
        header['CRPIX1A'] = header['CRPIX1A'] - x1

        header['CRPIX2'] = header['CRPIX2'] - y1
        header['CRPIX2A'] = header['CRPIX2A'] - y1

        wcoord = wcs.WCS(header)
        xycen = wcoord.wcs_pix2world((header['naxis1'] - 1.) / 2., (header['naxis2'] - 1.) / 2., 0)

        header['xcen'] = float(xycen[0])
        header['ycen'] = float(xycen[1])

        header['HISTORY'] = histinfo

    return img, header

def scc_get_missing(hdr, silent=True):
    """
    This function returns the index of the missing pixels.
    
    Args:
        hdr: Image header, either FITS or SECCHI structure.
        silent: Boolean flag to suppress output.
    
    Returns:
        missing: 1D array of longword vector containing the subscripts of the missing pixels.
    """

    # Convert MISSLIST to Superpixel 1D index

    base = 34
    misslist_str = hdr['MISSLIST']
    len_misslist = len(misslist_str)
    
    if len_misslist % 2 != 0:
        misslist_str = ' ' + misslist_str
        len_misslist += 1

    dex = np.arange(0, len_misslist, 2)
    misslist = np.asarray([int(misslist_str[i:i+2].strip(), base) for i in dex])
    n = len(misslist)

    if n != hdr['NMISSING']:
        if not silent:
            print('MISSLIST does not equal NMISSING')
        return np.array(0)
    
    if n ==0:
        return np.array(0)

    
    if hdr['COMPRSSN'] < 89:
        # Rice Compression and H-compress
        blksz = 64
        blklen = blksz ** 2
        missing = np.zeros(n * blklen, dtype=np.int64)

        ax1 = hdr['naxis1'] // blksz
        ax2 = hdr['naxis2'] // blksz
        blocks = np.vstack((misslist % ax1, misslist // ax2)).T

        dot = np.ones(blksz)
        plus = np.arange(blksz)

        x = np.outer(dot, plus)
        y = np.outer(plus, dot)

        for i in range(n):
            xx = x + blocks[i, 0] * blksz
            yy = y + blocks[i, 1] * blksz
            missing[i * blklen:(i + 1) * blklen] = yy.flatten() * hdr['naxis1'] + xx.flatten()

    elif hdr['COMPRSSN'] in [96, 97]:
        # 16 Segment ICER Compression
        ax1 = 4
        ax2 = 4
        blksz = hdr['naxis1'] // ax1

        blocks = np.vstack((misslist % ax1, misslist // ax2)).T

        if hdr['RECTIFY'] == True:
            if hdr['OBSRVTRY'] == 'STEREO_A':
                if hdr['DETECTOR'] == 'EUVI':
                    blocks = np.column_stack((ax1 - blocks[:, 1] - 1, ax1 - blocks[:, 0] - 1))
                elif hdr['DETECTOR'] == 'COR1':
                    blocks = np.column_stack((blocks[:, 1], ax1 - blocks[:, 0] - 1))
                elif hdr['DETECTOR'] == 'COR2':
                    blocks = np.column_stack((ax1 - blocks[:, 1] - 1, blocks[:, 0]))
                    
            elif hdr['OBSRVTRY'] == 'STEREO_B':
                if hdr['DETECTOR'] == 'EUVI':
                    blocks = np.column_stack((blocks[:, 1], ax1 - blocks[:, 0] - 1))
                elif hdr['DETECTOR'] == 'COR1':
                    blocks = np.column_stack((ax1 - blocks[:, 1] - 1, blocks[:, 0]))
                elif hdr['DETECTOR'] == 'COR2':
                    blocks = np.column_stack((blocks[:, 1], ax1 - blocks[:, 0] - 1))
                elif hdr['DETECTOR'] in ['HI1', 'HI2']:
                    blocks = np.column_stack((ax1 - blocks[:, 0] - 1, ax1 - blocks[:, 1] - 1))
       
        t = np.zeros((4, 4), dtype=int)
        t[blocks[:, 0], blocks[:, 1]] = 1

        buffer = np.zeros((4, 4, 4), dtype=int)

        buffer[0:3, :, 0] = np.where(t[0:3, :] - t[1:, :] < 0, 0, t[0:3, :] - t[1:, :])
        buffer[1:, :, 1] = np.where(t[1:, :] - t[0:3, :] < 0, 0, t[1:, :] - t[0:3, :])
        buffer[:, 0:3, 2] = np.where(t[:, 0:3] - t[:, 1:] < 0, 0, t[:, 0:3] - t[:, 1:])
        buffer[:, 1:, 3] = np.where(t[:, 1:] - t[:, 0:3] < 0, 0, t[:, 1:] - t[:, 0:3])

        buffer = buffer.reshape(16, 4)
        buffer = buffer[blocks[:, 1] * ax1 + blocks[:, 0], :]

        blklen = np.tile(blksz, n)
        blklen = (blklen + np.sum(buffer[:, 0:2], axis=1) * 20) * (blklen + np.sum(buffer[:, 2:4], axis=1) * 20)
        missing = np.zeros(np.sum(blklen), dtype=np.int64)

        dot = np.ones(blksz + 40)
        plus = np.arange(blksz + 40) - 20

        x = np.outer(dot, plus)
        y = np.outer(plus, dot)

        for i in range(n):
            xx = x.copy()
            yy = y.copy()
            if not buffer[i, 0]:
                xx = xx[0:blksz + 20, :]
                yy = yy[0:blksz + 20, :]
            if not buffer[i, 1]:
                xx = xx[20:, :]
                yy = yy[20:, :]
            if not buffer[i, 2]:
                xx = xx[:, 0:blksz + 20]
                yy = yy[:, 0:blksz + 20]
            if not buffer[i, 3]:
                xx = xx[:, 20:]
                yy = yy[:, 20:]

            xx = xx + blocks[i, 0] * blksz
            yy = yy + blocks[i, 1] * blksz

            missing_slice = slice(np.sum(blklen[:i+1])-blklen[i], np.sum(blklen[:i+1]))
            missing[missing_slice] = yy.flatten() * hdr['naxis1'] + xx.flatten()

    elif 90 <= hdr['COMPRSSN'] <= 95:
        # 32 Segment ICER Compression
        sg = np.zeros(n, dtype=int)

        blksz = np.array([[400, 416, 336, 352], [320, 320, 384, 384]]) // 2**(hdr['summed'] - 1)
        ax1 = [5, 6]
        ax2 = [4, 2]

        s = np.array([[0, -32, 0, -64], [0, 0, -256, -256]]) // 2**(hdr['summed'] - 1)

        # blocks = np.vstack((misslist, misslist)).T
        blocks = np.column_stack((misslist, misslist)).astype(int)

        bot = np.where(misslist <= 19)[0]
        top = np.where(misslist >= 20)[0]


        if top.size > 0:
            blocks[top, 0] = (blocks[top, 0] - 20) % ax1[1]
            blocks[top, 1] = ((blocks[top, 1] - 20) // ax1[1]) + ax2[0]

            three = np.where(blocks[top, 0] >= 4)[0]
            if three.size > 0:
                sg[top[three]] = 3
            two = np.where(blocks[top, 0] <= 3)[0]
            if two.size > 0:
                sg[top[two]] = 2

        if bot.size > 0:
            blocks[bot, 0] = blocks[bot, 0] % ax1[0]
            blocks[bot, 1] = blocks[bot, 1] // ax1[0]

            one = np.where(blocks[bot, 0] >= 2)[0]
            if one.size > 0:
                sg[bot[one]] = 1
            zero = np.where(blocks[bot, 0] <= 1)[0]
            if zero.size > 0:
                sg[bot[zero]] = 0


        t = np.zeros((6, 6), dtype=int)
        t[blocks[:, 0], blocks[:, 1]] = 1

        t = t.flatten()
        t[[5, 11, 17, 23]] = 2
        t = t.reshape((6,6))

       

        buffer = np.zeros((6, 6, 4), dtype=int)
        buffer[0:5, :, 0] = (t[0:5, :] - t[1:, :]) > 0
        buffer[1:, :, 1] = (t[1:, :] - t[0:5, :]) > 0


        buffer[:, 0:5, 2] = (t[:, 0:5] - t[:, 1:]) > 0
        buffer[:, 1:, 3] = (t[:, 1:] - t[:, 0:5]) > 0
        

        c = np.where(t.flatten() != 2)[0]


        # 2. Reform buffer
        buffer = buffer.reshape(36, 4)[c]
        buffer = buffer[misslist]




        # 3. Define length of each block
        # blklen=long64([[blksz[sg,0]],[blksz[sg,1]]])
        # blklen = (blklen[*,0]+total(buffer[*,0:1],2,/int)*20L)*$
        #      (blklen[*,1]+total(buffer[*,2:3],2,/int)*20L)
    


        blklen = np.array([blksz[0,sg], blksz[1,sg] ], dtype=np.int64)


        # blklen = (blklen[::2] + np.sum(buffer[:, 0:2], axis=1).astype(np.int64) * 20) * \
                # (blklen[1::2] + np.sum(buffer[:, 2:3], axis=1).astype(np.int64) * 20)
   


        blklen = (blklen[0,:] + np.sum(buffer[:, 0:2], axis=1) * 20) * \
                  (blklen[1,:] + np.sum(buffer[:, 2:], axis=1) * 20)
        

        missing = np.zeros((np.sum(blklen).astype(int), 2), dtype=np.int64)

        # 4. Math Cheats
        dot = np.ones(416 + 40)
        plus = np.arange(416 + 40) - 20

        # 5. Expanded Superpixel index
        x = np.outer(dot, plus)
        y = np.outer(plus, dot)


        start_idx = 0
        # 6. Loop over each Super-Superpixel
        n = len(sg)
        for i in range(n):
            sx, sy = blksz[0,sg[i]], blksz[1,sg[i]]
            xx = x[0:int(sx + 40), 0:int(sy + 40)]
            yy = y[0:int(sx + 40), 0:int(sy + 40)]

            if not buffer[i, 0]:
                xx = xx[0:int(sx + 20),:]
                yy = yy[0:int(sx + 20),:]
            if not buffer[i, 1]:
                xx = xx[20:,:]
                yy = yy[20:,:]
            if not buffer[i, 2]:
                xx = xx[:,0:int(sy + 20)]
                yy = yy[:,0:int(sy + 20)]
            if not buffer[i, 3]:
                xx = xx[:,20:]
                yy = yy[:,20:]

            xx = (xx + blocks[i, 0] * sx + s[0,sg[i]]).flatten()
            yy = (yy + blocks[i, 1] * sy + s[1,sg[i]]).flatten()


            end_idx = np.sum(blklen[:i+1])

            missing[start_idx:end_idx, 0] = xx
            missing[start_idx:end_idx, 1] = yy
            start_idx = end_idx

        # 7. Calculate the Rectified 2D index
        if hdr['RECTIFY'] == True:
            if hdr['OBSRVTRY'] == 'STEREO_A':
                if hdr['DETECTOR'] == 'EUVI':
                    missing = np.column_stack((hdr['NAXIS1'] - missing[:, 1] - 1, hdr['NAXIS1'] - missing[:, 0] - 1))
                elif hdr['DETECTOR'] == 'COR1':
                    missing = np.column_stack((missing[:, 1], hdr['NAXIS1'] - missing[:, 0] - 1))
                elif hdr['DETECTOR'] == 'COR2':
                    missing = np.column_stack((hdr['NAXIS1'] - missing[:, 1] - 1, missing[:, 0]))
                # HI1 and HI2 detectors do not require changes

            elif hdr['OBSRVTRY'] == 'STEREO_B':
                if hdr['DETECTOR'] == 'EUVI':
                    missing = np.column_stack((missing[:, 1], hdr['NAXIS1'] - missing[:, 0] - 1))
                elif hdr['DETECTOR'] == 'COR1':
                    missing = np.column_stack((hdr['NAXIS1'] - missing[:, 1] - 1, missing[:, 0]))
                elif hdr['DETECTOR'] == 'COR2':
                    missing = np.column_stack((missing[:, 1], hdr['NAXIS1'] - missing[:, 0] - 1))
                elif hdr['DETECTOR'] in ['HI1', 'HI2']:
                    missing = np.column_stack((hdr['NAXIS1'] - missing[:, 0] - 1, hdr['NAXIS1'] - missing[:, 1] - 1))

        # 8. Calculate final missing values
        missing = (missing[:, 1] * hdr['NAXIS1'] + missing[:, 0]).astype(np.int64)

        if hdr["comprssn"] > 98:
            if hdr["nmissing"] > 0:
                missing = np.arange(float(hdr['NAXIS1']) * hdr['NAXIS2']).astype(np.int64)
            else:
                missing = -1
        else:
            if not silent:
                print('ICER8 (8-segment) compression not accommodated; returning -1')
            missing = -1

    try:
        return np.asarray(missing)

    except UnboundLocalError:
        return np.array(0)
    


def get_smask(hdr, calpath, post_conj, silent=True):

    if hdr['DETECTOR'] == 'HI1':
        raise NotImplementedError('Not implemented for H1')
        return

    if hdr['DETECTOR'] == 'EUVI':
        filename = 'euvi_mask.fts'
    elif hdr['DETECTOR'] == 'COR1':
        filename = 'cor1_mask.fts'
    elif hdr['DETECTOR'] == 'COR2':
        if hdr['OBSRVTRY'] == 'STEREO_A':
            filename = 'cor2A_mask.fts'
        elif hdr['OBSRVTRY'] == 'STEREO_B':
            filename = 'cor2B_mask.fts'
    elif hdr['DETECTOR'] == 'HI2':
        if hdr['OBSRVTRY'] == 'STEREO_A':
            filename = 'hi2A_mask.fts'
        elif hdr['OBSRVTRY'] == 'STEREO_B':
            filename = 'hi2B_mask.fts'
    
    filename = calpath + filename

    try:
        hdul_smask = fits.open(filename)
        smask = hdul_smask[0].data

    except Exception:
        print('Error reading {}'.format(filename))
        sys.exit()


    xy = sccrorigin(hdr)

    fullm = np.zeros((2176, 2176), dtype=np.uint8)

    x1 = 2048 - np.shape(fullm[xy[0] - 1:, xy[1] - 1:])[0]
    y1 = 2048 - np.shape(fullm[xy[0] - 1:, xy[1] - 1:])[1]

    if x1 == 0: x1 = 2176
    if y1 == 0: y1 = 2176

    
    
    fullm[xy[1] - 1:y1, xy[0] - 1:x1] = smask

    

    date_header = datetime.strptime(hdr['DATE-OBS'], '%Y-%m-%dT%H:%M:%S.%f')
    date_1 = datetime.strptime("2015-05-19", '%Y-%m-%d')
    date_2 = datetime.strptime("2023-08-12", '%Y-%m-%d')
  
  
    if date_header>=date_1 and date_header<date_2 and hdr['DETECTOR'] != 'EUVI':
        fullm = np.rot90(fullm, 2)

    mask = rebin(fullm[int(hdr['R1ROW'])-1:int(hdr['R2ROW']),
                       int(hdr['R1COL'])-1:int(hdr['R2COL'])], (hdr['NAXIS1'], hdr['NAXIS2']))

   

    if(hdr["DETECTOR"]=="COR2" and hdr['OBSRVTRY'] == 'STEREO_A'):


        x_shape = hdr["NAXIS1"]
        y_shape = hdr["NAXIS2"]

        center_x = hdr["CRPIX1"]
        center_y = hdr["CRPIX2"]

        R_sun = 960. / hdr["CDELT1"]

        radius = 3.0

        y_array = [(j_step - center_y) / R_sun for j_step in range(y_shape)]
        outer_edge = np.max(y_array) * R_sun

        
        inner_radius = (radius * R_sun) ** 2
        outer_radius = outer_edge ** 2
        xx, yy = np.ogrid[0:x_shape, 0:y_shape]
        mask = np.ones((x_shape, y_shape), dtype=bool)
        mask[(xx - center_x) ** 2 + (yy - center_y) ** 2 < inner_radius] = 0
        mask[(xx - center_x) ** 2 + (yy - center_y) ** 2 > outer_radius] = 0

    

    return mask


def sccrorigin(hdr):

    if hdr['RECTIFY'] == True:
        
        if hdr['OBSRVTRY'] == 'STEREO_A':
            if hdr['detector'] == 'EUVI':
                r1col = 129
                r1row = 79
            elif hdr['detector'] == 'COR1':
                r1col = 1
                r1row = 79
            elif hdr['detector'] == 'COR2':
                r1col = 129
                r1row = 51
            elif hdr['detector'] == 'HI1':
                r1col = 51
                r1row = 1
            elif hdr['detector'] == 'HI2':
                r1col = 51
                r1row = 1

        elif hdr['OBSRVTRY'] == 'STEREO_B':
            if hdr['detector'] == 'EUVI':
                r1col = 1
                r1row = 79
            elif hdr['detector'] == 'COR1':
                r1col = 129
                r1row = 51
            elif hdr['detector'] == 'COR2':
                r1col = 1
                r1row = 79
            elif hdr['detector'] == 'HI1':
                r1col = 79
                r1row = 129
            elif hdr['detector'] == 'HI2':
                r1col = 79
                r1row = 129

        else:
            # LASCO/EIT
            r1col = 20
            r1row = 1
    else:
        r1col = 51
        r1row = 1

    return [r1col, r1row]


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


def scc_update_hdr(im, hdr0, silent=True):
    """
    This function returns updated header structure for level 1 processing.

    Parameters:
    im (np.ndarray): Calibrated image
    hdr0 (dict): Image header, SECCHI structure
    silent (bool, optional): Flag to suppress messages

    Returns:
    dict: Updated header
    """

    hdr = hdr0.copy()
    
    # Update structure
    hdr['BSCALE'] = 1.0
    hdr['BZERO'] = 0.0
    

    # Calculate Data Dependent Values

    stats = scc_img_stats(im)
    hdr['DATAMIN'] = stats['mn']
    hdr['DATAMAX'] = stats['mx']
    hdr['DATAZER'] = stats['zeros']
    hdr['DATAAVG'] = stats['men']
    hdr['DATASIG'] = stats['sig']
    hdr['DATAP01'] = stats['percentile'][0]
    hdr['DATAP10'] = stats['percentile'][1]
    hdr['DATAP25'] = stats['percentile'][2]
    hdr['DATAP50'] = stats['percentile'][3]
    hdr['DATAP75'] = stats['percentile'][4]
    hdr['DATAP90'] = stats['percentile'][5]
    hdr['DATAP95'] = stats['percentile'][6]
    hdr['DATAP98'] = stats['percentile'][7]
    hdr['DATAP99'] = stats['percentile'][8]
    
    date_mod = datetime.now().strftime('%Y-%m-%dT%H:%M:%S')
    hdr['date'] = date_mod
    
    return hdr

def scc_img_stats(img0):
    """
    This procedure generates image statistics for the header.

    Parameters:
    img0 (np.ndarray): Input image
    satmax (float, optional): Set saturation value of image; default is image maximum
    satmin (float, optional): Set minimum value of image; default is image minimum > 0
    verbose (bool, optional): Flag to print the statistics
    missing (np.ndarray, optional): Index of missing pixels where the statistics should not be calculated

    Returns:
    dict: A dictionary containing image statistics
    """
    
    img1 = img0.astype(float)
    img1[img1 == 0] = np.nan
    finite_mask = np.isfinite(img1)
    img = img1[finite_mask]
    zeros = np.sum(~finite_mask)
    
    # Calculate Minimum and Maximum
    mn = np.nanmin(img)
    mx = np.nanmax(img)
        
    # Calculate Standard Deviation and Mean
    sig = np.nanstd(img)
    men = np.nanmean(img)
    
    # Calculate Image Percentiles
    percentiles = [1, 10, 25, 50, 75, 90, 95, 98, 99]
    percentile = np.percentile(img, percentiles)
    
    return {
        'mn': mn,
        'mx': mx,
        'zeros': zeros,
        'men': men,
        'sig': sig,
        'percentile': percentile
    }

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


def azp2cart(vec, mu):
    """
    Conversion of azp2cart.pro for IDL. To convert points seen with an AZP projection with
    parameter, mu to a Cartesian frame. Note, this is a low level code, and would usually not be called directly.
    For details, see "Coordinate systems for solar image data", W.T. Thompson, A&A

    @param vec: Array of vector postions to transform
    @param mu: HI distortion parameter
    @return: An array of transformed vector positions
    """

    ## CHANGE Indexing changed here

    nstars = np.shape(vec)[1]
    vout = vec.copy()

    for i in range(nstars):

        rth = np.sqrt(vec[0, i] ** 2 + vec[1, i] ** 2)
        rho = rth / (mu + 1.0)
        cc = np.sqrt(1.0 + rho ** 2)
        th = np.arccos(1.0 / cc) + np.arcsin(mu * rho / cc)
        zz = np.cos(th)
        rr = np.sin(th)

        if rth < 1.0e-6:
            vout[0:2, i] = rr * vec[0:2, i]
        else:
            vout[0:2, i] = rr * vec[0:2, i] / rth

        vout[2, i] = zz

    return vout


def sc2cart(vec, roll_deg, pitch_deg, yaw_deg):
    """
    Conversion of sc2cart.pro for IDL. To convert spacecraft pointing to Cartesian points in a known reference frame.
    Note, this is a low level code, and would usually not be called directly. For the transformation we use 4x4 transformation
    matrices discussed in e.g. '3D computer graphics' by Alan Watt.

    @param vec: An array of vector postions to transform
    @param roll_deg: Spacecraft roll angle (in degrees)
    @param pitch_deg: Spacecraft pitch angle (in degrees)
    @param yaw_deg: Spacecraft yaw angle (in degrees)
    @return: An array of transformed vector positions
    """
    npts = len(vec[0, :])

    theta = (90 - pitch_deg) * np.pi / 180.
    phi = yaw_deg * np.pi / 180.
    roll = roll_deg * np.pi / 180.

    normx = np.sin(theta) * np.cos(phi)
    normy = np.sin(theta) * np.sin(phi)
    normz = np.cos(theta)

    vdx = 0.
    vdy = 0.
    vdz = 1.

    vd_norm = vdx * normx + vdy * normy + vdz * normz

    vxtmp = vdx - vd_norm * normx
    vytmp = vdy - vd_norm * normy
    vztmp = vdz - vd_norm * normz

    ndiv = np.sqrt(vxtmp ** 2 + vytmp ** 2 + vztmp ** 2)
    vx = vxtmp / ndiv
    vy = vytmp / ndiv
    vz = vztmp / ndiv

    ux = normy * vz - normz * vy
    uy = normz * vx - normx * vz
    uz = normx * vy - normy * vx

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


def get_calimg(header, calpath, post_conj, silent=True):
    """
    Conversion of get_calimg.pro for IDL. Returns calibration correction array. Checks common block before opening
    calibration file. Saves calibration file to common block. Trims calibration array for under/over scan.
    Re-scales calibration array for summing.

    @param header: Header of .fits file
    @param calpath: Path to calibration files
    @param post_conj: Indicates whether spacecraft is pre or post conjecture
    @param silent: Run on silent mode (True or False)
    @return: Array to correct for calibration
    """

    sumflg = 0
    if header['DETECTOR'] == 'HI1':

        if header['summed'] == 1:
            cal_version = '20061129_flatfld_raw_h1' + header['OBSRVTRY'][7].lower() + '.fts'
            sumflg = 0
        else:
            cal_version = '20100421_flatfld_sum_h1' + header['OBSRVTRY'][7].lower() + '.fts'
            sumflg = 1

    elif header['DETECTOR'] == 'HI2':

        if header['summed'] == 1:
            cal_version = '20150701_flatfld_raw_h2' + header['OBSRVTRY'][7].lower() + '.fts'
            sumflg = 0
        else:
            cal_version = '20150701_flatfld_sum_h2' + header['OBSRVTRY'][7].lower() + '.fts'
            sumflg = 1

    elif header['DETECTOR'] == 'COR2':
        if header['OBSRVTRY'] == 'STEREO_A':
            cal_version = '20060929_vignet'
        else:
            cal_version = '20140723_vignet'

        tail = '_vCc2'+header["OBSRVTRY"][7]+'.fts'
        cal_version = cal_version + tail
    else:
        ## TODO Implement get_calimg for other detectors
        print('get_calimg not implemented for detectors other than HI-1, HI-2, COR2')
        exit()


    calpath = calpath + cal_version

    try:
        hdul_cal = fits.open(calpath)

    except FileNotFoundError:
        print(f'Calibration file {calpath} not found')
        sys.exit()
    
    # if header['NAXIS1'] < 1024:
    #     print('get_calimg does not work with beacon data.')
    #     return np.ones((1,1)),1

    try:
        p1col = hdul_cal[0].header['P1COL']
    except KeyError:
        hdul_cal[0].header = fix_secchi_hdr(hdul_cal[0].header)
        p1col = hdul_cal[0].header['P1COL']

    if (p1col <= 1):
        if sumflg:
            x1 = 25
            x2 = 1048
            y1 = 0
            y2 = 1023
        else:
            x1 = 50
            x2 = 2047 + 50
            y1 = 0
            y2 = 2047

        cal = hdul_cal[0].data[y1:y2 + 1, x1:x2 + 1]

    else:
        cal = hdul_cal[0].data

    
    if (header['RECTIFY'] == True) and (hdul_cal[0].header['RECTIFY'] == 'F'):
        cal, _ = secchi_rectify(cal.copy(), hdul_cal[0].header, silent=True)

        if not silent:
            print('Rectified calibration image')

    if sumflg:
        if header['summed'] <= 2:
            hdr_sum = 1

        else:
            hdr_sum = 2 ** (header['summed'] - 2)

    else:
        hdr_sum = 2 ** (header['summed'] - 1)

    s = np.shape(cal)

    cal = rebin(cal, (int(s[1] / hdr_sum), int(s[0] / hdr_sum)))

    if post_conj:
        cal = np.rot90(cal, k=2)

    hdul_cal.close()
  
    return cal, cal_version

def scc_sebip(data, header, silent=True):
    """Direct conversion of scc_sebip.pro for IDL.
    Determines what has happened in terms of on-board sebip binning and corrects it.
    Takes image data and header as input and returns fixed image.
    @param data: Data of .fits file
    @param header: Header of .fits file
    @param silent: Run in silent mode
    @return: Data corrected for on-board binning"""

    # IP_00_19= '41128 31114 37113121  7 41 37120129  7 40 87 50  3 53  1100'
    # IP_00_19= '41128 31 34 37113121  7 41 37120129  7  0  0  0  0  0  0  0' /


    ip_raw = header['IP_00_19']

    while len(ip_raw) < 60:
        ip_raw = ' ' + ip_raw

        header['IP_00_19'] = ip_raw

    ip_bytes = bytearray(ip_raw, encoding='ascii')
    ip_arr = np.array(ip_bytes)
    ip_reform = ip_arr.reshape(-1, 3).transpose()

    ip_temp = []
    ip = []

    for i in range(ip_reform.shape[1]):
        ip_temp.append(ip_reform[:, i].tostring())

    for i in range(len(ip_temp)):
        ip.append(ip_temp[i].decode('ascii'))

    cnt = ip.count('117')

    if cnt == 1:
        ind = np.where(ip == '117')
        ip = ip[3 * ind:]
        while len(ip) < 60:
            ip = ip.append('  0')
            header['IP_00_19'] = ''.join(ip)

    ## CHANGE updated header, added count16 + count17 to match IDL behaviour (was xor before)
    

    cnt1   = ip.count('  1')
    cnt2   = ip.count('  2')
    cntspw = ip.count(' 16') + ip.count(' 17')
    cnt50  = ip.count(' 50')
    cnt53  = ip.count(' 53')
    cnt82  = ip.count(' 82')
    cnt83  = ip.count(' 83')
    cnt84  = ip.count(' 84')
    cnt85  = ip.count(' 85')
    cnt86  = ip.count(' 86')
    cnt87  = ip.count(' 87')
    cnt88  = ip.count(' 88')
    cnt118 = ip.count('118')
    # print(cnt1,cnt2,cntspw,cnt50,cnt53,cnt82,cnt83,cnt84,cnt85,cnt86,cnt87,cnt88,cnt118)



    if header['DIV2CORR']:
        cnt1 = cnt1 - 1

    if cnt1 < 0:
        cnt1 = 0

    if cnt1 > 0:
        data = data * (2.0 ** cnt1)
        if not silent:
            print('Corrected for divide by 2 x {}'.format(cnt1))

        header['HISTORY'] = 'seb_ip Corrected for divide by 2 x {}'.format(cnt1)

    if cnt2 > 0:
        ## CHANGE to square instead of multiply
        data = data ** (2.0 ** cnt2)
        if not silent:
            print('Corrected for square root x {}'.format(cnt2))
        
        header['HISTORY'] = 'seb_ip Corrected for square root x {}'.format(cnt2)

    if cntspw > 0:
        data = data * (64.0 ** cntspw)
        if not silent:
            print('Corrected for HI SPW divide by 64 x {}'.format(cntspw))
        
        header['HISTORY'] = 'seb_ip Corrected for HI SPW divide by 64 x {}'.format(cntspw)

    if cnt50 > 0:
        data = data * (4.0 ** cnt50)
        if not silent:
            print('Corrected for divide by 4 x {}'.format(cnt50))
        
        header['HISTORY'] = 'seb_ip Corrected for divide by 4 x {}'.format(cnt50)

    if cnt53 > 0 and header['ipsum'] > 0:
        data = data * (4.0 ** cnt53)
        if not silent:
            print('Corrected for divide by 4 x {}'.format(cnt53))
        
        header['HISTORY'] = 'seb_ip Corrected for divide by 4 x {}'.format(cnt53)

    if cnt82 > 0:
        data = data * (2.0 ** cnt82)
        if not silent:
            print('Corrected for divide by 2 x {}'.format(cnt82))
        
        header['HISTORY'] = 'seb_ip Corrected for divide by 2 x {}'.format(cnt82)

    if cnt83 > 0:
        data = data * (4.0 ** cnt83)
        if not silent:
            print('Corrected for divide by 4 x {}'.format(cnt83))
        
        header['HISTORY'] = 'seb_ip Corrected for divide by 4 x {}'.format(cnt83)

    if cnt84 > 0:
        data = data * (8.0 ** cnt84)
        if not silent:
            print('Corrected for divide by 8 x {}'.format(cnt84))
        
        header['HISTORY'] = 'seb_ip Corrected for divide by 8 x {}'.format(cnt84)

    if cnt85 > 0:
        data = data * (16.0 ** cnt85)
        if not silent:
            print('Corrected for divide by 16 x {}'.format(cnt85))
        
        header['HISTORY'] = 'seb_ip Corrected for divide by 16 x {}'.format(cnt85)

    if cnt86 > 0:
        data = data * (32.0 ** cnt86)
        if not silent:
            print('Corrected for divide by 32 x {}'.format(cnt86))
        
        header['HISTORY'] = 'seb_ip Corrected for divide by 32 x {}'.format(cnt86)

    if cnt87 > 0:
        data = data * (64.0 ** cnt87)
        if not silent:
            print('Corrected for divide by 64 x {}'.format(cnt87))
        
        header['HISTORY'] = 'seb_ip Corrected for divide by 64 x {}'.format(cnt87)

    if cnt88 > 0:
        data = data * (128.0 ** cnt88)
        if not silent:
            print('Corrected for divide by 128 x {}'.format(cnt88))
        
        header['HISTORY'] = 'seb_ip Corrected for divide by 128 x {}'.format(cnt88)

    if cnt118 > 0:
        data = data * (3.0 ** cnt118)
        if not silent:
            print('Corrected for divide by 3 x {}'.format(cnt118))
        
        header['HISTORY'] = 'seb_ip Corrected for divide by 3 x {}'.format(cnt118)

    if not silent:
        print('------------------------------------------------------')

    return data, header

def get_biasmean(header, silent=True):
    """
    Conversion of get_biasmean.pro for IDL. Returns mean bias for a give image.

    @param header: Header of .fits file
    @return: Bias to be subtracted from the image
    """
    bias = header['BIASMEAN']
    ipsum = header['IPSUM']

    if ('103' in header['IP_00_19']) or (' 37' in header['IP_00_19']) or (' 38' in header['IP_00_19']):

        if not silent:
            print('Biasmean subtracted onboard in seb ip.')
            
        bias = 0
        return bias
    
    if header['DETECTOR'][0:2] == 'HI':
        bias = bias-(bias/header['N_IMAGES'])

    if header['OFFSETCR'] > 0:
        bias = 0
        return bias

    if ipsum > 1:
        bias = bias*((2**(ipsum-1))**2)

    return bias

def sc_inverse(n, diag, below, above):

    wt_above = float(above) / diag
    wt_below = float(below) / diag

    wt_above_1 = wt_above - 1
    wt_below_1 = wt_below - 1
    power_above = np.zeros(n-1, dtype=float)
    power_below = np.zeros(n-1, dtype=float)

    power_above[0] = 1
    power_below[0] = 1

    for row in range(1, n-1):
        power_above[row] = power_above[row-1] * wt_above_1
        power_below[row] = power_below[row-1] * wt_below_1
    

    v = np.concatenate(([0], wt_below * (power_below * power_above[::-1])))
    u = np.concatenate(([0], wt_above * (power_above * power_below[::-1])))
    

    d = -u[1] / wt_above - (np.sum(v) - v[-1])
    f = 1 / (diag * (d + wt_above * np.sum(v)))

    u[0] = d
    v[0] = d
    u = u * f
    v = v[::-1] * f

    p = np.zeros((n, n), dtype=np.float64)
   
    

    p[0, 0] = u[0]

    for row in range(1, n-1):
        p[row, 0:row] = v[n-row-1:n-1]
        p[row, row:] = u[:n-row]

    p[-1,:] = v

   

   
    return p

def load_and_remove_background(dates,path,spc):
    # dates_one,"/Volumes/Data_drive/data_maike/L1/SA/cor2/science/",time_215=datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S')
    background_date = datetime.strptime(dates[0], '%Y%m%d')-timedelta(hours=1)

    backround_file   = "/Volumes/Data_drive/data/backgrounds/cor2/20080102/dc2"+spc+"_pTBr_"+background_date.strftime('%y%m%d')+".fts"
    print(backround_file)
    try:
        background = fits.open(backround_file)
    except:
        print("couldnt load background")
        return 
    background_data = background[0].data
    background_header = background[0].header

    print(background_header["EXPTIME"])

    datas = []
    dates_jplot = []
    for d in dates:
        loadpath = path + d
        files = natsorted(glob.glob(loadpath+"/*"))
        prev = None
        for i in range(0,len(files)):
            hdul = fits.open(files[i])
            header = hdul[0].header
            if header["EXPTIME"]!=-1:
                data = hdul[0].data

                calimg,_ = get_calimg(hdul[0].header,'SolarSoftWare/stereo/secchi/calibration/',False)
                calfac,_ = get_calfac(hdul[0].header)


                background_data = resize(background_data,(data.shape[0],data.shape[1]),preserve_range=True)
                data = np.nan_to_num(data,np.nanmedian(data),np.nanmedian(data))
                data_l2 = (data - background_data)  #* calimg * calfac 


                if not prev is None :
  
                    
                    mask = get_smask(header,'SolarSoftWare/stereo/secchi/calibration/',False)
           

                    data3 = data_l2 - prev 
                    data3 = data3 * mask

                    vmin = np.median(data3)-np.std(data3)
                    vmax = np.median(data3)+np.std(data3)

                    data3[data3>vmax] = vmax 
                    data3[data3<vmin] = vmin 
                    data3 = (data3-vmin)/(vmax-vmin)
                    data3 = resize(data3,(512,512),preserve_range=True)
                    
                    # print(files[i])
                    # fig,ax = plt.subplots(1,3)
                    # ax[0].imshow(data3,cmap="gray")
                    # ax[1].imshow(background_data,vmin= np.median(background_data)-np.std(background_data),vmax= np.median(background_data)+np.std(background_data),cmap="gray")
                    # ax[2].imshow(data_l2,vmin= np.median(data_l2)-np.std(data_l2),vmax= np.median(data_l2)+np.std(data_l2),cmap="gray")
                    # plt.show()

                    # datas.append(data3[1024-64:1024+64,:])
                    # dates_jplot.append(datetime.strptime(header["DATE-OBS"],'%Y-%m-%dT%H:%M:%S.%f'))
                    img = Image.fromarray((data3*255.0).astype(np.uint8))
                    img.save("/Volumes/Data_drive/data/pngs/S"+spc+"/"+files[i].split("/")[-1][:-4]+".png")

                prev = data_l2 