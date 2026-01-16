# SSW_python

## SECCHI 

### HI
Refactor of the code from https://github.com/maikebauer/STEREO-HI-Data-Processing for SECCHI HI-1 and HI-2 (most is ported just missing running differences and jplot creation, coming soon)

### COR2

Most of COR2 reduction is working and implemented and working, produces nice images, calfac and calimg have to be done separately to use provide monthly min backgrounds


## LASCO

### C3
Working for most examples, not 100% tested

### C2
Working nearly fully, does not handle polarized sequences


Will add soon how to use code, one can look at config.yaml for an idea and use: python main.py config.yaml to execute reduction and downloading. (Might have to download pointing files and calibration for HI manually for now).

