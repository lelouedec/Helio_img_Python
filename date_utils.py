from datetime import datetime,timedelta
import numpy as np




def create_range_from_dates(date_start,date_end,background_length):
    """
    Create list of dates from two strings of date start and date end in the format YYYYMMDD
    background_length is the number of days before the date_start to use as background
    """

    date = datetime.strptime(date_start, '%Y%m%d')
    date_end = datetime.strptime(date_end, '%Y%m%d')

    date_red = datetime.strptime(date_start, '%Y%m%d') - timedelta(days=background_length) 
    
    datelist = np.arange(date, date_end + timedelta(days=1), timedelta(days=1)).astype(datetime)
    datelist = [dat.strftime('%Y%m%d') for dat in datelist]

    datelist_red = np.arange(date_red, date_end + timedelta(days=1), timedelta(days=1)).astype(datetime)
    datelist_red = [dat.strftime('%Y%m%d') for dat in datelist_red]

    return datelist,datelist_red