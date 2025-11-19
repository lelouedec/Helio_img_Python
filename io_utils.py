import os 
from multiprocessing.pool import Pool
from multiprocessing import cpu_count
from itertools import repeat
from requests.adapters import HTTPAdapter, Retry
import psutil
from urllib3.exceptions import InsecureRequestWarning
from urllib3 import disable_warnings
import requests
from bs4 import BeautifulSoup
import traceback
import logging
import sys 
import tqdm

disable_warnings(InsecureRequestWarning)


scs = {
    'SA': 'ahead',
    'SB': 'behind',
    'SO': 'something'
}

def multi_process_dl(num_cpus,url,ext,path_dir):
    pool = Pool(int(num_cpus/2), limit_cpu)

    urls = listfd(url, ext)
    inputs = zip(repeat(path_dir), urls)


    try:
        results = pool.starmap(fetch_url, inputs, chunksize=2)

    except ValueError:
        print("There was an error multiprocessing lol ")
            
    pool.close()
    pool.join()

def download_files(datelist, save_path, ftpsc, ins, bflag, silent=True):
    """
    Downloads images from pub directory

    @param datelist: List of dates for which to download files
    @param save_path: Path for saving downloaded files
    @param ftpsc: SA,SB,SO
    @param ins: hi_1, hi_2, cor2, C3
    @param bflag: Data type (science/beacon) for which to download files
    @param silent: Run in silent mode
    """
    sc = scs[ftpsc] 



    if not silent:
        print('Fetching files...')

    for d in tqdm.tqdm(range(0,len(datelist))):
        date = datelist[d]
        if bflag == 'beacon':

            url = 'https://stereo-ssc.nascom.nasa.gov/pub/beacon/' + sc + '/secchi/img/' + ins + '/' + str(date)
            path_dir = save_path + '/' + str(date)

            if ins == 'hi_1':
                if sc == 'ahead':
                    ext = 's7h1A.fts'
                if sc == 'behind':
                    ext = 's7h1B.fts'

            elif ins == 'hi_2':
                if sc == 'ahead':
                    ext = 's7h2A.fts'
                if sc == 'behind':
                    ext = 's7h2B.fts'

            elif ins == 'cor2':
                print("COR2 beacon not implemented")
                exit()

        else:

            url = 'https://stereo-ssc.nascom.nasa.gov/pub/ins_data/secchi/L0/' + sc[0] + '/img/' + ins + '/' + str(date)
            path_dir = save_path + '/' + str(date)

            if ins == 'hi_1':
                if sc == 'ahead':
                    ext = 's4h1A.fts'
                if sc == 'behind':
                    ext = 's4h1B.fts'

            elif ins == 'hi_2':
                if sc == 'ahead':
                    ext = 's4h2A.fts'
                if sc == 'behind':
                    ext = 's4h2B.fts'

            elif ins == 'cor2':
                url = 'https://stereo-ssc.nascom.nasa.gov/pub/ins_data/secchi/L0/'+ sc[0] +'/seq/cor2/' + str(date)
                url2 = 'https://stereo-ssc.nascom.nasa.gov/pub/ins_data/secchi/L0/'+ sc[0] +'/img/cor2/'+ str(date)
                if sc == 'ahead':
                    ext = 'n4c2A.fts'
                    ext2 = 'd4c2A.fts'
                if sc == 'behind':
                    ext = 'n4c2B.fts'
                    ext2 = 'd4c2B.fts'
            
            elif ins == 'C3':
                date_lasco = date[2:]
                url  = "https://lasco-www.nrl.navy.mil/lz/level_05/"+str(date_lasco)+"/c3/"
                ext  = '.fts'

            elif ins == 'C2':
                date_lasco = date[2:]
                url  = "https://lasco-www.nrl.navy.mil/lz/level_05/"+str(date_lasco)+"/c2/"
                ext  = '.fts'
        
        if not os.path.exists(path_dir):
            os.makedirs(path_dir)
        
        num_cpus = cpu_count()
        multi_process_dl(num_cpus,url,ext,path_dir)

        if(ins =='cor2'):
            multi_process_dl(num_cpus,url2,ext2,path_dir)

      
def limit_cpu():
    """
    Is called when starting a new multiprocessing pool. Decreases priority of processes to limit total CPU usage. 
    """
    p = psutil.Process(os.getpid())
    # set to lowest priority
    p.nice(19)

#######################################################################################################################################


def listfd(input_url, extension):
    """
    Provides list of urls and corresponding file names to download.

    @param input_url: URL of STEREO-HI image files
    @param extension: File ending of STEREO-HI image files
    @return: List of URLs and corresponding filenames to be downloaded
    """

    disable_warnings(InsecureRequestWarning)

    
    session = requests.Session()
    retry = Retry(connect=3, backoff_factor=0.5)
    adapter = HTTPAdapter(max_retries=retry)

    session.mount('http://', adapter)
    session.mount('https://', adapter)

    output_urls = []

    page = session.get(input_url).text
    #page = requests.get(input_url, verify=False).text

    soup = BeautifulSoup(page, 'html.parser')
    url_found = [input_url + '/' + node.get('href') for node in soup.find_all('a') if node.get('href').endswith(extension)]
    filename = [node.get('href') for node in soup.find_all('a') if node.get('href').endswith(extension)]

    for i in range(len(filename)):
        output_urls.append((filename[i], url_found[i]))

    return output_urls


#######################################################################################################################################

def fetch_url(path, entry):
    """
    Downloads URLs specified by listfd.

    @param path: Path where downloaded files are to be saved
    @param entry: Combination of filename and URL of downloaded file
    """
    filename, uri = entry

    if not os.path.exists(path + '/' + filename):
        r = requests.get(uri, allow_redirects=True)
        open(path + '/' + filename, 'wb').write(r.content)
#######################################################################################################################################

def check_calfiles(path):
    """
    Checks if SSW IDL HI calibration files are present - creates appropriate directory and downloads them if not.

    @param path: Path in which calibration files are located/should be located
    """
    url_cal = "https://soho.nascom.nasa.gov/solarsoft/stereo/secchi/calibration/"
    print('Checking calibration files...')

    if not os.path.exists(path + 'calibration/'):
        os.makedirs(path + 'calibration/')

    try:
        uri = listfd(url_cal, '.fts')
    
        for entry in uri:
            if not os.path.isfile(path + 'calibration/' + entry[0]):
                fetch_url(path + 'calibration', entry)
            else:
                pass
        
        return
        
    except KeyboardInterrupt:
        return
    
    except Exception as e:
        logging.error(traceback.format_exc())
        sys.exit()

#######################################################################################################################################

def check_pointfiles(path):
    """
    Checks if SSW IDL HI calibration files are present - creates appropriate directory and downloads them if not.

    @param path: Path in which calibration files are located/should be located
    """
    url_point = "https://soho.nascom.nasa.gov/solarsoft/stereo/secchi/data/hi/"

    
    print('Checking pointing files...')

    if not os.path.exists(path + 'data/hi/'):
        os.makedirs(path + 'data/hi/')

    try:
        uri = listfd(url_point, '.fts')
          
        for entry in uri:
            if not os.path.isfile(path + 'data/hi/' + entry[0]):
                fetch_url(path + 'data/hi', entry)
            else:
                pass
        return
            
    except KeyboardInterrupt:
        return
       
    except Exception as e:
        logging.error(traceback.format_exc())
        sys.exit()