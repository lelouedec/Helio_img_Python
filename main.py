
import sys
import yaml
import io_utils
import constants
import date_utils
from  secchi import prep as secchi_prep
from lasco import prep as lasco_prep
import warnings 

warnings.filterwarnings("ignore")

if __name__ == '__main__':

    if(sys.argv[1]!=""):
        config_path = sys.argv[1]
    else:
        print("please provide path to config file for this run")

    with open(config_path) as stream:
        try:
            content = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    # content = 
    # 'output_directory': 
    # 'data_directory':
    # 'solarsoft_directory': 
    # 'runs':
    #     .....

    for r in content["runs"]:
        dates,dates_incbackground = date_utils.create_range_from_dates(r["start_date"],r["end_date"],r["background_length"]) 
        match r["task"]:
            case 'download':
                print("Downloading files for ",constants.spacecrafts[r["spacecraft"]],", instrument: ",r["instrument"],", mode: ",r["data_type"],", ",r["start_date"],"---> ",r["end_date"])
                io_utils.download_files(dates_incbackground,
                                        content["data_directory"]+"L0/"+constants.spacecrafts[r["spacecraft"]]+"/"+r["instrument"]+"/"+r["data_type"]+"/",r["spacecraft"],
                                        r["instrument"],
                                        r["data_type"])
            case 'reduction':
                if(r["spacecraft"]=="SA" or r["spacecraft"]=="SB"):
                    print("Using SECCHI prep functions")
                    secchi_prep.data_reduction(dates_incbackground,  
                                          content["data_directory"]+"L1/"+constants.spacecrafts[r["spacecraft"]]+"/"+r["instrument"]+"/"+r["data_type"]+"/", 
                                          content["solarsoft_directory"], 
                                          r["spacecraft"], 
                                          r["instrument"], 
                                          r["data_type"]
                                          )
                else:
                    print("Using LASCO prep functions")
                    lasco_prep.lasco_prep(dates_incbackground,
                                          content["data_directory"]+"L1/"+constants.spacecrafts[r["spacecraft"]]+"/"+r["instrument"]+"/"+r["data_type"]+"/",
                                          content["solarsoft_directory"],
                                          r["instrument"]
                                          )
                    
            case 'difference':
                if(r["spacecraft"]=="SA" or r["spacecraft"]=="SB"):
                    print("Rdifs SECCHI prep function")
                    secchi_prep.create_running_differences(
                        dates,
                        content["data_directory"]+"Rdif/"+constants.spacecrafts[r["spacecraft"]]+"/"+r["instrument"]+"/"+r["data_type"]+"/",
                        r["spacecraft"],
                        r["instrument"],
                        r["data_type"]

                    )
            case default:
                print("something")

   