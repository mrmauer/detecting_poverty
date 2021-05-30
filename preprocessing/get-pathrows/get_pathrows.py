import io
import urllib.request
import zipfile
import os

def download_pathrows():
    '''
    Check whether the shapefile containg all WRS path-rows is in the file system.
    If not, download and unzip the files from paladium.
    '''
    # os checks
    if not os.path.isdir("landsat-path-row"):
        print("Downloading landsat-path-row shapes.")
        url = "https://prd-wret.s3-us-west-2.amazonaws.com/assets/palladium/"
        url += "production/s3fs-public/atoms/files/WRS2_descending_0.zip"
        r = urllib.request.urlopen(url)
        print('Unzipping file.')
        zip_file = zipfile.ZipFile(io.BytesIO(r.read()))
        zip_file.extractall("landsat-path-row")
        zip_file.close()
        print('Successful download and unzipping.')
        os.listdir('landsat-path-row')
    else:
        print("Directory landsat-path-row of shape fiels already exists.")

    return

if __name__ == "__main__":
    download_pathrows()
    
