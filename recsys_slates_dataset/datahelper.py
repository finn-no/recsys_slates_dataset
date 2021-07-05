# AUTOGENERATED! DO NOT EDIT! File to edit: datahelper.ipynb (unless otherwise specified).

__all__ = ['download_data_files']

# Cell
import logging
from google_drive_downloader import GoogleDriveDownloader as gdd
def download_data_files(data_dir : str = "data", overwrite=False, progbar=True, use_int32=True):
    """
    Downloads the data from google drive.

    - data_dir: relative path to where data is downloaded.
    - overwrite: If files exist they will not be downloaded again. NB/todo: the function does not check if there is a complete download of the file.
    - progbar: simple progbar that says how much data that has been downloaded for each file.
    - use_int32: The interaction data is a very large file and is not possible to load into memory in some cases (e.g. google colab). Therefore, we recommend using the int32 data type when loading the data.
    """

    if use_int32:
        data_fileid = '1XHqyk01qi9qnvBTfWWwqgDzrdjv1eBVV'
    else:
        data_fileid = '1VXKXIvPCJ7z4BCa4G_5-Q2XMAD7nXOc7'

    gdrive_file_ids = {
        'data.npz' : data_fileid,
        'ind2val.json' : '1WOCKfuttMacCb84yQYcRjxjEtgPp6F4N',
        'itemattr.npz' : '1rKKyMQZqWp8vQ-Pl1SeHrQxzc5dXldnR'
    }

    for filename, gdrive_id in gdrive_file_ids.items():
        logging.info("Downloading {}".format(filename))
        gdd.download_file_from_google_drive(file_id=gdrive_id,
                                        dest_path="{}/{}".format(data_dir, filename),
                                        overwrite=overwrite, showsize=progbar)
    logging.info("Done downloading all files.")
    return True
