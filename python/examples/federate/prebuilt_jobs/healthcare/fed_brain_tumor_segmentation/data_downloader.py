import os
import logging
import subprocess
import sys
from pathlib import PurePath

USERNAME = "m.abdelmaged@nu.edu.eg"
PASSWORD = "4444samsung."
SHAREPOINT_SITE = "https://nileuniversity.sharepoint.com/sites/FederatedLearningImageSegmentation"
SHAREPOINT_SITE_NAME = "FederatedLearningImageSegmentation"
SHAREPOINT_DOC = "Shared Documents"

# HELPING FUNCTION 1
def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# HELPING FUNCTION 2
def save_file(file_n, file_obj):
    file_dir_path = PurePath(".", file_n)
    with open(file_dir_path, 'wb') as f:
        f.write(file_obj)

# MAIN FUNTION 
def download_data(*,file_name, folder_name, output_folder, remove_zip = False):
    install("office365-rest-client")
    from office365.sharepoint.client_context import ClientContext
    from office365.runtime.auth.user_credential import UserCredential
    from office365.sharepoint.files.file import File
    logging.info("downloader - package installed")
    
    conn = ClientContext(SHAREPOINT_SITE).with_credentials(
        UserCredential(
            USERNAME,
            PASSWORD
        )
    )
    file_url = f'/sites/{SHAREPOINT_SITE_NAME}/{SHAREPOINT_DOC}/{folder_name}/{file_name}'
    file = File.open_binary(conn, file_url)
    file_obj = file.content
    save_file(file_name, file_obj)
    logging.info("downloader - data downloaded")
    
    install("patool")
    import patoolib  
    patoolib.extract_archive(file_name, outdir=output_folder)
    logging.info("downloader - data unzipped")
    if remove_zip:
      zip_path = os.path.join(os.getcwd(),file_name)
      os.remove(zip_path)

# FUNCTION CALL
# download_data(file_name = "client 3.zip",
#               folder_name = "Clients",
#               output_folder = "client 3",
#               remove_zip = True
#               )