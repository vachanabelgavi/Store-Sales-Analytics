import boto3
import botocore
import traceback
import zipfile
import os

def download_from_s3(BUCKET_NAME, PATH_TO_FILE, LOCAL_PATH):
    s3 = boto3.resource('s3')
    try:
        print(f"Starting to download the file - {PATH_TO_FILE} locally to {LOCAL_PATH}")
        s3.Bucket(BUCKET_NAME).download_file(PATH_TO_FILE, LOCAL_PATH)
        print(f"Downloaded the file - {PATH_TO_FILE} locally to {LOCAL_PATH}")
    except botocore.exceptions.ClientError as e:
        print(traceback.format_exc())
        if e.response['Error']['Code'] == "404":
            print("The object does not exist.")
        else:
                raise
    
def extract_zip_file(file_path, unzip_location=""):
    try:
        print(f"Starting to unzip file - {file_path}")
        if not unzip_location:
            unzip_location = os.path.dirname(file_path)
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            zip_ref.extractall(unzip_location)
        print(f"Unzipped the file to - {unzip_location}")
    except Exception as e:
        print(traceback.format_exc())
