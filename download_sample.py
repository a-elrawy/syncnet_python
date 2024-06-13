# Download random sample from the gcp bucket syncnet

import os
import random
import subprocess
import argparse

from google.cloud import storage
from google.oauth2 import service_account

# Set the path to the credentials file
parser = argparse.ArgumentParser()
parser.add_argument('--credentials', help='Path to the credentials file')
args = parser.parse_args()

# Set the number of samples to download
num_samples = 500

bucket_name = 'syncnet'

credentials = args.credentials

bucket = storage.Client.from_service_account_json(credentials).get_bucket(bucket_name)

# Get the list of files in the bucket
blobs = bucket.list_blobs()

# Get the list of files in the bucket
files = [blob.name for blob in blobs]

# Get the list of files to download
sample_files = random.sample(files, num_samples)

# Download the files
for file in sample_files:
    print(f'Downloading {file}')
    blob = bucket.blob(file)
    blob.download_to_filename('random_sample/' + file)