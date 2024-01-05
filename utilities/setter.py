"""
    setter
    create by: Diego Fernando Cardona Pineda
    Date: 05/01/2024

    This script contains access to the GitHub Actions service account key, 
    allowing connection to Google Cloud and its services.
"""
import os
from base64 import b64decode

def main():
    key = os.environ.get('SERVICE_ACCOUNT_KEY')
    with open('path.json', 'w') as json_file:
        json_file.write(b64decode(key).decode())
    print(os.path.realpath('path.json'))

if __name__ == '__main__':
    main()