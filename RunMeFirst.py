# -*- coding: utf-8 -*-
"""
DATS6501: Capstone
Spring 2020
"""
from ftplib import FTP
from zipfile import ZipFile
import os

#This code connects to the U.S. Census Bureau's ftp site downloads and extracts
# the data zip file then deletes the zip file.
ftp = FTP('ftp2.census.gov')
ftp.login()
for year in ['2007','2012', '2018']:
    bases = '/programs-surveys/acs/data/pums/'
    suffix = '/1-Year/'
    path2file = bases + year + suffix
    ftp.cwd(path2file)
    datadir = os.path.join('.','data')    
    # Get All Files
    files = ['csv_pus.zip', 'csv_hus.zip']
    
    # Print out the files
    for file in files:
        print("Downloading..." + file)
        with open(file, 'wb') as fp:
            ftp.retrbinary("RETR " + file , fp.write)
        
        with ZipFile(file, 'r') as zip:
            # extracting all the files 
            print('-------Extracting files-------') 
            zip.extractall(path=datadir)
            
        path = os.path.join(os.getcwd(), file)
        os.remove(path)
ftp.quit()


