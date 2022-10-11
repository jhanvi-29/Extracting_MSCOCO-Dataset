import pandas as pd 
import os
from tqdm import tqdm
import numpy as np
from pycocotools.coco import COCO
import skimage.io as io
import matplotlib.pyplot as plt
from pathlib import Path
from pandas import *
import glob
import shutil
import os

src_dir = input("Enter source directory full path:")
dst_dir = input("Enter destination directory full path:")
ImageCount = 0
def getImageFiles(src_dir,dst_dir):
    data = pd.read_csv("DataWithDesiredKeypoints.csv")
    fileList = data['filename'].tolist()
    for image in fileList:
        ImageCount + 1
        for jpgfile in glob.iglob(os.path.join(src_dir, image)):
            shutil.copy(jpgfile, dst_dir)
    print("Files Copied!")
    return ImageCount


def main():
    imageFiles = getImageFiles(src_dir,dst_dir)
    return imageFiles
    
main()
