import pandas as pd 
import os
from tqdm import tqdm
import numpy as np
from pycocotools.coco import COCO
import skimage.io as io
import matplotlib.pyplot as plt
from pathlib import Path
import json
from numpyencoder import NumpyEncoder



dataDirectory = input("Enter the image folder:: ")
annFile = input("Enter the JSON filename:: ")
num_keypoints = int(input("Enter number of keypoints you need:: "))

dataDir=Path(dataDirectory)
annFile = Path(annFile)
coco = COCO(annFile)
imgIds = coco.getImgIds()
imgs = coco.loadImgs(imgIds)

df = pd.DataFrame(imgs)
c=0
filenames = []
ann_data = []
imageidList = []
segmentationList = []
num_keypointsList = []
areaList = []
iscrowdList = []
keypointsList = []
image_idList = []
bboxList = []
category_idList = []
imgidList = []

def annotationDetails(dataDirectory,annFile):
    for img in imgs:
        I = io.imread(dataDir/img['file_name'])
        fn = img['file_name']
        filenames.append(fn)
        imgidss=[img['id']] #to append image id
        annIds = coco.getAnnIds(imgIds=[img['id']])
    #     print(imgidss,annIds)
        anns = coco.loadAnns(annIds)
        if len(anns) > 0 :
            segmentationList.append(anns[0]['segmentation'])
            num_keypointsList.append(anns[0]['num_keypoints'])
            keypointsList.append(anns[0]['keypoints'])
            areaList.append(anns[0]['area'])
            iscrowdList.append(anns[0]['iscrowd'])
            image_idList.append(anns[0]['image_id'])
            category_idList.append(anns[0]['category_id'])
            imgidList.append(anns[0]['id'])
            bboxList.append(anns[0]['bbox'])
        else:
            segmentationList.append(0)
            num_keypointsList.append(0)
            keypointsList.append(0)
            areaList.append(0)
            iscrowdList.append(0)
            image_idList.append(0)
            category_idList.append(0)
            imgidList.append(0)
            bboxList.append(0)
        imageidList.append(imgidss)
        ann_data.append(anns)
    print("===========================================")
    #print(ann_data)
    annot_df = pd.DataFrame()
    annot_df['imagepath'] = imgs
    annot_df['filename'] = filenames
    annot_df['annotations'] = ann_data
    annot_df['segmentation'] = segmentationList
    annot_df['num_keypoints'] = num_keypointsList
    annot_df['keypoints'] = keypointsList
    annot_df['area'] = areaList
    annot_df['crowd'] = iscrowdList
    annot_df['imgid'] = image_idList
    annot_df['category_id'] = category_idList
    annot_df['id'] = imgidList
    annot_df['bbox'] = bboxList
    print(annot_df.head(5))
    annot_df.to_csv('funcCSV.csv')
    return annot_df

def getDesiredData(num_keypoints):
    data = pd.read_csv('funcCSV.csv')
    data = data.loc[(data['num_keypoints'] > num_keypoints)]
    data["id"] = data["imgid"]
    data.to_csv('DataWithDesiredKeypoints.csv')
    return data

def ImageDetails(dataDirectory,annFile):
    license = []
    flname = []
    curl = []
    height = []
    width = []
    dtcap = []
    flurl = []
    idlist = []
    for i in range(0,len(imgs)):
        ls = (imgs[i]['license'])
        fn = (imgs[i]['file_name'])
        cu = (imgs[i]['coco_url'])
        h = (imgs[i]['height'])
        w = (imgs[i]['width'])
        dc = (imgs[i]['date_captured'])
        fu = (imgs[i]['flickr_url'])
        ids = (imgs[i]['id'])
        license.append(ls)
        flname.append(fn)
        curl.append(cu)
        height.append(h)
        width.append(w)
        dtcap.append(dc)
        flurl.append(fu)
        idlist.append(ids)

    imagedata_df = pd.DataFrame()
    imagedata_df['license'] = license
    imagedata_df['file_name'] = flname
    imagedata_df['coco_url'] = curl
    imagedata_df['height'] = height
    imagedata_df['width'] = width
    imagedata_df['date_captured'] = dtcap
    imagedata_df['flickr_url'] = flurl
    imagedata_df['id'] = idlist
    imagedata_df.to_csv('anndata.csv')
    return imagedata_df

def detailsofImage(data,imagedata_df):
    finaldf = pd.merge(data, imagedata_df, on='id')
    finaldf.to_csv("MergerdCSVFiles.csv")
    data = pd.read_csv("MergerdCSVFiles.csv")
    dictObj = []
    images = []
    infodetails = {"description": "COCO 2017 Dataset","url": "http://cocodataset.org","version": "1.0","year": 2017,"contributor": "COCO Consortium","date_created": "2017/09/01"}
    licensedetail =[{"url": "http://creativecommons.org/licenses/by-nc-sa/2.0/","id": 1,"name": "Attribution-NonCommercial-ShareAlike License"},{"url": "http://creativecommons.org/licenses/by-nc/2.0/","id": 2,"name": "Attribution-NonCommercial License"},{"url": "http://creativecommons.org/licenses/by-nc-nd/2.0/","id": 3,"name": "Attribution-NonCommercial-NoDerivs License"},{"url": "http://creativecommons.org/licenses/by/2.0/","id": 4,"name": "Attribution License"},{"url": "http://creativecommons.org/licenses/by-sa/2.0/","id": 5,"name": "Attribution-ShareAlike License"},{"url": "http://creativecommons.org/licenses/by-nd/2.0/","id": 6,"name": "Attribution-NoDerivs License"},{"url": "http://flickr.com/commons/usage/","id": 7,"name": "No known copyright restrictions"},{"url": "http://www.usa.gov/copyright.shtml","id": 8,"name": "United States Government Work"}]
    for i in range(len(data)):
        dictimg = {
            'license': (data.license[i]),
            'file_name': data.filename[i],
            'coco_url': data.coco_url[i],
            'height': (data.height[i]),
            'width': (data.width[i]),
            'date_captured': data.date_captured[i],
            'flickr_url': data.flickr_url[i],
            'id': (data.id[i])
        }
        images.append(dictimg)
        
    annotations = []
    for i in range(len(data)):
        annimg = {
        'segmentation': segmentationList[i],    #directly append the list
        'num_keypoints': data.num_keypoints[i],
        'area': data.area[i],
        'iscrowd': (data.crowd[i]),
        'keypoints': (keypointsList[i]),           #directly append list
        'image_id': data.imgid[i],
        'bbox': bboxList[i],                        #directly append list
        'category_id': (data.category_id[i]),
        'id':(data.id[i])
        }
        annotations.append(annimg)
        
    dictionary = {
    "info": infodetails,
    "licenses": licensedetail,
    "images": images,
    "annotations":annotations,
    "categories": [
        {
            "supercategory": "person",
            "id": 1,
            "name": "person",
            "keypoints": ["nose","left_eye","right_eye","left_ear","right_ear","left_shoulder","right_shoulder","left_elbow","right_elbow","left_wrist","right_wrist","left_hip","right_hip","left_knee","right_knee","left_ankle","right_ankle"],
            "skeleton": [[16,14],[14,12],[17,15],[15,13],[12,13],[6,12],[7,13],[6,7],[6,8],[7,9],[8,10],[9,11],[2,3],[1,2],[1,3],[2,4],[3,5],[4,6],[5,7]]
        }
        ]
    }

    json_object = json.dumps(dictionary, indent = 4,cls=NumpyEncoder)
  
    # Writing to sample.json
    with open("expFINAL.json", "w") as outfile:
        outfile.write(json_object)
        
    return dictionary

def main(dataDirectory,annFile,num_keypoints):
    annotations_data = annotationDetails(dataDirectory,annFile)
    data = getDesiredData(num_keypoints)
    imagedata_df = ImageDetails(dataDirectory,annFile)
    detailsImage = detailsofImage(data,imagedata_df)
    return detailsImage
    
main(dataDirectory,annFile,num_keypoints)
