# Extracting MS COCO Dataset
MS COCO (Microsoft Common Objects in Context) is a large-scale image dataset containing 328,000 images of everyday objects and humans. The dataset contains annotations you can use to train machine learning models to recognize, label, and describe objects. 

This is code to extract the dataset w.r.t class like person,vehicle,cat etc. from the original dataset and create the JSON file respectively.

MS_COCO Dataset provides 80 classes let us consider that we want to train a model for particular class. We would require image files and their respective annotations from the original dataset.

Steps to implement:
1. Download the original dataset and annotatios from following link https://cocodataset.org/#download 
2. Execute code.py It will ask for foldername, name of annotation JSON file and number of keypoints
   For ex: 
   foldername : val2017
   jsonfile : person_keypoints_val2017.json
   num_keypoints : 6 
   It will provide us JSON file with image details and annotations for images having number of keypoints > 6
3. Exectute Images.py this script will copy the images having num_keypoints > 6 to another directory 

P.S: Kindly follow same procedure for train dataset
