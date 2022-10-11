# Extracting_COCO-Dataset
Code to extract the dataset w.r.t class like person,vehicle,cat etc from the original dataset and create the JSON file accordingly.

MS_COCO Dataset provides 80 classes let us consider that we want to train a model for particular class. We would require image files and their respective annotations from the original dataset.
Code.py provides the annotations and image details in form of json file. Give the foldername, JSON file name and the num_keypoints for class. For ex: foldername : val2017
jsonfile : person_keypoints_val2017.json
num_keypoints : 6 
Here i am considering person as a class and considering all the annotations that have more than 6 keypoints as my data.

Images.py provides all the corresponding images from dataset folder that have num_keypoints > 6 
