import cv2
import numpy as np
from scipy import ndimage
import os
import errno
import sys
import logging
import shutil
import matplotlib.pyplot as plt

# size 120x
def read_images_from_single_face_profile(face_profile, face_profile_name_index, dim = (120, 120)):
    """
    Reads all the images from one specified face profile into ndarrays
    Parameters
    ----------
    face_profile: string
        The directory path of a specified face profile
    face_profile_name_index: int
        The name corresponding to the face profile is encoded in its index
    dim: tuple = (int, int)
        The new dimensions of the images to resize to
    Returns
    -------
    X_data : numpy array, shape = (number_of_faces_in_one_face_profile, face_pixel_width * face_pixel_height)
        A face data array contains the face image pixel rgb values of all the images in the specified face profile 
    Y_data : numpy array, shape = (number_of_images_in_face_profiles, 1)
        A face_profile_index data array contains the index of the face profile name of the specified face profile directory
    """
    index = 0
    
    print(face_profile)
    for the_file in os.listdir(face_profile):
        file_path = os.path.join(face_profile, the_file)
    
        print(file_path)
        if file_path.endswith(".png") or file_path.endswith(".jpg") or file_path.endswith(".jpeg") or file_path.endswith(".pgm"):
            
            img = cv2.imread(file_path, 0)
            # print(img.shape)
            # cv2.waitKey(0)
            img = cv2.resize(img, (120, 160))
            cv2.imwrite(file_path, img)
            
            index += 1

    if index == 0 : 
        shutil.rmtree(face_profile)
        logging.error("\nThere exists face profiles without images")


def delete_empty_profile(face_profile_directory):
    """
    Deletes empty face profiles in face profile directory and logs error if face profiles contain too little images
    Parameters
    ----------
    face_profile_directory: string
        The directory path of the specified face profile directory
    """
    for face_profile in os.listdir(face_profile_directory):
        if "." not in str(face_profile):
            face_profile = os.path.join(face_profile_directory, face_profile)
            index = 0
            for the_file in os.listdir(face_profile):
                file_path = os.path.join(face_profile, the_file)
                if file_path.endswith(".png") or file_path.endswith(".jpg") or file_path.endswith(".jpeg") or file_path.endswith(".pgm"):
                    index += 1
            if index == 0 : 
                shutil.rmtree(face_profile)
                print("\nDeleted ", face_profile, " because it contains no images")
            if index < 2 : 
                logging.error("\nFace profile " + str(face_profile) + " contains too little images (At least 2 images are needed)")


def load_training_data(face_profile_directory):
    """
    Loads all the images from the face profile directory into ndarrays
    Parameters
    ----------
    face_profile_directory: string
        The directory path of the specified face profile directory
    face_profile_names: list
        The index corresponding to the names corresponding to the face profile directory
    Returns
    -------
    X_data : numpy array, shape = (number_of_faces_in_face_profiles, face_pixel_width * face_pixel_height)
        A face data array contains the face image pixel rgb values of all face_profiles
    Y_data : numpy array, shape = (number_of_face_profiles, 1)
        A face_profile_index data array contains the indexs of all the face profile names
    """
    # delete_empty_profile(face_profile_directory)  # delete profile directory without images

    # Get a the list of folder names in face_profile as the profile names
    face_profile_names = [d for d in os.listdir(face_profile_directory) if "." not in str(d)]

    if len(face_profile_names) < 2: 
        logging.error("\nFace profile contains too little profiles (At least 2 profiles are needed)")
        exit()
    # print('cc')
    # print(face_profile_names)
    first_data = str(face_profile_names[0])
    first_data_path = os.path.join(face_profile_directory, first_data)
    read_images_from_single_face_profile(first_data_path, 0)
    
    
    for i in range(1, len(face_profile_names)):
        directory_name = str(face_profile_names[i])
        directory_path = os.path.join(face_profile_directory, directory_name)
        read_images_from_single_face_profile(directory_path, i)



load_training_data('./datasets')









