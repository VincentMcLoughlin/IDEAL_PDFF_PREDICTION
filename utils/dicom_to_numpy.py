import pandas as pd 
import pydicom
from glob import glob 
import os
import numpy as np
import argparse

# IMAGE_HEIGHT = 232
# IMAGE_WIDTH = 256
# MAX_PIXEL_VALUE = 4095.0

ideal_dict = {"IMAGE_HEIGHT": 232, "IMAGE_WIDTH": 256, "MAX_PIXEL_VALUE": 4095.0, "IMAGES_PER_FOLDER": 72}
gre_dict = {"IMAGE_HEIGHT": 160, "IMAGE_WIDTH": 160, "MAX_PIXEL_VALUE": 4095.0, "IMAGES_PER_FOLDER": 20}

def extract_eid_from_folder_name(folder_name):
    return folder_name.split("_")[0]

def write_frames_to_file(light_data_file_name, dark_data_file_name, light_frames, dark_frames):    

    np.save(light_data_file_name, light_frames)
    np.save(dark_data_file_name, dark_frames)    

def read_data_folder(dicom_folder, light_frames, dark_frames, bad_file_list, config_dict, normalize):
    dark_k = 0
    light_k = 0
    file_list = glob(f"{dicom_folder}/*.dcm") #Max value appears to be 4095
    
    if len(file_list) != config_dict["IMAGES_PER_FOLDER"]: 
        bad_file_list.append(dicom_folder)        
        print(f"Skipping {dicom_folder}, length {len(file_list)}") 
        print(f"Expected length: {config_dict['IMAGES_PER_FOLDER']}, received length: {len(file_list)}, path: {dicom_folder}/*.dcm")
    else:
        for file_name in file_list:
            try:
                image_array = pydicom.dcmread(file_name).pixel_array #get image array
                img_mean = image_array.mean() #Divide between light and dark images        
                if img_mean < 1000:
                    if normalize:
                        dark_frames[:,:,dark_k] = image_array/config_dict["MAX_PIXEL_VALUE"] #Max possible value
                    else:
                        dark_frames[:,:,dark_k] = image_array
                    dark_k += 1
                else: 
                    if normalize:
                        light_frames[:,:,light_k] = image_array/config_dict["MAX_PIXEL_VALUE"]
                    else:
                        light_frames[:,:,light_k] = image_array
                    light_k += 1
            except Exception as e:
                print(f"Error reading filename {file_name}")
                bad_file_list.append(file_name)
                raise e
    
    return light_frames, dark_frames

def main(data_paths_file, mri_config, dark_dir, light_dir, augment=False, normalize=False):    
    
    print(f"Reading from {data_paths_file}")
    data_paths_df = pd.read_csv(data_paths_file)
    scores = np.array(data_paths_df["PDFF"])
    data_paths_df["filename_eid"]  = data_paths_df["filename_from_path"].apply(extract_eid_from_folder_name)
    IMAGE_HEIGHT = mri_config["IMAGE_HEIGHT"]
    IMAGE_WIDTH = mri_config["IMAGE_WIDTH"]

    eids = data_paths_df["filename_eid"]
    curr_list_of_scores = []
    bad_file_list = []    
    num_samples = int(mri_config["IMAGES_PER_FOLDER"]/2) #Equal numbers of light and dark per folder    

    dark_augmented_dir = f"{dark_dir}_flipped"
    light_augmented_dir = f"{light_dir}_flipped"

    i = 0
    for path, filename_eid in zip(data_paths_df["dicom_data_folder"], data_paths_df["filename_eid"]):
        light_frames = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH, num_samples), dtype=np.float32)
        dark_frames = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH, num_samples), dtype=np.float32)
        
        if i%20 == 0: 
            print(f"{i} : {path} : {filename_eid}")        

        try:
            old_len = len(bad_file_list)
            light_frames, dark_frames = read_data_folder(path, light_frames, dark_frames, bad_file_list, mri_config, normalize)            

        except Exception as e:
            print(f"Error reading files at path {path}")            
            print(f"Current Index {i}")
            print(e)
            with open('bad_gre_numpys.txt', 'w') as f:
                for entry in bad_file_list:
                    f.write(f"{entry}\n")
            #raise(e)        

        light_frames_flipped = np.fliplr(light_frames)
        dark_frames_flipped = np.fliplr(dark_frames)

        light_file_name = f"{filename_eid}_light"
        dark_file_name = f"{filename_eid}_dark"
        light_full_path = os.path.join(light_dir, light_file_name)
        dark_full_path = os.path.join(dark_dir, dark_file_name)                
        write_frames_to_file(light_full_path, dark_full_path, light_frames, dark_frames)

        if augment:
            light_flipped_full_path = os.path.join(light_augmented_dir, f"{light_file_name}_flipped")
            dark_flipped_full_path = os.path.join(dark_augmented_dir, f"{dark_file_name}_flipped")
            write_frames_to_file(light_flipped_full_path, dark_flipped_full_path, light_frames_flipped, dark_frames_flipped)

        i += 1

    with open('bad_gre_reads.txt', 'w') as f:
        for entry in bad_file_list:
            f.write(f"{entry}\n")

if __name__ == "__main__":
    import tempfile
    print(tempfile.gettempdir())
    tempfile.tmpdir = "/home/mclougv/tmp"
    parser = argparse.ArgumentParser(description="Sort dcm files into numpy files")
    parser.add_argument("-f", "--file")
    parser.add_argument("-t", "--type")
    parser.add_argument("-d", "--dark_directory", help="Output folder for dark MRIs")
    parser.add_argument("-l", "--light_directory", help="Output folder for light MRIs")
    parser.add_argument("-n", "--normalize", default=False, help="Divide numpy arrays for each MRI by max pixel value (4095)")
    parser.add_argument("-a", "--augment", default=False, help="Augment by flipping images from left to right")

    args = parser.parse_args()
    config = vars(args)
    print(config)

    file_path = config["file"]
    MRI_type = config["type"]
    augment = config["augment"]
    normalize = config["normalize"]
    dark_dir = config["dark_directory"]
    light_dir = config["light_directory"]    

    if MRI_type == "IDEAL":
        mri_config = ideal_dict
    elif MRI_type == "GRE":
        mri_config = gre_dict
    else:
        print("Unsupported MRI type. Script only accepts IDEAL or GRE MRIs.\
             Use IDEAL or GRE as the input to the script's type argument.")
        exit
    
    #python3 utils/dicom_to_numpy.py -f reference_data/mini_IDEAL_x20254_2_with_Path.csv -t IDEAL -d /home/mclougv/IDEAL_PDFF_prediction/X20254_mini_dark_files -l /home/mclougv/IDEAL_PDFF_prediction/X20254_mini_light_files
    #python3 utils/dicom_to_numpy.py -f reference_data/mini_IDEAL_with_extracted_paths.csv -t IDEAL -d /home/mclougv/IDEAL_PDFF_prediction/X20254_mini_dark_files -l /home/mclougv/IDEAL_PDFF_prediction/X20254_mini_light_files
    #python3 utils/dicom_to_numpy.py -f reference_data/micro_IDEAL_with_extracted_paths.csv -t IDEAL -d /home/mclougv/IDEAL_PDFF_prediction/X20254_micro_dark_files -l /home/mclougv/IDEAL_PDFF_prediction/X20254_micro_light_files
    #python3 utils/dicom_to_numpy.py -f reference_data/IDEAL_x20254_2_clean_w_extracted.csv -t IDEAL -d /genetics3/mclougv/IDEAL_x20254_dark_files/ -l /genetics3/mclougv/IDEAL_x20254_light_files/
    main(file_path, mri_config, dark_dir, light_dir, augment, normalize)