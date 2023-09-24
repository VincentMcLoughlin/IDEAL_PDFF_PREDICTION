from zipfile import ZipFile
from glob import glob
import pandas as pd

def extract_folder_name_from_zip_path(file_path):
    return file_path[file_path.rfind("/")+1: len(file_path)-4]

def main():
    
    #zip_files_path = "/genetics3/mclougv/LIVER_MRI/raw_data_field_20203/*/*.zip"
    #zip_files = glob(zip_files_path)
    #output_folder = "/home/mclougv/IDEAL_PDFF_prediction/mini_ideal_dicom/"
    output_folder = "/genetics3/mclougv/IDEAL_X20254/"
    #data_list_df = pd.read_csv("/home/mclougv/IDEAL_PDFF_prediction/reference_data/IDEAL_x20254_2_clean_mini.csv")
    data_list_df = pd.read_csv("/home/mclougv/IDEAL_PDFF_prediction/reference_data/IDEAL_x20254_2_clean_with_path.csv")
    zip_files = data_list_df["full_path"]    
    print(len(zip_files))
    i = 0
    output_files = []
    for zip_file_path in zip_files:        
        if i%100 == 0:
            print(i)
        archive = ZipFile(zip_file_path, 'r')
        folder_name = extract_folder_name_from_zip_path(zip_file_path)
        #destination = "/genetics3/mclougv/IDEAL_X20254/" + folder_name
        destination = output_folder + folder_name
        output_files.append(destination)
        archive.extractall(destination)

        i += 1
    data_list_df["dicom_data_folder"] = output_files
    data_list_df.to_csv("reference_data/IDEAL_x20254_2_clean_mini_w_extracted.csv")
if __name__ == "__main__":
    main()