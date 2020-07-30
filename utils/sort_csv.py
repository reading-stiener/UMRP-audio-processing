import csv
import os

def create_p_np_annotations(folder_path): 
    """
    Creates csv annotations for dataset

    Parameters
    ----------

    folder_path : str
    
    Returns
    -------

    None
    """
    sorted_filelist = [] 
    for r_dir, sub_dir_list, file_list in os.walk(folder_path):
        file_list.sort(key = lambda x : int(x.split('_')[2]))
        with open(os.path.join(r_dir, 'cleaned_dataset.csv'), 'w') as csvfile:
            fieldnames = ['file_name', 'class' ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for file_path in file_list:
                if 'np' in file_path:
                    writer.writerow({'file_name': os.path.join(r_dir, file_path), 'class': 'np'})
                else: 
                    writer.writerow({'file_name': os.path.join(r_dir, file_path), 'class': 'p'})



if __name__ == '__main__':
    create_p_np_annotations(folder_path='/home/camel/Documents/URMP_audio_processing/LSTM/data/violin')