import csv
import os

def create_p_np_annotations(folder_path, skip): 
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
        # sort the file list in the order of file numbers
        if r_dir != folder_path:
            return
        file_list.sort(key = lambda x : int(x.split('_')[2]))
        with open(os.path.join(r_dir, 'dataset_prop', 'train_prop.csv'), 'w') as csvfile:
            fieldnames = ['file_name', 'class' ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            count = 0
            for file_path in file_list:
                if 'np' in file_path:
                    writer.writerow({'file_name': os.path.join(r_dir, file_path), 'class': 'np'})
                else:
                    if count < skip:
                        count += 1
                    else:
                        count = 0
                        writer.writerow({'file_name': os.path.join(r_dir, file_path), 'class': 'p'})

def count_p_np(file_csv_path):
    p_count = 0
    np_count = 0
    with open(file_csv_path, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if row['class'] == 'p':
                p_count += 1
            else:
                np_count += 1
    return p_count, np_count

if __name__ == '__main__':
    create_p_np_annotations(folder_path='/home/camel/Documents/URMP_audio_processing/LSTM/data/violin/raw', skip=10)
    print(count_p_np(file_csv_path='/home/camel/Documents/URMP_audio_processing/LSTM/data/violin/raw/dataset_prop/train_prop.csv'))