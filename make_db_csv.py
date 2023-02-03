import pandas as pd
import os

# data_info: Dataframe with DB information for us to use [folder, image name, class]
root = 'C:/Users/rlawj/sample_DB/'
folders = ['train', 'test']
labels = ['cats', 'dogs']

# names = os.listdir(f'{root}/{folder}/{label[0]}')

columns = ['folder', 'image_name', 'class']

for folder in folders:
    values = []
    for label in labels:
        if label == 'cats':
            temp_label = 0
        else:
            temp_label = 1
        names = os.listdir(f'{root}/{folder}/{label}')
        for name in names:
            values.append([label, name, temp_label])

    data_info_df = pd.DataFrame(data=values, columns=columns)
    data_info_df.to_csv(f'{root}/{folder}.csv', index=False)