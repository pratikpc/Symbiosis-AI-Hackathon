import pandas as pd
import numpy as np

df = pd.read_csv('testdrive_form.csv')
new_df = df.iloc[:,1]
for index in range(2, df.shape[1]):
    new_df = new_df.append(df.iloc[:,index], ignore_index=True)

with open('./all_data/manual_text/typed_testdrive.txt', 'r') as file:
    for line in file.readlines():
        new_df = new_df.append(pd.DataFrame([line]), ignore_index=True)
        
new_df = new_df.dropna()
new_df = pd.DataFrame(new_df)
new_df.rename(columns={0: 'text'}, 
                 inplace=True)

new_df['label'] = np.full(new_df.shape[0], 1, dtype='int')

new_df.to_csv('new_testdrive.csv', index=False)