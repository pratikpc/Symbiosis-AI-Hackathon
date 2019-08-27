import pandas as pd
import numpy as np

lines = []
with open('./test_test.txt', 'r') as file:
    for line in file.readlines():
        lines.append(line)
        
df = pd.DataFrame({"text": lines})
df['label'] = np.full(df.shape[0], 2, dtype='int')
df.to_csv('test_test.csv', index=False)
print('Done') 