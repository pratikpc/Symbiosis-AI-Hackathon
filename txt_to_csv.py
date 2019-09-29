import pandas as pd
import numpy as np

lines = []
with open('/Users/kad99kev/Desktop/AI-Hackathon-Preliminary/all_data/manual_text/typed_breakdown.txt', 'r') as file:
    for line in file.readlines():
        lines.append(line)
        
df = pd.DataFrame({"text": lines})
df['label'] = np.full(df.shape[0], 2, dtype='int')
df.to_csv('typed_breakdown2.csv', index=False)
print('Done') 