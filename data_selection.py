###Selecting data###
import pandas as pd
import sklearn
import glob
import os
import csv
import matplotlib.pyplot as plt

###Filtering out the unnecessary columns###

f = pd.read_csv("C:/Users/Reinis Fisers/Desktop/University Year 4/Individual Project/Extracted data/2018-06.csv")
keep_col = ['TimeStamp', 'WindSpeed_mps', 'WindSpeed1', 'WindSpeed2', 'WindSpeed3', 'Pitch_Deg', 'AmbTemp_DegC',
            'NacelleOrientation_Deg']
new_f = f[keep_col]
new_f.to_csv("2018-06f.csv", index=False)

### Comibining all the files into one###

# files = glob.glob("C:/Users/Reinis Fisers/Desktop/University Year 4/Individual Project/filtered data/*.csv")
# df = pd.concat((pd.read_csv(f, header = 0) for f in files))
# df.to_csv("combined.csv")
