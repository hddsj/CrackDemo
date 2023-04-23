# 作者: hxd
# 2023年04月11日10时51分36秒
from sklearn.model_selection import train_test_split
import os

imagedir = 'CrackLS315/CrackLS315'
outdir = 'FCNDemo'

images = []
for file in os.listdir(imagedir):
    filename = file.split('.')[0]
    images.append(filename)

train, test = train_test_split(images, train_size=0.7, random_state=0)
val, test = train_test_split(test, train_size=0.2 / 0.3, random_state=0)

with open(outdir + "train.txt", 'w') as f:
    f.write('\n'.join(train))

with open(outdir + "val.txt", 'w') as f:
    f.write('\n'.join(val))

with open(outdir + "test.txt", 'w') as f:
    f.write('\n'.join(test))
