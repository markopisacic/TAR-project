import os
import random
from shutil import copyfile

# split ratios
TRAIN = 0.65
VALIDATE = 0.15
TEST = 0.20

data = os.listdir('./labeled_articles')
random.shuffle(data)

train_n = int(len(data) * TRAIN)
validate_n = int(len(data) * VALIDATE)

train = data[0:train_n]
validate = data[train_n:(train_n + validate_n)]
test = data[(train_n + validate_n):]

# delete all existing files in folders
for file in os.listdir('./train'): os.unlink(os.path.join('./train', file))
for file in os.listdir('./validate'): os.unlink(os.path.join('./validate', file))
for file in os.listdir('./test'): os.unlink(os.path.join('./test', file))

# copy new files to folders
for file in train: copyfile('./labeled_articles/' + file, './train/' + file)
for file in validate: copyfile('./labeled_articles/' + file, './validate/' + file)
for file in test: copyfile('./labeled_articles/' + file, './test/' + file)

# merge all in one file for use in Flair datasets
for split_type in ['train', 'validate', 'test']:
    out_file = open('./' + split_type + '.txt', 'w+')
    for filename in os.listdir('./' + split_type):
        with open('./' + split_type + '/' + filename) as file:
            text = file.read()
            out_file.write(text)
        out_file.write("\n")
    out_file.close()

print("Done")
