# FER TAR course project
## Detection of propaganda techniques in news articles

###### Repo for our team's solution (WallstreetBets)
Original shared task data and labels are located in data/articles and data/labels.
* **data_parsing.py** is responsible for generating labeled words from original data, suitable for sequence labeling task
* **split_data.py** splits processed data (located in data/labeled_articles) into train-validate-test split. Also, it merges all articles into one file for each split, which is needed for Flair datasets
* **training.py** is example of Flair usage for sequence labeling tasks. Its output can be some simple baseline, and it is stored in folder resources (ignored by git)
