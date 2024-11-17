# Building a MovieLens Recommender System Tutorial

## Overview

This tutorial walks through building different types of recommender systems using the MovieLens dataset. The MovieLens dataset contains 1,000,209 anonymous ratings of approximately 3,900 movies made by 6,040 MovieLens users who joined MovieLens in 2000.

## Tutorial Structure

The tutorial is broken down into 6 chapters:

1. Loading the data and exploratory data analysis
2. (Heuristics) Popularity-based recommendations
3. (Machine Learning) Content-based filtering
4. (Machine Learning) Collaborative Filtering
5. Evaluation Metrics for Recommender Systems
6. (Deep Learning) Two-tower Neural Networks

## Dataset Description

The MovieLens dataset includes:

- **Ratings**: 1,000,209 ratings on a 5-star scale
- **Movies**: ~3,900 movies with titles and genres
- **Users**: 6,040 users with demographic information

Key dataset characteristics:
- UserIDs range between 1 and 6040
- MovieIDs range between 1 and 3952
- Ratings are made on a 5-star scale (whole-star ratings only)
- Each user has at least 20 ratings

## Requirements

The tutorial uses the following Python libraries:

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer
from wordcloud import WordCloud, STOPWORDS
```

## Getting Started

1. Clone this repository
2. Install the required dependencies
3. The data files should be placed in a `data-1m/` directory with the following files:
   - ratings.csv
   - movies.csv
   - users.csv

## Usage License

As per the MovieLens dataset terms:

- The user may not state or imply any endorsement from the University of Minnesota or the GroupLens Research Group
- The user must acknowledge the use of the data set in publications
- The user may not redistribute the data without separate permission
- The user may not use this information for commercial purposes without permission

## Citation

To acknowledge use of the dataset in publications, please cite:

F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens Datasets: History and Context. ACM Transactions on Interactive Intelligent Systems (TiiS) 5, 4, Article 19 (December 2015), 19 pages. DOI=http://dx.doi.org/10.1145/2827872

## Acknowledgements

Thanks to GroupLens Research Project at the University of Minnesota for providing the MovieLens dataset and Shyong Lam and Jon Herlocker for cleaning up and generating the data set.

## Additional Resources

For more information about GroupLens Research:
- Website: http://www.grouplens.org/
- MovieLens: http://www.movielens.org/