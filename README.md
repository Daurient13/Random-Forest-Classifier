# Random Forest Algorithm

This algorithm is a combination of each tree from the decision tree which is then combined into a single model. 

Random Forest is an algorithm for classification. Then, how does it work? Random Forest works by building several decision trees and combining them to get more stable and accurate predictions. The 'forest' built by Random Forest is a collection of decision trees which are usually trained by the bagging method. The general idea of ​​the bagging method is a combination of learning models to improve overall results

The Random Forest algorithm increases the randomness of the model while growing the tree. Instead of looking for the most important feature when splitting a node, Random Forest looks for the best feature among a random subset of features. As a result, this method produces a wide variety and generally results in better models.

![image](https://user-images.githubusercontent.com/86812576/167052892-bde666a6-f8fe-4a27-bcf5-12d73f1b1092.png)

# Dataset

In this project, we are going to predict class of mobile phone. The dataset consists of 2000 rows and 21 columns. and we have prepared a description of each column below:

![image](https://user-images.githubusercontent.com/86812576/167053858-e7eea186-3094-4ad0-b59b-267534ad5a4e.png)

id              : ID

battery_power   : battery capacity (mAh)

blue            : bluetooth support or not

clock_speed     : clock speed microprocessor

dual_sim        : dual sim support or not

fc              : front camera (megapixel)

four_g          : 4G support or not

int_memory      : internal memory (GB)

m_dep           : thickness (cm)

mobile_wt       : mobile weight (g)

n_cores         : number of processor cores

pc              : main camera (megapixel)

px_height       : pixel resolution (height)

px_width        : pixel resolution (width

ram             : RAM (GB)

sc_h            : screen height (cm)

sc_w            : screen width (cm)

talk_time       : how long the battery lasts when calling

three_g         : 3G support or not

touch_screen    : touch screen support or not

wifi            : wifi support or not

**price_range     : price range (as target)**

# Import Package

import common package:

import **numpy as np**

import **pandas as pd**

from **sklearn.model_selection** import **train_test_split**

from **sklearn.pipeline** import **Pipeline**

from **sklearn.compose** import **ColumnTransformer**

from **jcopml.utils** import **save_model, load_model**

from j**copml.pipeline** import **num_pipe, cat_pipe**

from **jcopml.plot** import **plot_missing_value**

from **jcopml.feature_importance** import **mean_score_decrease**

import Algorithm's Package:

from **sklearn.ensemble** import **RandomForestClassifier**

from **sklearn.model_selection** import **GridSearchCV**

from **jcopml.tuning** import **grid_search_params as gsp**

# Import Data

which i have explained before, the dataset has a column index called ID

# Mini Exploratory Data Analysis

I always work on data science projects with simple think so that I can benchmark. Using a simple model to benchmark. And most of the time it's more efficient and sometimes find a good one. but at the beginning I did mini Exploratory Data Analysis. because i focus more on the algorithm.

