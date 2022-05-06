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

which i have explained before.

# Mini Exploratory Data Analysis

I always work on data science projects with simple think so that I can benchmark. Using a simple model to benchmark. And most of the time it's more efficient and sometimes find a good one. but at the beginning I did mini Exploratory Data Analysis. because i focus more on the algorithm.

We check whether there is any missing data or not. We can see that our data is clean.
For now there are no features removed, and will be done at the time of feature importance will use them all and go straight to dataset splitting.

# Dataset Splitting

split the data into X, and y

X = all columns except the target column.

y = 'price_range' as target

test_size = 0.2 (which means 80% for train, and 20% for test)

# Training

In the Training step there are 3 main things that I specify.

First, the preprocessor: here the columns will be grouped into numeric and categoric.

included in the numeric column are: 'battery_power', 'clock_speed', 'fc', 'int_memory', 'm_dep', 'mobile_wt', 'n_cores', 'pc',
                            'px_height', 'px_width', 'ram', 'sc_h', 'sc_w', 'talk_time'.

and in the categoric column are: 'blue', 'dual_sim', 'four_g', 'three_g', 'touch_screen', 'wifi'. 

second, pipeline: contains the preprocessor as 'prep' which I defined earlier, and the algorithm as 'algo' which in this case I use Random Forest Classifier.

and third, tuning with Grid Search: in this case I use the tuning recommendations (gsp.rf_params) that often occur in many cases. but does not rule out hyperparameter tuning if the model results are not good. with cross validation = 3.

**GridSearch Parameters Recommendation :**

**{'algo__n_estimators': [100, 150, 200],**

**'algo__max_depth': [20, 50, 80],**

**'algo__max_features': [0.3, 0.6, 0.8]**,

**'algo__min_samples_leaf': [1, 5, 10]}**

# Results and Feature Importance

![RF clas](https://user-images.githubusercontent.com/86812576/167058603-6407dbd4-b5cd-4a1c-ad26-e9ec59cb2459.png)

From the above results it can be seen that in training the accuracy reaches 100%, and the test accuracy reaches 92%.

### Feature Importance

![image](https://user-images.githubusercontent.com/86812576/167059365-377bdc8f-8a98-4d74-be97-b587a3ee3882.png)

it turns out that if we look at the Feature Importance (Mean Square Decrease) the most important features are:

1. 'ram' 
 
2. 'battery_power'
 
3. 'px_width'
 
4. 'px_height'

5. 'mobile_wt'

And other features don't seem to have any effect. So we can cut out the features, and focus on those five features. why do we want to cut features? why don't we use all the features? because the machine has a weakness. If we provide more information, then the pattern will be more difficult to find. because the machine only looks for patterns. it's like a maze, but if we give things that are important, specific to something then it will be easier to find, generalizing and produce a better model

We can't guarantee, but most of the time it can helps. it could be a better model, or it could be worst.

### Model After Feature Importance

Now we only select 5 features to the model after Feature Importance, and all these 5 features is categoric columns. Here's the result:

![RF clas 2](https://user-images.githubusercontent.com/86812576/167062392-d11ee1f5-950d-464a-83c0-569f1029bf2b.png)

the result is an increase in accuracy of about 2.5%. And with lighter computing because it uses fewer features.

Actually cutting features can help, because if we give a lot of information then the machine will be confused because many features are not useful. On the other hand, if we have selected with only important information, the machine will find it easier. So it's a little funny because we cut features but the score actually increases.

# RandomizedSearchCV and Try Polynomial After Feature Selection

### RandomizedSearchCV

we're going to continue using RandomizedSearchCV, so we're not going to use GridSearch anymore and use a different one.

![image](https://user-images.githubusercontent.com/86812576/167064365-fb4a659a-6cb2-4501-ae67-c638f19b3123.png)

The one on the left is GridSearch, which we combine in the form of a table and it will search one by one.

for example in the case of SVM with two parameters, namely Gamma, and C (Penalty) with total 49 total trials. whereas on the other hand out of these 49 we pick maybe 10 at random, so no need to try all of them. Why is it like that? because for example a Gamma in GridSearch is bad, it will actually produce a bad model but it will still be run by GridSearch and it is not computational efficient. So we're just wasting time on a model.

![GS n RS](https://user-images.githubusercontent.com/86812576/167067042-f545776f-1e41-4848-9c4f-eec9cebd239c.png)


So why don't we take 10 at random, then we choose the best. Indeed there will be no guarantee of getting the best one, but computationally more efficient. We can even choose at intervals, meaning RandomizedSearchCV can reach areas that GridSearchCV cannot reach.

For the Random Forest Algorithm which has many parameters, it is more efficient to use RandomizedSearchCV. Because we immediately give a range and limit it to a certain trial, if not then use GridSearchCV and it will be able to reach thousands of combinations.

### Polynomial and RandomizedSearchCv

After we get meaningful information, then we add polynomials with RandomizedSearchCV

![Screenshot 2022-05-06 113707](https://user-images.githubusercontent.com/86812576/167068018-882f056f-837b-40b0-b930-8774ce19296e.png)

in this model, we actually just replace **gsp** with **rsp**, and add polynomials rest is the same as the previous model. And we have **n_iter = 50** which is trials.In this model the computation is very fast and even has been added with poly. 

