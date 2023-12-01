# House Prices Prediction Project

## Dataset

The dataset consists of residential homes in Ames, Iowa, with 79 explanatory variables. It is divided into two CSV files: `train` and `test`. In the test dataset, house prices are left blank, and the goal is to estimate these values.

You can access the dataset and competition page on Kaggle [here](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/overview/evaluation).

## Requirements

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
