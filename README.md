# US Accidents Severity Prediction

## Milestone 1: Project Abstract

Traffic accidents pose a significant public health issue, with severe implications for safety and economic impact. This project aims to leverage machine learning techniques to predict accident severity using a comprehensive dataset of US accidents from 2016 to 2023, encompassing 7.7 million unique accident records across 49 states and featuring 46 rich attributes. The primary goal is to develop robust, high-accuracy classification models to predict accident severity, characterized by a severity score ranging from 1 (least impact) to 4 (most significant impact). This supervised classification task will employ various machine learning models, including logistic regression, support vector machine, random forest, gradient boosted decision tree, and multi-layer perceptron. The project will address several critical aspects: handling massive datasets, conducting exploratory data analysis (EDA), uncovering meaningful patterns within the data, and rigorously evaluating the performance of the classification models. The outcomes are expected to provide valuable insights into accident causation and contribute to the development of predictive tools for reducing accidents, optimizing traffic flow, and enhancing ETA prediction systems.

_Keywords: machine learning, accident prediction, supervised classification, exploratory data analysis, logistic regression, support vector machine, decision tree, random forest, naive bayes, multi-layer perceptron, US accidents dataset._

### Dataset

US Accidents (2016 - 2023) from [Kaggle](https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents/code)

- Moosavi, Sobhan, Mohammad Hossein Samavatian, Srinivasan Parthasarathy, and Rajiv Ramnath. “A Countrywide Traffic Accident Dataset.”, 2019.

- Moosavi, Sobhan, Mohammad Hossein Samavatian, Srinivasan Parthasarathy, Radu Teodorescu, and Rajiv Ramnath. "Accident Risk Prediction based on Heterogeneous Sparse Data: New Dataset and Insights." In proceedings of the 27th ACM SIGSPATIAL International Conference on Advances in Geographic Information Systems, ACM, 2019.

### Notebook

All code and instructions to download data and setup environment are in the [us-accidents-severity-prediction](us-accidents-severity-prediction.ipynb) notebook.

## Milestone 2: Data Exploration & Initial Preprocessing

Based on the data exploration phase, we plan on following the below steps for the data preprocessing phase:

- Handling missing values: During our data exploration phase, we noticed certain variables (like `End_Lat`, `End_Lng`, `Wind_Chill`, and more). We have decided to use other variables that are strongly correlated with these variables instead. This allows us to retain as much information for the classification task without having to lose large amounts of data.

- Feature engineering: We plan on parsing `Weather_Timestamp` variable (which is a string) into multiple integer type variables (like year, month, day, hour, and minute) so they are easy to process in the later stages of the pipeline. Additionally, we would one-hot encode variables like `Weather_Condition` and `Wind_Direction` to potentially improve model training.

- Feature selection: Since the dataset has a large number of columns, it is important for us to choose our feature columns wisely. The EDA process gave us a lot of insight into which feature columns we want to particularly use and which ones we want to drop. The correlation matrix gave us insights into redundant variables that do can easily be ignored in the training process.
