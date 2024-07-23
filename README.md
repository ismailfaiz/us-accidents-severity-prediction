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
<<<<<<< HEAD

## Milestone 3: Data Preprocessing

- Feature Selection: Columns removed: `ID`, `Source`, `End_Lat`, `End_Lng`, `Timezone`, `Airport_Code`, `Weather_Timestamp`, `Street`, `County`, `State`, `Zipcode`, `Country`, `Turning_Loop`, `Civil_Twilight`, `Nautical_Twilight`, `Astronomical_Twilight`, `Description`, `Wind_Chill`. These columns were removed due to redundancy and irrelevance (e.g., `ID`, `Source`, `Street`, `County`) or high correlation with other features (e.g., `End_Lat`, `End_Lng`, `Wind_Chill`). Text descriptions were excluded to avoid the complexity of natural language processing tasks.

- Feature Engineering: Time-related features extracted: `Year`, `Month`, `Day`, `Hour`, `Minute`, `Day of Week` from `Start_Time`. These features help capture temporal patterns like rush hour traffic or seasonal variations, which may influence accident severity.

- One-Hot Encoding: Simplified and encoded features: `Wind_Direction` and `Weather_Condition`. Grouping similar wind directions and weather conditions makes the data more manageable. Binary features for weather conditions capture patterns without introducing too much complexity. `Day` is encoded as 0, and `Night` is encoded as 1. This simplifies the feature `Sunrise_Sunset` while preserving its impact on accident severity.

- Handling Missing Values: Numerical columns filled with median values, and categorical columns filled with mode (most frequent) values. Median imputation is less affected by outliers, and mode imputation ensures the most common category is used, preserving data distribution.

- Feature Scaling: Z-score method identifies and removes data points with Z-scores greater than a threshold (commonly 3). `StandardScaler` normalizes numerical features (except `Year`). Outlier removal ensures extreme values do not skew the scaling process. Normalization puts all features on a similar scale, improving performance and convergence of machine learning algorithms.

- Final Dataset Preparation: Target variable `Severity` is treated as a categorical variable. The preprocessed features are selected as input (`X`), and `Severity` as the target (`y`) for model training. This ensures that the final dataset is ready for model training with all necessary preprocessing steps applied.

### Model Training

The logistic regression model achieves 74% accuracy on both training and test sets, with similar F1 scores (0.36), indicating no overfitting. However, the model struggles with minority classes (1 and 4), suggesting underfitting for these classes. The model seems to struggle with **imbalanced data**, performing well on the majority class but poorly on minority classes. This suggests potential overfitting to the majority class. On the fitting graph (model complexiy vs. error), this model likely sits in the high bias region (low complexity and high error), showing potential for improvement with increased complexity.

**Next models to consider:**
1. Random Forest Classifier: Can handle non-linear relationships and class imbalance.
2. Gradient Boosting Classifier (e.g., XGBoost): Performs well on imbalanced datasets.
3. Support Vector Machine (SVM): Can capture non-linear relationships with kernel tricks.

These models are chosen to address the limitations of logistic regression in handling non-linear relationships and class imbalance.

**Next Steps**
- Address class imbalance using techniques like SMOTE or class weights.
- Implement non-linear models such as Random Forest or Gradient Boosting to capture complex relationships and handle imbalanced data better.
- Increase max_iter parameter to allow for convergence, as the model consistently hit the maximum iterations.
- Feature engineering and selection to improve model performance.
  
These steps aim to move the model towards the "sweet spot" on the fitting graph, balancing bias and variance while addressing the challenges of imbalanced data and convergence issues.
=======
>>>>>>> 8ed420e4832fa6a39b34e9ca57dff245724ecbcf
