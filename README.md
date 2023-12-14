# Overview

I undertook this project to enhance my skills as a software engineer and to deepen my understanding of data analysis. The dataset used for analysis comprises flight data from US flights in January 2019 and 2020, obtained from [Kaggle](https://www.kaggle.com/divyansh22/flight-delay-prediction).

The primary goal of this project is to perform an exploratory data analysis, create predictive models to anticipate flight delays, and answer specific questions related to the dataset.

For a comprehensive demonstration, please refer to the [Software Demo Video](http://youtube.link.goes.here), which provides a 4-5 minute walkthrough of the data, analysis, and code execution.

# Data Analysis Results

1. **Can we make a model to predict flight delays?**
   - Yes, using various models such as Logistic Regression **score: 90**.

2. **Are we more likely to experience delays at specific dates?**
   - The analysis shows variations in delays based on specific dates.
   - For example, the 18th has the highest percentage of delays at **25%**, while the 9th has the lowest at **10%**.

3. **Are we more likely to experience delays at specific weekdays?**
   - Weekdays do exhibit some variations in delay percentages.
   - Tuesdays have the lowest percentage, with **12.3%** of flights getting delayed, while Thursdays have the highest at **17.8%**.

4. **Is there a difference in the amount of delays based on the carrier?**
   - Different carriers show significant differences in delay rates.
   - The airline with the highest delay rate has **26.5%** of all flights delayed, while the airline with the lowest delay rate has **11.4%**.

5. **Is there a difference in the amount of delays based on the departure airport?**
   - The size and type of airports contribute to variations in delay rates.
   - The nine airports with the most delays have between **30% and 40%** of their flights delayed.

6. **Is there a difference in the amount of delays based on the arrival airport?**
   - Similar to departure airports, arrival airports show variations in delay percentages.
   - The airport with the highest percentage of delays is Muskegon airport (mkg) in Michigan, where **47.5%** of all arrivals are delayed.
# Model Training Outcomes
**Confusion Matrix**
![Confusion Matrix](/img/confusion_matrix.png)
**ROC curve and Precision-recall curve**
![ROC curve and Precision-recall curve](/img/curves.png)
**ROC AUC SCORE**
![ROC AUC SCORE](/img/score.png)

# Development Environment

The software was developed using Python and Visual Studio Code. The key tools and libraries include:
- Visual Studio Code for code development
- Pandas for data manipulation
- NumPy for numerical operations
- Seaborn and Matplotlib for data visualization
- Scikit-learn for machine learning models
- Imbalanced-learn (imblearn) for handling imbalanced datasets
- SMOTENC for oversampling imbalanced data
- OrdinalEncoder, OneHotEncoder, and StandardScaler for data preprocessing
- GridSearchCV for hyperparameter tuning
- LogisticRegression for the predictive model
- TruncatedSVD for dimensionality reduction

# Useful Websites

* [Kaggle - Flight Delay Prediction Dataset](https://www.kaggle.com/divyansh22/flight-delay-prediction)
* [Pandas Documentation](https://pandas.pydata.org/pandas-docs/stable/index.html)

* [DataCamp - Online Learning Platform for Data Science](https://www.datacamp.com/)
* [YouTube - Corey Schafer's Python Pandas Tutorial](https://www.youtube.com/playlist?list=PL-osiE80TeTtWZHE__I842fI8ZQzPHIyr) 


# Future Work

* Fine-tune model parameters for improved performance.
* Explore additional features for predictive modeling.
* Enhance data cleaning and preprocessing steps.
* Expand analysis to cover more months and years for a comprehensive understanding.
