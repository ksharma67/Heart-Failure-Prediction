# Heart-Failure-Prediction

This problem is a typical Classification Machine Learning task. Building various classifiers by using the following Machine Learning models:
* Logistic Regression (LR)
* Decision Tree (DT)
* Random Forest (RF)
* XGBoost (XGB)

1) Performing Explanatory Data Analysis (EDA) / indicating how features correlate among themselves, with emphasis to the target/label one.

2) Applying Machine Learning Modeling on the dataset using all the above 4 algorithms. Tuning (hyper-parameter tuning) each model by calling the GridSearchCV method. Indicating which combination of Hyperparameters produces the best result.
Note: Use accuracy and AUC-ROC metrics when evaluating your models.

3) Performing Machine Learning Interpretability/Explanability tasks as follows:

* Using the 'eli5' library to interpret the "white box" model of Logistic Regression. Applying 'eli5' to visualize the weights associated to each feature.
Using 'eli5' to explain specific predictions, pick a row in the test data with negative label and one with positive.

* Using the 'eli5' library to interpret the "white box" model of Decision Tree. Applying 'eli5' to list the feature importance ordered by the highest value.
Geting an explanation for a given prediction, one positive and one negative. This will calculate the contribution of each feature in the prediction. The explanation for a single prediction is calculated by following the decision path in the tree, and adding up contribution of each feature from each node crossed into the overall probability predicted.
eli5 can also be used to explain black box models, but we will use LIME and SHAP for our two last models instead.

* Using LIME to explain both the Random Forest and the XGBoost models.
Creating a LIME explainer by using the LimeTabularExplainer method, the main explainer to use for tabular data.
LIME fits a linear model on a local shuffled dataset. Accessing the coefficients, the intercept
and the R2 of the linear model, for both model interpretability.
Note: If R2 is low, the linear model that LIME fitted isn't a great approximation to your model, which means you should not rely too much on the explanation it provides.

* Using SHAP library to interpret the XGBoost model. Specifically, call the TreeExplainer method of SHAP, TreeExplainer is optimized for tree based models.
Visualizing explanations, one for positive and one for negative, using the ‘force_plot’ function.
Note: You need to establish a ‘base value’ in order to be used by ‘force_plot’. The explainer.expected_value is the ‘base value’.
Create the feature importance plot by calling SHAP’s ‘summary_plot’ function, for each class/label.

4) Predicting observations, one for positive and one for negative label, by using all four (4) models and indicate which one gives the better prediction.
Providing output for showing the accuracy of each model as follows:
False/True label: 0/1 (or 0/1 depending how you define labels)
* LR: [prob_T prob_F]
* DT: [prob_T prob_F]
* RF: [prob_T prob_F]
* XGB: [prob_T prob_F]

The above calculations are derived by calling the predict_proba method. Note: predict_proba(X): Predict class probabilities for X.
