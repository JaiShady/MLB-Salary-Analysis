MLB Player Salary Analysis and Model Evaluation Report: https://docs.google.com/document/d/1AqgzDuwnJzf6v-bfcDD6GrND6SNL_wIIq_1fQeVxISk/edit?usp=sharing

1. Introduction

This report presents an exploratory data analysis (EDA) and predictive modeling study on Major League Baseball (MLB) player salaries. The analysis focuses on identifying the key factors influencing player compensation, examining statistical distributions, and building regression and classification models to predict player salaries. Both raw and standardized data were utilized to improve model interpretability and performance.

2. Exploratory Data Analysis (EDA)

2.1 Data Cleaning

Categorical columns League, Division, NewLeague, and Player were removed.

Missing values in the Salary column were excluded.

Continuous variables were analyzed for skewness, and distributions were visualized to identify outliers.

2.2 Distribution of Performance Metrics: Statistic	Distribution Characteristics	Interpretation

At Bats	Slightly right-skewed	Majority of players fall between 250–400 at bats; low values indicate bench players or injury absences
Hits	Right-skewed	Most players accumulate ~100 hits; a few elite players exceed 200 hits
Home Runs	Strong right skew	Maximum ~40 HR; few power hitters inflate the mean
Runs	Right skew	Typical range: 30–70; outliers >100 indicate top-of-lineup hitters
RBIs	Right skew	Most between 30–70; outliers above 100 correlate with power hitters
Walks	Right skew	Majority between 20–50; elite plate discipline reflected in >90 walks
Years	Left skew	More early-career players; long tail of veterans inflates mean
Career At Bats, Hits, HR, Runs, RBIs, Walks	Strong right skew	Career accumulation leads to extreme values for long-tenured players
Putouts and Assists	Highly positional and right-skewed	Catchers/first basemen accumulate most; outfielders and pitchers much less
Errors	Right skew	Most players have 3–12 errors
Salary	Heavily right skewed	Majority < $500k; few superstars inflate the mean, supporting a log transformation

2.3 Relationship to Log-Transformed Salary
Career offensive statistics (CRuns, CHits, CRBI) and career longevity (Years) have strong positive correlations with log_salary.

Defensive metrics (Assists, Errors, Putouts) show weak correlations.

Scatterplots confirm that log transformation of salary improves linearity and reduces skewness, revealing clearer relationships between performance and compensation.

3. Standardization and Feature Scaling

Standardization retained skewness patterns but allowed comparisons across features.

Regression coefficients now represent the effect per standard deviation, simplifying interpretation.

Key trends remained linear; slope values quantify the contribution of each performance metric to salary.

4. Regression Analysis
   
4.1 Feature Selection

Recursive Feature Elimination (RFE) and stepwise selection were used.

Selected features: CRuns, Hits, Years, PutOuts, Walks, AtBat, CWalks

These features consistently demonstrated strong predictive power for player salaries.

4.2 OLS Regression Results

Original Data: R² = 0.536, suggesting moderate explanatory power.

Standardized Data: Coefficients indicate relative impact of each feature:

CRuns, Hits, Years, Walks positively influence salary.

AtBat and CWalks had negative coefficients after adjustment.

Normalized Data: Improved R² = 0.713, reflecting stronger predictive capacity with scaled features.

Overall, regression analysis highlights that career performance and experience are the most significant determinants of MLB salaries.

5. Decision Tree Modeling
   
5.1 Model Complexity

Unrestricted depth produced near-perfect training accuracy, indicating overfitting.

Depth restriction to 2 caused underfitting (high bias), while depth of 3 balanced bias and variance.

A maximum depth of 3 was selected as optimal to maintain generalization while capturing important patterns.

5.2 Observations

Deeper trees capture complex interactions but risk overfitting.

Shallower trees simplify interpretation and generalize better to unseen data.

6. Classification Metrics: Precision, Recall, and F1 Score
   
6.1 Definitions

Precision:
Precision = True Positives/ (True Positives + False Positives)
Measures the proportion of predicted positive cases that are correctly identified.

Recall (Sensitivity):
Recall = True Positives/ (True Positives + False Negatives)
Measures the proportion of actual positive cases correctly identified.

F1 Score:
𝐹1= 2⋅ (Precision⋅Recall)/(Precision + Recall)
Represents the harmonic mean of precision and recall, balancing the tradeoff between false positives and false negatives.

6.2 Precision-Recall Tradeoff

Increasing the classification threshold improves precision but decreases recall.

Lowering the threshold increases recall but may reduce precision.

Optimal threshold selection depends on the application; for MLB player classification (e.g., predicting high salary players), the balance is key.

6.3 Interpretation

High F1 score: Indicates that the model achieves both high precision and high recall.

For example, tuning the model threshold to 0.85 improved precision, though recall decreased, demonstrating a classic tradeoff.

7. Model Performance

ROC AUC Score: 0.98, indicating excellent model discrimination.

Regression and tree-based models capture the majority of variance in log-transformed salary.

Offensive metrics and experience remain the dominant predictors, while defensive metrics have limited predictive impact.

8. Conclusion

Salary Determinants: Career offensive performance (hits, runs, RBIs) and years of experience are primary drivers of MLB player salaries.

Data Transformations: Log transformation of salary and standardization of features improved linearity, interpretability, and model performance.

Regression vs. Tree Models: Regression provides interpretable linear relationships; decision trees capture non-linear patterns but require careful depth tuning to avoid overfitting.

Classification Metrics: F1 score and ROC-AUC highlight model effectiveness, while precision-recall tradeoffs illustrate the importance of threshold selection for classification tasks.
