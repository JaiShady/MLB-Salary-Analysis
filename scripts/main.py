


import pandas as pd
import matplotlib.pyplot as plt
from pandas import set_option
from pandas import read_csv
from sklearn.preprocessing import StandardScaler, Normalizer
from numpy import set_printoptions
import seaborn as sns
from pandas.plotting import scatter_matrix
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
import numpy as np
import statsmodels.api as sm
from sklearn.tree import DecisionTreeRegressor, plot_tree


filePath = 'data/'
filename = 'Baseball_salary.csv'
data1 = read_csv(filePath+filename)



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
from pandas.plotting import scatter_matrix
from pandas import set_option

filePath = 'C:/Users/jaina/Downloads/'
filename = 'Baseball_salary.csv'
df = pd.read_csv(filePath + filename)

df = df.drop(columns=['League', 'Player', 'Division', 'NewLeague'], errors='ignore')
df = df.dropna(subset=['Salary'])
df = df[df['Salary'] > 0]


print(df.head())
set_option('display.width', 100)
set_option('display.precision', 2)
print(df.describe())

# log salary
df['log_salary'] = np.log(df['Salary'])
print(df[['Salary', 'log_salary']].head())

# only want numeric features
numeric_df = df.select_dtypes(include=[np.number])
X = numeric_df.drop(columns=['Salary', 'log_salary'])
y = numeric_df['log_salary']
Xnames = X.columns

# standardize
scaler = StandardScaler()
X_std = pd.DataFrame(
    scaler.fit_transform(X),
    columns=X.columns,
    index=X.index
)
data_std = X_std.copy()
data_std['log_salary'] = y
print(data_std.describe())

# featyre selection
def perform_rfe(X, y, n_features=6):
    lr = LinearRegression()
    rfe = RFE(lr, n_features_to_select=n_features)
    rfe.fit(X, y)
    selected_features = X.columns[rfe.support_]
    print("Selected features by RFE:", selected_features.tolist())
    return selected_features

def stepwise_selection(X, y,
                       initial_list=[],
                       threshold_in=0.05,
                       threshold_out=0.10,
                       verbose=True):
    included = list(initial_list)
    while True:
        changed = False
        excluded = list(set(X.columns) - set(included))
        new_pval = pd.Series(index=excluded, dtype=float)
        for new_column in excluded:
            model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included + [new_column]]))).fit()
            new_pval[new_column] = model.pvalues[new_column]
        best_pval = new_pval.min()
        if best_pval < threshold_in:
            best_feature = new_pval.idxmin()
            included.append(best_feature)
            changed = True
            if verbose:
                print(f'Add  {best_feature:30} with p-value {best_pval:.6f}')
        model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included]))).fit()
        pvalues = model.pvalues.iloc[1:]
        worst_pval = pvalues.max()
        if worst_pval > threshold_out:
            worst_feature = pvalues.idxmax()
            included.remove(worst_feature)
            changed = True
            if verbose:
                print(f'Drop {worst_feature:30} with p-value {worst_pval:.6f}')
        if not changed:
            break
    return included

def perform_stepwise(X, y):
    final_features = stepwise_selection(X, y)
    print("Selected features by Stepwise:", final_features)
    final_model = sm.OLS(y, sm.add_constant(X[final_features])).fit()
    print(final_model.summary())
    return final_features



#raw data
df.hist(figsize=(10, 8))
plt.suptitle("Original Data Histograms")
plt.show()

plt.figure(figsize=(8, 6))
sns.heatmap(df.corr(), square=True, cmap='coolwarm', annot=False)
plt.title("Correlation Heatmap - Original Data")
plt.show()

scatter_matrix(df, figsize=(10, 8))
plt.suptitle("Scatter Matrix - Original Data")
plt.show()

# stanadrd data
data_std.hist(figsize=(10, 8))
plt.suptitle("Standardized Data Histograms")
plt.show()

plt.figure(figsize=(8, 6))
sns.heatmap(data_std.corr(), square=True, cmap='coolwarm', annot=False)
plt.title("Correlation Heatmap - Standardized Data")
plt.show()

scatter_matrix(data_std, figsize=(10, 8))
plt.suptitle("Scatter Matrix - Standardized Data")
plt.show()

# feature select
print("\n Feature Selection-Standardized Data")
rfe_features = perform_rfe(pd.DataFrame(X_std, columns=Xnames), y)
stepwise_features = perform_stepwise(pd.DataFrame(X_std, columns=Xnames), y)

# RFE DATA
selected_df = data_std[rfe_features.tolist() + ['log_salary']]

plt.figure(figsize=(8, 6))
sns.heatmap(selected_df.corr(), annot=True, cmap="coolwarm", square=True)
plt.title("Correlation Heatmap - Selected RFE Features")
plt.show()

scatter_matrix(selected_df, figsize=(8, 6))
plt.suptitle("Scatter Matrix - RFE Selected Features")
plt.show()

X_tree = selected_df.drop(columns=['log_salary'])
y_tree = selected_df['log_salary']



tree_reg = DecisionTreeRegressor()
tree_reg.fit(X_tree, y_tree)
plt.figure(figsize=(20, 10))
plot_tree(
    tree_reg,
    feature_names=X_tree.columns,
    filled=True,
    rounded=True,
    fontsize=12
)
print("Tree depth:", tree_reg.get_depth())
print("Number of leaves:", tree_reg.get_n_leaves())
plt.title("Decision Tree Regressor - Default Settings")
plt.show()


tree_two = DecisionTreeRegressor(max_depth=2)
tree_two.fit(X_tree, y_tree)
plt.figure(figsize=(20, 10))
plot_tree(
    tree_two,
    feature_names=X_tree.columns,
    filled=True,
    rounded=True,
    fontsize=12
)
print("Tree depth:", tree_two.get_depth())
print("Number of leaves:", tree_two.get_n_leaves())
plt.title("Decision Tree Regressor - depth 2")
plt.show()

tree_three = DecisionTreeRegressor(max_depth=3)
tree_three.fit(X_tree, y_tree)
plt.figure(figsize=(20, 10))
plot_tree(
    tree_three,
    feature_names=X_tree.columns,
    filled=True,
    rounded=True,
    fontsize=12
)
print("Tree depth:", tree_three.get_depth())
print("Number of leaves:", tree_three.get_n_leaves())
plt.title("Decision Tree Regressor - depth 3")
plt.show()
