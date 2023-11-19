#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import chi2_contingency
from scipy.stats import kruskal
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
import warnings
import time
file_path = 'marketing_trn_class_labels.csv'
df = pd.read_csv(file_path)
print(df.head())


# In[2]:


total_elements = df.size
zero_count = (df == 0).sum().sum()
percentage_zeros = (zero_count / total_elements) * 100
print(f"Percentage of failure of campaign: {percentage_zeros:.2f}%")


# In[3]:


file_path1 = 'marketing_trn_data.csv'
df1 = pd.read_csv(file_path1)


# In[4]:


df1.columns


# In[5]:


df1.rename(columns={' Income ': 'Income'}, inplace=True)


# In[6]:


data = df1.copy()


# In[7]:


existing_columns = df.columns.tolist()
new_columns = ['label', 'target'] + existing_columns[:-2]  # Exclude the last two columns
new_df = pd.DataFrame(df.values, columns=new_columns)
df = df.iloc[1:]
new_df.reset_index(drop=True, inplace=True)
print("Original DataFrame:")
print(df)
print("\nModified DataFrame:")
print(new_df)


# In[8]:


existing_columns = new_df.columns.tolist()
new_row = ['d1', '0']
df2 = pd.concat([pd.DataFrame([new_row], columns=existing_columns), new_df], ignore_index=True)
print("Modified DataFrame:")
print(df2)


# In[9]:


if 'target' in df2.columns:
    target_column = df2['target']
    data['target'] = target_column


# In[10]:


print(data)


# In[11]:


data.info()


# In[12]:


num_duplicates = df1.duplicated().sum()
print(f"Number of duplicate rows in the dataset: {num_duplicates}")


# In[13]:


contingency_table_data = pd.crosstab(index=data['Marital_Status'], columns='Count in data')
print(contingency_table_data)


# In[14]:


data.rename(columns={' Income ': 'Income'}, inplace=True)


# In[15]:


data['Income'] = data['Income'].str.replace('[\$,]', '', regex=True) 
data['Income'] = data['Income'].str.strip()  
data['Income'] = data['Income'].replace('', '0')  
data['Income'] = data['Income'].astype(float)  


# In[16]:


data_filled = data.copy()
data_filled['Income'] = data_filled.groupby('Education')['Income'].transform(lambda x: x.fillna(x.mean()))
data=data_filled


# In[17]:


data.isnull().sum()


# In[18]:


data.info()


# In[19]:


data = data.drop(columns='Dt_Customer')


# In[20]:


print(data["Marital_Status"].value_counts())
print("*"*25)
print(data["Education"].value_counts())


# In[21]:


data['target'] = data['target'].astype(int)
print("Modified DataFrame:")
print(data)


# In[22]:


print(data["target"].value_counts())


# In[23]:


data.index.shape


# In[24]:


plt.figure(figsize=(8, 6))
ax = sns.countplot(x="target", data=data)
total = len(data["target"]) * 1.
for p in ax.patches:
    ax.annotate('{:.1f}%'.format(100 * p.get_height() / total), (p.get_x() + p.get_width() / 2., p.get_height() + 5),
                ha='center', va='bottom')
ax.yaxis.set_ticks(np.linspace(0, total, 11))
ax.set_yticklabels(map('{:.1f}%'.format, 100 * ax.yaxis.get_majorticklocs() / total))
plt.xticks(rotation=40, ha="right")
plt.show()


# In[25]:


def countplot(label, dataset):
    plt.figure(figsize=(15, 10))
    Y = dataset[label]
    total = len(Y) * 1.
    ax = sns.countplot(x=label, data=dataset)
    for p in ax.patches:
        ax.annotate('{:.1f}%'.format(100 * p.get_height() / total), (p.get_x() + 0.1, p.get_height() + 5))
    ax.yaxis.set_ticks(np.linspace(0, total, 11))
    ax.set_yticklabels(map('{:.1f}%'.format, 100 * ax.yaxis.get_majorticklocs() / total))
    plt.xticks(rotation=40, ha="right")
    plt.show()


# In[26]:


get_ipython().run_line_magic('matplotlib', 'inline')

def countplot_withY(label, dataset):
    plt.figure(figsize=(20, 10))
    Y = dataset[label]
    total = len(Y) * 1.
    ax = sns.countplot(x=label, data=dataset, hue="target")
    for p in ax.patches:
        ax.annotate('{:.1f}%'.format(100 * p.get_height() / total), (p.get_x() + 0.1, p.get_height() + 5))
    ax.yaxis.set_ticks(np.linspace(0, total, 11))
    ax.set_yticklabels(map('{:.1f}%'.format, 100 * ax.yaxis.get_majorticklocs() / total))
    plt.xticks(rotation=40, ha="right")
    plt.show()


# In[27]:


countplot("Marital_Status", data)


# In[28]:


countplot_withY('Marital_Status',data)


# In[29]:


countplot("Education", data)


# In[30]:


countplot_withY('Education',data)


# In[31]:


# Function to calculate Cramer's V
def cramers_v(confusion_matrix):
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
    rcorr = r - ((r-1)**2)/(n-1)
    kcorr = k - ((k-1)**2)/(n-1)
    return np.sqrt(phi2corr / min((kcorr-1), (rcorr-1)))
categorical_variables = ['Education', 'Marital_Status']
for cat_var in categorical_variables:
    contingency_table = pd.crosstab(data[cat_var], data['target'])
    print(f"\nContingency Table for {cat_var} vs. target:")
    print(contingency_table)
    chi2, p, _, _ = chi2_contingency(contingency_table)
    print(f"Chi-square statistic: {chi2}")
    print(f"P-value: {p}")
    # Calculate Cramer's V using the custom function
    cramer_v_value = cramers_v(contingency_table.values)
    print(f"Cramer's V for {cat_var} vs. target:", cramer_v_value)


# In[32]:


get_ipython().run_line_magic('matplotlib', 'inline')
sns.boxplot(data=data, x="target", y="NumStorePurchases")
plt.show()


# In[33]:


plt.figure(figsize=(10,8))
sns.distplot(data["NumStorePurchases"])


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
sns.boxplot(data=data, x="target", y="NumDealsPurchases")
plt.show()


# In[ ]:


plt.figure(figsize=(10,8))
sns.distplot(data["NumDealsPurchases"])


# In[34]:


get_ipython().run_line_magic('matplotlib', 'inline')
sns.boxplot(data=data, x="target", y="NumWebPurchases")
plt.show()


# In[35]:


plt.figure(figsize=(10,8))
sns.distplot(data["NumWebPurchases"])


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
sns.boxplot(data=data, x="target", y="NumCatalogPurchases")
plt.show()


# In[36]:


plt.figure(figsize=(10,8))
sns.distplot(data["NumCatalogPurchases"])


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
sns.boxplot(data=data, x="target", y="NumWebVisitsMonth")
plt.show()


# In[37]:


plt.figure(figsize=(10,8))
sns.distplot(data["NumWebVisitsMonth"])


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
sns.boxplot(data=data, x="target", y="Year_Birth")
plt.show()


# In[38]:


plt.figure(figsize=(10,8))
sns.distplot(data["Year_Birth"])


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
sns.boxplot(data=data, x="target", y="Income")
plt.show()


# In[39]:


plt.figure(figsize=(10,8))
sns.distplot(data["Income"])


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
sns.boxplot(data=data, x="target", y="Recency")
plt.show()


# In[40]:


plt.figure(figsize=(10,8))
sns.distplot(data["Recency"])


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
sns.boxplot(data=data, x="target", y="MntGoldProds")
plt.show()


# In[41]:


plt.figure(figsize=(10,8))
sns.distplot(data["MntGoldProds"])


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
sns.boxplot(data=data, x="target", y="MntWines")
plt.show()


# In[42]:


# plt.figure(figsize=(10,8))
sns.distplot(data["MntWines"])


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
sns.boxplot(data=data, x="target", y="MntFruits")
plt.show()


# In[43]:


plt.figure(figsize=(10,8))
sns.distplot(data["MntFruits"])


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
sns.boxplot(data=data, x="target", y="MntFishProducts")
plt.show()


# In[44]:


plt.figure(figsize=(10,8))
sns.distplot(data["MntFishProducts"])


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
sns.boxplot(data=data, x="target", y="MntSweetProducts")
plt.show()


# In[45]:


plt.figure(figsize=(10,8))
sns.distplot(data["MntSweetProducts"])


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
sns.boxplot(data=data, x="target", y="MntMeatProducts")
plt.show()


# In[46]:


plt.figure(figsize=(10,8))
sns.distplot(data["MntMeatProducts"])


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import seaborn as sns

selected_variables = ['Year_Birth', 'Income', 'Kidhome', 'Teenhome', 'Recency',
                      'MntGoldProds', 'MntWines', 'MntFruits', 'MntFishProducts',
                      'MntSweetProducts', 'MntMeatProducts', 'NumDealsPurchases',
                      'NumWebPurchases', 'AcceptedCmp3', 'NumCatalogPurchases',
                      'NumStorePurchases', 'NumWebVisitsMonth', 'AcceptedCmp2',
                      'AcceptedCmp4', 'AcceptedCmp5', 'AcceptedCmp1', 'Complain']

selected_data = data[selected_variables]
corr = selected_data.corr()

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(10, 12))

# Create a diverging color palette
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with custom formatting
_ = sns.heatmap(corr, cmap="YlGn", square=True, ax=ax, annot=True, fmt=".2f", annot_kws={"size": 8}, linewidth=0.1)

plt.title("Pearson correlation of Selected Variables", y=1.05, size=15)
plt.show()


# In[47]:


numeric_variables = ['Year_Birth', 'Income', 'Kidhome', 'Teenhome', 'Recency',
                      'MntGoldProds', 'MntWines', 'MntFruits', 'MntFishProducts',
                      'MntSweetProducts', 'MntMeatProducts', 'NumDealsPurchases',
                      'NumWebPurchases', 'AcceptedCmp3', 'NumCatalogPurchases',
                      'NumStorePurchases', 'NumWebVisitsMonth', 'AcceptedCmp2',
                      'AcceptedCmp4', 'AcceptedCmp5', 'AcceptedCmp1', 'Complain']
kruskal_results = {}
for numeric_var in numeric_variables:
    h_statistic, p_value = kruskal(*[data[data['target'] == i][numeric_var] for i in data['target'].unique()])
    print(f"\nKruskal-Wallis H test for {numeric_var} vs. target:")
    print(f"H Statistic: {h_statistic}")
    print(f"P-value: {p_value}")
    kruskal_results[numeric_var] = {'H Statistic': h_statistic, 'P-value': p_value}
    if p_value < 0.05: 
        print("Reject NULL hypothesis - Significant differences exist between groups.")
        print("The continuous variable has some impact over the target variable.")
    else:
        print("Fail to reject NULL hypothesis - No significant differences between groups.")
        print("The continuous variable does not have a significant impact over the target variable.")


# In[48]:


#Columns are removed as their significance wasn't there over the target variable as per the kruskal-wallis H test


# In[49]:


columns_to_drop = ['Year_Birth', 'NumDealsPurchases', 'NumWebVisitsMonth', 'Complain']
data = data.drop(columns=columns_to_drop)


# In[50]:


import pandas as pd
encoded_df = pd.get_dummies(data, columns=['Education', 'Marital_Status'],dtype=int)
cleaned_df = encoded_df.dropna()
print(encoded_df.head())


# In[51]:


encoded_df.info()


# In[52]:


data=encoded_df


# In[53]:


data.to_csv('train_data.csv', index=False)


# In[54]:


data.info()


# In[55]:


file_path = 'marketing_tst_data.csv'
data = pd.read_csv(file_path)


# In[56]:


data.rename(columns={' Income ': 'Income'}, inplace=True)
data['Income'] = data['Income'].str.replace('[\$,]', '', regex=True) 
data['Income'] = data['Income'].str.strip()  
data['Income'] = data['Income'].replace('', '0')  
data['Income'] = data['Income'].astype(float)  
data_filled = data.copy()
data_filled['Income'] = data_filled.groupby('Education')['Income'].transform(lambda x: x.fillna(x.mean()))
data=data_filled
data = data.drop(columns='Dt_Customer')
columns_to_drop = ['Year_Birth', 'NumDealsPurchases', 'NumWebVisitsMonth', 'Complain']
data = data.drop(columns=columns_to_drop)
encoded_df = pd.get_dummies(data, columns=['Education', 'Marital_Status'],dtype=int)
cleaned_df = encoded_df.dropna()
data=encoded_df
data['Marital_Status_Absurd'] = 0
data['Marital_Status_YOLO'] = 0
data.info()


# In[57]:


data.to_csv('test_data.csv', index=False)


# In[58]:


#code will take long time to run and will generate warning as well in between


# In[59]:


warnings.filterwarnings("ignore", category=UserWarning)

# Load and split the data
dataset = pd.read_csv('train_data.csv')
Train_Data = dataset.drop('target', axis=1)
Traindata_classlabels = dataset['target']
Train_Data_train, Train_Data_test, Traindata_classlabels_train, Traindata_classlabels_test = train_test_split(
    Train_Data, Traindata_classlabels, test_size=0.3, random_state=53
)

def train_evaluate_model(clf, param_grid, X_train, y_train, X_test, y_test):
    # Check if the classifier has the 'class_weight' parameter
    has_class_weight = 'class_weight' in clf().get_params()
 
    # Use GridSearchCV for hyperparameter tuning with parallel processing
    if has_class_weight:
        # Use balanced class weights if available
        class_weights = {"class_weight": "balanced"}
    else:
        class_weights = {}

    grid_search = GridSearchCV(
        estimator=clf(**class_weights),
        param_grid=param_grid,
        scoring="f1_macro",
        cv=5,
        n_jobs=-1  # Use all available CPU cores
    )

    start_time = time.time()

    grid_search.fit(X_train, y_train)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time} seconds")

    # Print best parameters
    print(f"Best Parameters: {grid_search.best_params_}")

    # Print best score
    print(f"Best {clf.__name__} Score: {grid_search.best_score_}")

    # Print model score on the test set
    print(f"{clf.__name__} score = {grid_search.best_estimator_.score(X_test, y_test)}")

    # Make predictions on the test set
    pred = grid_search.best_estimator_.predict(X_test)

    # Evaluate predictions
    pred_acc = accuracy_score(y_test, pred)
    pred_f = f1_score(y_test, pred, average="macro")
    pred_p = precision_score(y_test, pred, average="macro")
    pred_r = recall_score(y_test, pred, average="macro")

    print(f"Prediction accuracy = {pred_acc}")
    print(f"Prediction f measure = {pred_f}")
    print(f"Prediction Precision = {pred_p}")
    print(f"Prediction Recall = {pred_r}")
    print(confusion_matrix(pred, y_test))

# K Nearest Neighbours
knn_param_grid = {
    "n_neighbors": [5, 7, 9, 11, 13, 15, 16, 17, 18, 19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40],
    "weights": ["uniform", "distance"],
    "metric": ["minkowski", "euclidean", "manhattan"],
}
train_evaluate_model(KNeighborsClassifier, knn_param_grid, Train_Data_train, Traindata_classlabels_train, Train_Data_test, Traindata_classlabels_test)

# Decision Tree
dt_param_grid = {
    "criterion": ["gini", "entropy"],
    "max_features": ["auto", "sqrt", "log2", None],
    "max_depth": [15, 30, 45, 60],
    "ccp_alpha": [0.009, 0.005, 0.05],
}
train_evaluate_model(DecisionTreeClassifier, dt_param_grid, Train_Data_train, Traindata_classlabels_train, Train_Data_test, Traindata_classlabels_test)

# Gaussian Naive Bayes
train_evaluate_model(GaussianNB,{}, Train_Data_train, Traindata_classlabels_train, Train_Data_test, Traindata_classlabels_test)

# Logistic Regression
lr_param_grid = {
    "C": np.linspace(start=0.1, stop=10, num=100),
    "penalty": ["l1", "l2", "elasticnet"],
    "solver": ["newton-cg", "lbfgs", "liblinear"],
}
train_evaluate_model(LogisticRegression, lr_param_grid, Train_Data_train, Traindata_classlabels_train, Train_Data_test, Traindata_classlabels_test)

# Random Forest Classifier
rf_param_grid = {
    "criterion": ["entropy", "gini"],
    "max_features": ["auto", "sqrt", "log2", None],
    "n_estimators": [int(x) for x in np.linspace(start=200, stop=300, num=100)],
    "max_depth": [10, 20, 30, 50, 100, 200],
}
train_evaluate_model(RandomForestClassifier, rf_param_grid, Train_Data_train, Traindata_classlabels_train, Train_Data_test, Traindata_classlabels_test)


# Support Vector Machine
svm_param_grid = {
    "C": np.logspace(-2, 7, num=25, base=2),
    "gamma": [1, 0.1, 0.01, 0.001],
    "kernel": ["linear", "rbf", "polynomial", "sigmoid"],
}
train_evaluate_model(svm.SVC, svm_param_grid, Train_Data_train, Traindata_classlabels_train, Train_Data_test, Traindata_classlabels_test)



# In[ ]:


Test_Data=pd.read_csv('test_data.csv')
new_column_order = ['Income', 'Kidhome', 'Teenhome', 'Recency', 'MntGoldProds', 'MntWines',
       'MntFruits', 'MntFishProducts', 'MntSweetProducts', 'MntMeatProducts',
       'NumWebPurchases', 'AcceptedCmp3', 'NumCatalogPurchases',
       'NumStorePurchases', 'AcceptedCmp2', 'AcceptedCmp4', 'AcceptedCmp5',
       'AcceptedCmp1', 'Education_2n Cycle', 'Education_Basic',
       'Education_Graduation', 'Education_Master', 'Education_PhD',
       'Marital_Status_Absurd',
       'Marital_Status_Alone', 'Marital_Status_Divorced',
       'Marital_Status_Married', 'Marital_Status_Single',
       'Marital_Status_Together', 'Marital_Status_Widow', 'Marital_Status_YOLO']

# Reorder columns in the DataFrame
Test_Data = Test_Data[new_column_order]


# In[ ]:


""""Random Forest has the highest f value as compared to other models used in this project. So I am using this model to predict the target values of the test data"""
clf=RandomForestClassifier(criterion='entropy',max_depth=10, n_estimators=283)
clf.fit(Train_Data_train,Traindata_classlabels_train)
predict = clf.predict(Test_Data)
predict


# In[ ]:




