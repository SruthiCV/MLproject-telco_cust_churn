# Data Exploration

#exploratory data analysis of the dataframe churn into which we have read our dataset

churn.shape
#(7043, 21) - 7043 rows and 21 columns including target variable

#This shows the unique values in each column


def rstr(df): return df.apply(lambda x: [x.unique()])
print(rstr(churn))

#Delete columns ‘customerID’ and ‘PhoneService’ from dataframe ‘churn’, because customer ID is not useful for model prediction, and the information in variable ‘PhoneService’ is included inside variable ‘MultipleLines’


churn=churn.drop(['customerID','PhoneService'],axis=1)

#Check if the dataframe still includes any null values or not


churn.isnull().any()

#Since we do not have any NULL values, we proceed with some plotting


%matplotlib inline


churn.hist()


![Hist1](https://user-images.githubusercontent.com/38309595/122701856-7c27b500-d291-11eb-9711-298c047b434b.PNG)

The histogram shows that the highest number of customers have low monthly and hence total charges. A large number of customers have short tenure (new customers) and a slighly 
lesser number have a long tenure (old customers)

Now, we will proceed to plot the barplots of nominal features

![Barplots](https://user-images.githubusercontent.com/38309595/122846892-22cf8c80-d34a-11eb-94f4-0cb0b6d044dd.PNG)

Observations -
Customer gender distribution is uniform.
Most customers are not senior citizens.
Distribution of customers with/ without partners is relatively uniform.
Most customers have no dependents.
Few customers have no phone service. Among those who do, the number of customers who have multiple lines is almost same as those don't.
A large number of customers have Fibre Optic, compared to DSL. A smaller number of customers have no internet service.

![Barplots1](https://user-images.githubusercontent.com/38309595/122847041-704bf980-d34a-11eb-82ef-d5211d6a39f4.PNG)

Observations -
The number of customers who have Online Security is almost same as those who have no internet. A large number of customers have no Online Security.
The number of customers who have Online Backup is almost same as those who have no internet. A large number of customers have no Online Backup.
The number of customers who have Device Protection is almost same as those who have no internet. A large number of customers have no Device Protection.
The number of customers who have Tech Support is almost same as those who have no internet. A large number of customers have no Tech Support.
The number of customers who have Streaming TV and Streaming Movies is almost same as those who don't.

![Barplots2](https://user-images.githubusercontent.com/38309595/122850623-e3f10500-d350-11eb-835a-890a2d94250a.PNG)

Observations -
Most customers have month-month contract. The number of customers having 1-year and 2-year contracts is almost the same.
Most customers have paperless billing.
The number of customers using Mailed Check, Credit Card and Bank Transfer for payment is almost the same. The number is highest for Electronic Check.
Distribution of target variable is not uniform - the number of customers who haven't churned is more than double of those who have.

#Relationship between numeric variables
Used a scatter plot to identify any linear relationships


from pandas.plotting import scatter_matrix
scatter_matrix(churn,diagonal = 'kde',alpha = 0.2, figsize = (6, 6))
![Scatter](https://user-images.githubusercontent.com/38309595/122859345-3b966d00-d35f-11eb-9090-ce66af35f401.PNG)

#Relationship between Target and Numeric variables
import numpy as np
churn.groupby('Churn')['tenure'].aggregate(np.mean).plot(kind='bar',stacked=False,title='Mean tenure vs. Churn')
churn.groupby('Churn')['MonthlyCharges'].aggregate(np.mean).plot(kind='bar',stacked=False,title='Mean MonthlyCharges vs. Churn')
churn.groupby('Churn')['TotalCharges'].aggregate(np.mean).plot(kind='bar',stacked=False,title='Mean TotalCharges vs. Churn')
![MeanTenureVs](https://user-images.githubusercontent.com/38309595/122860412-fffca280-d360-11eb-989f-5fcaf0084fb5.PNG)

Observations -
The customers who churned have lower mean tenure and higher mean monthly charges, but lower mean total charges

Relationship between Target and Nominal variables
churn.groupby(['gender','Churn']).size().unstack().plot(kind='bar',stacked=True,title='Relationship btw gender & Churn')
churn.groupby(['SeniorCitizen','Churn']).size().unstack().plot(kind='bar',stacked=True,title='Relationship btw Senior Citizen & Churn')
![Gendervschurn](https://user-images.githubusercontent.com/38309595/122862708-e52c2d00-d364-11eb-95aa-0f1b24faac42.PNG)

churn.groupby(['Partner','Churn']).size()/7032
Similarly for other variables

Observations -
Gender does not affect customer churn, whereas Senior Citizens are more likely to churn.
Customers without a dependent or partner are more likely to churn.
Multiple Lines does not affect churn.
Customers having Fiber Optic are more likely to churn.
Customers having internet, without Device Protection, Online Security, Backup and Tech Support are more likely to churn.
Customers who have no internet service are less likely to churn. Streaming Movies and TV does not affect churn much.
Customers on month-month contract, having paperless billing and paying by electronic check and more likely to churn

#Data Preparation
Before building the model, all the nominal variables must be converted into numeric values using Label encoding
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
le.fit(churn['gender'])
churn['gender']=le.transform(churn['gender'])

le.fit(churn['SeniorCitizen'])
churn['SeniorCitizen']=le.transform(churn['SeniorCitizen'])

le.fit(churn['Partner'])
churn['Partner']=le.transform(churn['Partner'])

le.fit(churn['Dependents'])
churn['Dependents']=le.transform(churn['Dependents'])

le.fit(churn['MultipleLines'])
churn['MultipleLines']=le.transform(churn['MultipleLines'])

le.fit(churn['InternetService'])
churn['InternetService']=le.transform(churn['InternetService'])

le.fit(churn['OnlineSecurity'])
churn['OnlineSecurity']=le.transform(churn['OnlineSecurity'])

le.fit(churn['OnlineBackup'])
churn['OnlineBackup']=le.transform(churn['OnlineBackup'])

le.fit(churn['DeviceProtection'])
churn['DeviceProtection']=le.transform(churn['DeviceProtection'])

le.fit(churn['TechSupport'])
churn['TechSupport']=le.transform(churn['TechSupport'])

le.fit(churn['StreamingTV'])
churn['StreamingTV']=le.transform(churn['StreamingTV'])

le.fit(churn['StreamingMovies'])
churn['StreamingMovies']=le.transform(churn['StreamingMovies'])

le.fit(churn['Contract'])
churn['Contract']=le.transform(churn['Contract'])

le.fit(churn['PaperlessBilling'])
churn['PaperlessBilling']=le.transform(churn['PaperlessBilling'])

le.fit(churn['PaymentMethod'])
churn['PaymentMethod']=le.transform(churn['PaymentMethod'])

Below is what the tranformed dataframe looks like
![Df](https://user-images.githubusercontent.com/38309595/123719537-283a5300-d8c5-11eb-97f4-34f56f545f83.PNG)

The TotalCharges attribute was not used for building the model because it is highly correlated to MonthlyCharges
from sklearn.model_selection import train_test_split
churn_train, churn_test = train_test_split(churn, test_size=0.25)

#removed total charges attribute because it is correlated to monthly charges
features = ['gender', 'SeniorCitizen', 'Partner',
        'Dependents', 'tenure',
        'InternetService',
        'OnlineSecurity', 'MultipleLines',
       'OnlineBackup',  'DeviceProtection',
        'TechSupport',  'StreamingTV',
        'StreamingMovies',  'Contract',
       'PaperlessBilling', 
       'PaymentMethod', 'MonthlyCharges']

target=['Churn']

# Data Modeling


The independent variables have been transformed to integers for ease in building the model. The dependent variable, Churn, is a class label. Therefore, Classification models can be chosen to predict the churn of customers. Several algorithms which help predict class labels have been modelled such as Decision Tree, Random Forest, AdaBoost etc.
Further, it was noticed that the dataset was not balanced with respect to the distribution of classes. Various sampling methods such as random over-sampling, SMOTE sampling and SMOTE with Tomek links were experimented with.

# Experimental Results


Various available models were experimented with to compare results.
10-fold cross validation was used for evaluation. The evaluation measures used are Accuracy, Precision, Recall and F1-score.
Accuracy is the degree to which the result of a measurement, calculation, or specification conforms to the correct value or a standard. Precision is the fraction of relevant instances among the retrieved instances,while Recall is the fraction of relevant instances that have been retrieved over the total amount of relevant instances.

from sklearn import tree
import pandas as pd

clf = tree.DecisionTreeClassifier()  # We want to build a DT

clf= clf.fit(churn_train[features], churn_train[target])
predictions = clf.predict(churn_test[features])
probs = clf.predict_proba(churn_test[features])
display(predictions)

score = clf.score(churn_test[features], churn_test[target])
print("Accuracy: ", score)

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

get_ipython().magic('matplotlib inline')
confusion_matrix = pd.DataFrame(
    confusion_matrix(churn_test[target], predictions), 
    columns=["Predicted False", "Predicted True"], 
    index=["Actual False", "Actual True"]
)
display(confusion_matrix)

print('Accuracy of DT classifier on test set: {:.2f}'.format(clf.score(churn_test[features], churn_test[target])))
print(classification_report(churn_test[target], predictions))

from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb = gnb.fit(churn_train[features], np.ravel(churn_train[target]))
predictions = gnb.predict(churn_test[features])
probs = gnb.predict_proba(churn_test[features])
display(predictions)

score = gnb.score(churn_test[features], np.ravel(churn_test[target]))
print("Accuracy: ", score)

print('Accuracy of NB classifier on test set: {:.2f}'.format(gnb.score(churn_test[features], churn_test[target])))
print(classification_report(churn_test[target], predictions))

from sklearn.linear_model import LogisticRegression
 
logisticRegr = LogisticRegression()
logisticRegr.fit(churn_train[features], np.ravel(churn_train[target]))
predictions = logisticRegr.predict(churn_test[features])
probs = logisticRegr.predict_proba(churn_test[features])
display(predictions)

score = logisticRegr.score(churn_test[features], churn_test[target])
print("Accuracy: ", score)

print('Accuracy of LR classifier on test set: {:.2f}'.format(logisticRegr.score(churn_test[features], churn_test[target])))
print(classification_report(churn_test[target], predictions))

from sklearn.model_selection import cross_val_score
clf = tree.DecisionTreeClassifier()
DTscores = cross_val_score(clf,churn[features],churn[target],cv=10,scoring='accuracy')
print (DTscores.mean())  

from sklearn.ensemble import AdaBoostClassifier
ABclf = AdaBoostClassifier(n_estimators=100)  # Build AdaBoost classier with 100 base classifiers
ABscores = cross_val_score(ABclf, churn[features],np.ravel(churn[target]),cv=10,scoring='accuracy')
print(ABscores.mean())  

from sklearn.ensemble import RandomForestClassifier
RFclf = RandomForestClassifier(n_estimators=100,max_features=10)
RFscores = cross_val_score(RFclf,churn[features],np.ravel(churn[target]),cv=10,scoring='accuracy')
print (RFscores.mean())

NBclf = GaussianNB()
NBscores = cross_val_score(NBclf, churn[features],np.ravel(churn[target]),cv=10,scoring='accuracy')
print(NBscores.mean()) 

from sklearn import svm
SVMclf = svm.SVC()     
SVMscores = cross_val_score(SVMclf, churn[features],np.ravel(churn[target]),cv=10,scoring='accuracy')
print(SVMscores.mean()) 

from imblearn.over_sampling import RandomOverSampler
#Random Over-sampling for balanced dataset
ros = RandomOverSampler()
churnx_os, churny_os = ros.fit_sample(churn[features], np.ravel(churn[target]))
data_upsampled = pd.DataFrame(churnx_os)
data_upsampled.columns = features
data_upsampled['Churn'] = churny_os
data_upsampled.Churn.value_counts()

churn_train, churn_test = train_test_split(data_upsampled, test_size = 0.25)

#Training the model with over-sampled dataset


clf= clf.fit(churn_train[features], churn_train[target])
predictions = clf.predict(churn_test[features])
probs = clf.predict_proba(churn_test[features])
display(predictions)

score = clf.score(churn_test[features], np.ravel(churn_test[target]))
print("Accuracy: ", score)

from sklearn.metrics import confusion_matrix
get_ipython().magic('matplotlib inline')
matrix = pd.DataFrame(
    confusion_matrix(churn_test[target], predictions), 
    columns=["Predicted False", "Predicted True"], 
    index=["Actual False", "Actual True"]
)
display(matrix)

Feature importance of the various features was computed based on the best performing models to identify the most useful features.

print('Accuracy of DT classifier on test set: {:.2f}'.format(clf.score(churn_test[features], churn_test[target])))
print(classification_report(churn_test[target], predictions))

churn_f = pd.DataFrame(clf.feature_importances_, columns=["importance"])
churn_f["labels"] = features
churn_f.sort_values("importance", inplace=True, ascending=False)
display(churn_f.head(5))

RFclf = RandomForestClassifier(n_estimators=100,max_features=10)
RFscores = cross_val_score(RFclf,data_upsampled[features],np.ravel(data_upsampled[target]),cv=10,scoring='accuracy')
print (RFscores.mean())

RFclf = RandomForestClassifier(n_estimators=100,max_features=10)
RFclf= RFclf.fit(churn_train[features], np.ravel(churn_train[target]))
predictions = RFclf.predict(churn_test[features])
probs = RFclf.predict_proba(churn_test[features])
display(predictions)

score = RFclf.score(churn_test[features], np.ravel(churn_test[target]))
print("Accuracy: ", score)

print('Accuracy of RF classifier on test set: {:.2f}'.format(RFclf.score(churn_test[features], churn_test[target])))
print(classification_report(churn_test[target], predictions))

get_ipython().magic('matplotlib inline')
matrix = pd.DataFrame(
    confusion_matrix(churn_test[target], predictions), 
    columns=["Predicted False", "Predicted True"], 
    index=["Actual False", "Actual True"]
)
display(matrix)

churn_f = pd.DataFrame(RFclf.feature_importances_, columns=["importance"])
churn_f["labels"] = features
churn_f.sort_values("importance", inplace=True, ascending=False)
display(churn_f.head(5))

#SMOTE over-sampling to balance dataset
from imblearn.over_sampling import SMOTE

smote = SMOTE(ratio='minority')
churnx_sm, churny_sm = smote.fit_sample(churn[features], np.ravel(churn[target]))

data_smote = pd.DataFrame(churnx_sm)
data_smote.columns = features
data_smote['Churn'] = churny_sm
data_smote.Churn.value_counts()

#Training the model with SMOTE-balanced dataset
clf = tree.DecisionTreeClassifier()
churn_train, churn_test = train_test_split(data_smote, test_size = 0.25)
clf= clf.fit(churn_train[features], churn_train[target])
predictions = clf.predict(churn_test[features])
probs = clf.predict_proba(churn_test[features])
display(predictions)

score = clf.score(churn_test[features], churn_test[target])
print("Accuracy: ", score)

RFclf = RandomForestClassifier(n_estimators=100,max_features=10)
RFscores = cross_val_score(RFclf,data_smote[features],np.ravel(data_smote[target]),cv=10,scoring='accuracy')
print (RFscores.mean())
#SMOTE does not improve performance

#Under-sampling + over-sampling to balance dataset
from imblearn.combine import SMOTETomek

smt = SMOTETomek(ratio='auto')
churnx_smt, churny_smt = smt.fit_sample(churn[features], np.ravel(churn[target]))

data_smotemek = pd.DataFrame(churnx_smt)
data_smotemek.columns = features
data_smotemek['Churn'] = churny_smt
data_smotemek.Churn.value_counts()

clf = tree.DecisionTreeClassifier()  # We want to build a DT

churn_train, churn_test = train_test_split(data_smotemek, test_size = 0.25)
clf= clf.fit(churn_train[features], churn_train[target])
predictions = clf.predict(churn_test[features])
probs = clf.predict_proba(churn_test[features])
display(predictions)

score = clf.score(churn_test[features], churn_test[target])
print("Accuracy: ", score)

print('Accuracy of DT classifier on test set: {:.2f}'.format(clf.score(churn_test[features], churn_test[target])))
print(classification_report(churn_test[target], predictions))

![Results](https://user-images.githubusercontent.com/38309595/123720480-8c5e1680-d8c7-11eb-90f4-e16d41d04310.PNG)

In the end, we could see that Random Forest model is the best out of all the models created. It performs the best with an accuracy of 0.904 average when random over-sampling dataset was used with 10-fold cross validation. It gives a precision, recall and accuracy of 0.87 without cross-validation when random over-sampling dataset was used.
Using the Feature Importance property of the fitted model, the 5 most important features were selected. Contract, MonthlyCharges, Tenure, OnlineSecurity and TechSupport (in this order) seem to be the features most indicative of customer churn.
Partial Dependence plots were then used to show how the most relevant features affects predictions of churn. They are calculated after a model has been fit.

from matplotlib import pyplot as plt
from pdpbox import pdp, get_dataset, info_plots

#Create the data that we will plot
pdp_tenure = pdp.pdp_isolate(model=RFclf, dataset=churn_test[features], model_features=features, feature='tenure')

#plot it
pdp.pdp_plot(pdp_tenure, 'Tenure')
plt.show()

# Conclusion and Insights


The aim of this project was to investigate efficient and accurate methods for predicting churn for a telco’s customers. Some pre-processing such as feature selection and label encoding, along with fine-tuning of parameters help attaining desired performance.
In conclusion, Contract, MonthlyCharges, OnlineSecurity and Tenure are strong predictors of Churn.
Customers who have been customers for shorter periods are more likely to leave. Higher the monthly charge is, higher the churn. Churn is also higher for those customers on Month-to-month contract. Churn is lower for customers who have Online Security facility.
As future work on this problem, we could try to use one-hot encoding and neural networks to improve precision and recall. Some sort of penalty or cost-based misclassification mechanism could be used to penalize incorrect classification of Churn as non-Churn.
























