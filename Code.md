#Data Exploration
#exploratory data analysis of the dataframe churn into which we have read our dataset

churn.shape
#(7043, 21) - 7043 rows and 21 columns including target variable

#This shows the unique values in each column
def rstr(df): return df.apply(lambda x: [x.unique()])
print(rstr(churn))

#Delete columns ‘customerID’ and ‘PhoneService’ from dataframe ‘churn’, 
#because customer ID is not useful for model prediction, and the information in variable ‘PhoneService’ is included inside variable ‘MultipleLines’
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



















