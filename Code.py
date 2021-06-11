# Data Exploration
# exploratory data analysis of the dataframe churn into which we have read our dataset

churn.shape
# (7043, 21) - 7043 rows and 21 columns including target variable

#This shows the unique values in each column
def rstr(df): return df.apply(lambda x: [x.unique()])
print(rstr(churn))

# Delete columns ‘customerID’ and ‘PhoneService’ from dataframe ‘churn’, 
# because customer ID is not useful for model prediction, and the information in variable ‘PhoneService’ is included inside variable ‘MultipleLines’
churn=churn.drop(['customerID','PhoneService'],axis=1)

#Check if the dataframe still includes any null values or not
churn.isnull().any()

#Since we do not have any NULL values, we proceed with some plotting
%matplotlib inline
churn.hist()






