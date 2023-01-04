import numpy as np
import pandas as pd
from  sklearn import preprocessing
import matplotlib.pyplot as plt
#Importing
data = pd.read_csv("bank-additional-full.csv",sep=';')


# Cleaning the dataset
#education encoding
data = data.copy()



# print(data['default'].value_counts())

#Encoding
#job convert value
lst=['basic.9y','basic.6y','basic.4y']
for i in lst:
    data.loc[data['education'] == i, 'education'] = "middle.school"

#month encoding
moth_dic={'may':5,'jul':7,'aug':8,'jun':6,'nov':11,'apr':4,'oct':10,'sep':9,'mar':3,'dec':12}
data['month'] =data['month'].map(moth_dic)

# print(data['month'].value_counts())

#day encoding
day_dict={'thu':5,'mon':2,'wed':4,'tue':3,'fri':6}
data['day_of_week'] = data['day_of_week'].map(day_dict)
# print(data['day_of_week'].value_counts())

#endcoding pdays
data.loc[data['pdays'] == 999, 'pdays'] = 0

#encoding yes,no,unknown value
binary_data_dict = {'yes':1,'no':0,'unknown':-1}
data['housing']=data['housing'].map(binary_data_dict)
data['default']=data['default'].map(binary_data_dict)
data['loan']=data['loan'].map(binary_data_dict)
data['y'] = data['y'].map(binary_data_dict)

#create dummy variables
dumy_for_poutcome = pd.get_dummies(data['poutcome'],prefix= 'dummy',drop_first=True)
dumy_for_contact = pd.get_dummies(data['contact'],prefix= 'dummy',drop_first=True)
data = pd.concat([data,dumy_for_contact,dumy_for_poutcome],axis=1)

data = data.drop(['poutcome','contact'],axis='columns')
#job,education and marital encoding by frequency encoding
frequency_encode_job=data['job'].value_counts().to_dict()
frequency_encode_education=data['education'].value_counts().to_dict()
frequency_encode_marital = data['marital'].value_counts().to_dict()
data['job']=data['job'].map(frequency_encode_job)
data['education']=data['education'].map(frequency_encode_education)
data['marital']=data['marital'].map(frequency_encode_marital)


# print(data['marital'].value_counts())


# Split into X and y
print(data.dtypes)
X = data.drop(['y'],axis=1) #delete the feature that not in the tables
y = data['y']
y = np.array(y)



#Choosing features
from sklearn.ensemble import ExtraTreesClassifier
model = ExtraTreesClassifier()
model.fit(X,y)

print("Showing the importance of each feature...")
feat_importances = pd.Series(model.feature_importances_, index=X.columns)
feat_importances.nlargest(X.shape[0]).plot(kind='barh')
plt.show()

X = data.drop(['pdays','previous','y'],axis=1)
y = data['y']
y = np.array(y)

#standandize
from sklearn.preprocessing import StandardScaler
sc_X= StandardScaler()
X = sc_X.fit_transform(X)


# Slpit train set and test set
from sklearn.model_selection import train_test_split
X_train ,X_test, y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)

# train
from sklearn.linear_model import LogisticRegression
model =  LogisticRegression()
model = model.fit(X_train,y_train)
print("The mean accuracy of the model is:",model.score(X_test,y_test)) #about 92%


