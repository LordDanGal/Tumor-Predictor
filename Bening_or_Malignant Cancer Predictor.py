# importing the modules
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

# reading the tumor csv file
tumor_data=pd.read_csv('tumor.csv')
print(tumor_data.head())

# seaborn plots of the data
sns.jointplot(x='radius_mean',y='texture_mean',data=tumor_data)
plt.show()
sns.heatmap(tumor_data.corr())
plt.show()

# checking for null values
print(tumor_data.isnull().sum())

# Forming our x and y based on the data in the worst category
X_worst=tumor_data[['radius_worst','texture_worst','perimeter_worst','compactness_worst',
                    'concavity_worst','concave points_worst','symmetry_worst','fractal_dimension_worst']]
print(X_worst.tail())
y=tumor_data['diagnosis']
print(y.head(20))

# Splitting the dataset into train and test
X_worst_train,X_worst_test,y_worst_train,y_worst_test=train_test_split(X_worst,y,
                    test_size=0.25,random_state=100)

# Creating the model and fitting the train data and using the model to predict the y values of the x test values
worst_model=LogisticRegression()
worst_model.fit(X_worst_train,y_worst_train)
y_worst_predictor=worst_model.predict(X_worst_test)
print(y_worst_predictor)


# Checking how great the model is doing
cf_worst=confusion_matrix(y_worst_test,y_worst_predictor)
print(cf_worst)
cr_worst=classification_report(y_worst_test,y_worst_predictor)
print(cr_worst)


# Doing the same using the mean dataset columns
print(tumor_data.columns)
X_mean=tumor_data[['radius_mean','texture_mean','perimeter_mean','area_mean','smoothness_mean',
                   'compactness_mean','concavity_mean','concave points_mean','symmetry_mean',
                   'fractal_dimension_mean']]
Y=tumor_data['diagnosis']
X_mean_train,X_mean_test,Y_mean_train,Y_mean_test=train_test_split(X_mean,Y,test_size=0.2,random_state=120)
mean_model=LogisticRegression()
mean_model.fit(X_mean_train,Y_mean_train)
y_mean_predictor=mean_model.predict(X_mean_test)
print(y_mean_predictor)
cf_mean=confusion_matrix(Y_mean_test,y_mean_predictor)
cr_mean=classification_report(Y_mean_test,y_mean_predictor)
print(cf_mean)
print(cr_mean)

# Doing the same using the se dataset columns
X_se=tumor_data[['radius_se','texture_se','perimeter_se','area_se','smoothness_se','compactness_se',
                'concavity_se','concave points_se','symmetry_se','fractal_dimension_se']]
Y=tumor_data['diagnosis']
X_se_train,X_se_test,Y_se_train,Y_se_test=train_test_split(X_se,Y,test_size=0.3)
se_model=LogisticRegression()
se_model.fit(X_se_train,Y_se_train)
y_se_predictor=se_model.predict(X_se_test)
print(y_se_predictor)
cr_se=classification_report(Y_se_test,y_se_predictor)
cf_se=confusion_matrix(Y_se_test,y_se_predictor)
print(cr_se)
print(cf_se)


# Labelling the malignant and bening results
diagnosis_label=np.where(tumor_data['diagnosis']=='M',0,1)
print(diagnosis_label)


# Creating the function for predicting based on all the datasets
def mean_cancer_predictor(rad,texture,perimeter,area,smoothness,compact,concave,concavePoints,symmetry,fractal):
    if mean_model.predict([[rad,texture,perimeter,area,smoothness,compact,concave,concavePoints,symmetry,fractal]])==0:
        print('Based on the mean data, this patient has malignant cancer')
    else:
        print('Based on the mean data, this patient has benign cancer')
rad_mean=float(input('Enter a value for the mean radius: '))
texture_mean=float(input('Enter a value for the mean texture: '))
perimeter_mean=float(input('Enter a value for the mean perimeter: '))
area_mean=float(input('Enter a value for the mean area: '))
smoothness_mean=float(input('Enter a value for the mean smoothness: '))
compact_mean=float(input('Enter a value for the mean compactness: '))
concave_mean=float(input('Enter a value for the mean concave: '))
concavePoints_mean=float(input('Enter a value for the mean concave Points: '))
symmetry_mean=float(input('Enter a value for the mean symmetry: '))
fractal_mean=float(input('Enter a value for the mean fractal: '))

print(mean_cancer_predictor(rad_mean,texture_mean,perimeter_mean,area_mean,smoothness_mean,compact_mean,
                      concave_mean,concavePoints_mean,symmetry_mean,fractal_mean))


def worst_cancer_predictor(rad,texture,perimeter,compact,concave,concavePoints,symmetry,fractal):
    if worst_model.predict([[rad,texture,perimeter,compact,concave,concavePoints,symmetry,fractal]])==0:
        print('Based on the worst data, this patient has malignant cancer')
    else:
        print('Based on the worst data, this patient has benign cancer')
rad_worst= float(input('Enter the worst radius value: '))
texture_worst= float(input('Enter the worst texture value: '))
perimeter_worst= float(input('Enter the worst perimeter value: '))
compact_worst= float(input('Enter the worst compact value: '))
concave_worst= float(input('Enter the worst concave value: '))
concavePoints_worst= float(input('Enter the worst concave points value: '))
symmetry_worst= float(input('Enter the worst symmetry value: '))
fractal_worst= float(input('Enter the worst fractal value: '))

print(worst_cancer_predictor(rad_worst,texture_worst,perimeter_worst,compact_worst,concave_worst,concavePoints_worst
                       ,symmetry_worst,fractal_worst))


def se_cancer_predictor(rad,texture,perimeter,area,smoothness,compact,concave,concavePoints,symmetry,fractal):
    if se_model.predict([[rad,texture,perimeter,area,smoothness,compact,concave,concavePoints,symmetry,fractal]])==0:
        print('Based on the se data, this patient has malignant cancer')
    else:
        print('Based on the se data, this patient has benign cancer')
rad_se= float(input('Enter the se radius value: '))
texture_se= float(input('Enter the se texture value: '))
perimeter_se= float(input('Enter the se perimeter value: '))
area_se= float(input('Enter the se area value: '))
smoothness_se= float(input('Enter the se smoothness value: '))
compact_se= float(input('Enter the se compact value: '))
concave_se= float(input('Enter the se concave value: '))
concavePoints_se= float(input('Enter the se concave points value: '))
symmetry_se= float(input('Enter the se symmetry value: '))
fractal_se= float(input('Enter the se fractal value: '))

print(se_cancer_predictor(rad_se,texture_se,perimeter_se,area_se,smoothness_se,compact_se,concave_se,
                    concavePoints_se,symmetry_se,fractal_se))