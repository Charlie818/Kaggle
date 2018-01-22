# coding=utf-8
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn import svm, linear_model
import xgboost as xgb
import re

from util import accuracy, cross_valid, parameter_search

path = "../data/"
df = pd.read_csv(path + "train.csv", index_col="PassengerId")
df_train = pd.read_csv(path + "train.csv", index_col="PassengerId")
df_test = pd.read_csv(path + "test.csv", index_col="PassengerId")
target = df_train['Survived']
df_train['training_set'] = True
df_test['training_set'] = False
df_full = pd.concat([df_train, df_test])
df_full = df_full.drop(['Name', 'Ticket', 'Fare', 'Embarked', 'Cabin'], axis=1)
df_full = df_full.fillna(value=0)
# for index,row in df_full.iterrows():
#     text=str(row['Cabin'])
#     cabin_class = re.search('[A-Z]+', text)
#     cabin_number = re.search('[0-9]+', text)
#     if cabin_class:cabin_class=cabin_class.group()
#     else:cabin_class='Z'
#     if cabin_number:cabin_number=cabin_number.group()
#     else:cabin_number=0
#     df_full.loc[index,'CabinClass']=cabin_class
# df_full.loc[index,'CabinNumber']=int(cabin_number)
# df_full=df_full.drop(['Cabin'],axis=1)
# df_full=df_full.fillna(value=0)
df_full = pd.get_dummies(df_full)
print(df_full.columns)
df_train = df_full[df_full['training_set'] == True]
df_train = df_train.drop('Survived', axis=1)
df_test = df_full[df_full['training_set'] == False]
df_test = df_test.drop('Survived', axis=1)
# rf=RandomForestRegressor(n_estimators=100,n_jobs=1)
# rf=RandomForestClassifier(n_estimators=50)
# rf=svm.SVC()
# rf=linear_model.SGDClassifier()
parameters = [100,200,300,400,500]
clf = xgb.XGBClassifier()
parameter_search(clf, df_train, target.values, parameters,cv=3)
#
# clf = xgb.XGBClassifier(n_estimators=200)
# clf.fit(df_train, target.values)
# preds = clf.predict(df_test)
# cross_valid(clf,df_train,target.values)
# submission = pd.DataFrame({"PassengerId": df_test.index, "Survived": preds})
# submission.to_csv(path + "submission11.csv", index=False)
