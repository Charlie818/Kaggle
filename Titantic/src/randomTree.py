# coding=utf-8
import pandas as pd
from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier
from sklearn import svm
import xgboost as xgb
import re
path = "../data/"
df = pd.read_csv(path+"train.csv",index_col="PassengerId")
df_train=pd.read_csv(path+"train.csv",index_col="PassengerId")
df_test=pd.read_csv(path+"test.csv",index_col="PassengerId")
target=df_train['Survived']
df_train['training_set']=True
df_test['training_set']=False
df_full=pd.concat([df_train,df_test])
df_full=df_full.drop(['Name','Ticket','Fare','Embarked'],axis=1)
df_full=df_full.fillna(value={'Cabin':'Z0'})
df_full=df_full.fillna(value=0)
for index,row in df_full.iterrows():
    text=str(row['Cabin'])
    cabin_class = re.search('[A-Z]+', text)
    cabin_number = re.search('[0-9]+', text)
    if cabin_class:cabin_class=cabin_class.group()
    else:cabin_class='Z'
    if cabin_number:cabin_number=cabin_number.group()
    else:cabin_number=0
    df_full.loc[index,'CabinClass']=cabin_class
    # df_full.loc[index,'CabinNumber']=int(cabin_number)
df_full=df_full.drop(['Cabin'],axis=1)
# df_full=df_full.fillna(value=0)
df_full=pd.get_dummies(df_full)
print(df_full.columns)
df_train=df_full[df_full['training_set']==True]
df_train=df_train.drop('Survived',axis=1)
df_test=df_full[df_full['training_set']==False]
df_test=df_test.drop('Survived',axis=1)
# rf=RandomForestRegressor(n_estimators=100,n_jobs=1)
# rf=RandomForestClassifier(n_estimators=50)
# rf=svm.SVC()
rf=xgb.XGBClassifier()
rf.fit(df_train,target)
preds=rf.predict(df_test)
# for i in range(preds.shape[0]):
#     if preds[i]>0.5:preds[i]=1
#     else:preds[i]=0
# preds=preds.astype('int')
submission=pd.DataFrame({"PassengerId":df_test.index,"Survived":preds})
submission.to_csv(path+"submission9.csv",index=False)
