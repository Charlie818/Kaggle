# coding=utf-8
import pandas as pd
import tensorflow as tf
import numpy as np


def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))

def model(X, w_h, w_o):
    h = tf.nn.sigmoid(tf.matmul(X, w_h)) # this is a basic mlp, think 2 stacked logistic regressions
    return tf.matmul(h, w_o) # note that we dont take the softmax at the end because our cost fn does that for us


path = "../data/"
df = pd.read_csv(path+"train.csv",index_col="PassengerId")
df_train=pd.read_csv(path+"train.csv",index_col="PassengerId")
df_test=pd.read_csv(path+"test.csv",index_col="PassengerId")
train_Y=df_train['Survived'].values
df_train['training_set']=True
df_test['training_set']=False
df_full=pd.concat([df_train,df_test])
df_full=df_full.drop(['Name','Ticket','Fare','Embarked'],axis=1)
df_full=df_full.fillna(value=0)
df_full=df_full.drop(['Cabin'],axis=1)
df_full=pd.get_dummies(df_full)
df_train=df_full[df_full['training_set']==True]
df_train=df_train.drop('Survived',axis=1)

df_test=df_full[df_full['training_set']==False]
df_test=df_test.drop('Survived',axis=1)

train_X=df_train.values
test_X=df_test.values
train_Y=train_Y.reshape(-1,1)
print(train_X.shape,test_X.shape,train_Y.shape)
X = tf.placeholder("float", [None, 7])
Y = tf.placeholder("float", [None, 1])
w1=init_weights([7,50])
w2=init_weights([50,1])
py_x=model(X,w1,w2)

print(py_x)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=py_x, labels=Y)) # compute costs
train_op = tf.train.GradientDescentOptimizer(0.05).minimize(cost) # construct an optimizer
predict_op = tf.argmax(py_x, 1)

# Launch the graph in a session
with tf.Session() as sess:
    # you need to initialize all variables
    tf.initialize_all_variables().run()
    for i in range(100):
        _,loss=sess.run([train_op,cost], feed_dict={X: train_X, Y: train_Y})
        print(loss)
    preds=sess.run(py_x,feed_dict={X:test_X}).squeeze()
    print(preds.shape)
    submission=pd.DataFrame({"PassengerId":df_test.index,"Survived":preds})
    submission.to_csv(path+"submission11.csv",index=False)
