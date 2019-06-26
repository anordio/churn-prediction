from google.colab import files
from google.colab import drive

#### MOUNT of my GOOGLE DRIVE FOLDER

drive.mount('/content/gdrive')


import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, balanced_accuracy_score
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from hyperopt import hp, tpe, fmin
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Activation, Dropout, Dense
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.ensemble import VotingClassifier
from keras import backend as K


df = pd.read_excel('/content/gdrive/My Drive/churn_prediction/churn_df_preprocessed_new3.xlsx') # Load Preprocessed Dataset

lb = LabelBinarizer()
targets = lb.fit_transform(df.loc[:,'Contract Status'].values)
df['Targets'] = targets

X = df.drop(columns=['Targets'])
Y = df['Targets']

corr = list()
for i in range(0,X.shape[1]):
    r,p = pearsonr(X[X.columns[i]].values,Y.values)
    if abs(r)>=0.1:
        print(X.columns[i])
    corr.append(r)
        
plt.plot(corr,'o-')
plt.show()

seed = 7
test_size = 0.33
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.values)
X_test_scaled = scaler.transform(X_test.values)

#### Hyperopt parameters ####

param = {'max_depth': hp.choice('max_depth', np.arange(3, 8+1,dtype=int)),
          'learning_rate': hp.uniform('learning_rate',0.01,0.2),
          'n_estimators': hp.choice('n_estimators', np.arange(150, 400+1,dtype=int)),
          'gamma':hp.choice('gamma', np.arange(0, 5+1, dtype=int)),
          'colsample_bytree': hp.uniform('colsample_bytree',0.3,1),
          'dropout1': hp.choice('dropout1',[0.2,0.3,0.4,0.5]),
          'dropout2': hp.choice('dropout2',[0.2,0.3,0.4,0.5]),
          'dropout3': hp.choice('dropout3',[0.2,0.3,0.4,0.5]),
          'epochs': hp.choice('epochs',[20,30,40,50]),
          'batch_size': hp.choice('batch_size',[32,64,128]),
          'weight_clf': hp.uniform('weight_clf',0,1),
          'weight_mlp': hp.uniform('weight_mlp',0,1)}


def mlp(dropout1=0.5,dropout2=0.5,dropout3=0.5): 
    mlp = Sequential()
    mlp.add(Dense(512,input_dim=X_train_scaled.shape[1],activation='relu'))
    mlp.add(Dropout(rate=dropout1))

    mlp.add(Dense(256,activation='relu'))
    mlp.add(Dropout(rate=dropout2))

    mlp.add(Dense(128,activation='relu'))
    mlp.add(Dropout(rate=dropout3))

    mlp.add(Dense(1,activation='sigmoid'))

    mlp.compile(loss='binary_crossentropy',optimizer='adam',metrics=['binary_accuracy'])
    
    return mlp

def objective(params):
    estimators = []
    
    mlp_model = KerasClassifier(build_fn=mlp, epochs=params['epochs'], batch_size=params['batch_size'], dropout1=params['dropout1'],
                                dropout2=params['dropout2'], dropout3=params['dropout3'], verbose=0)
    estimators.append(('MLP',mlp_model))
    
    clf = XGBClassifier(n_jobs=-1,booster='gbtree',verbosity=1,max_depth=params['max_depth'],learning_rate=params['learning_rate'],
                        n_estimators=params['n_estimators'],gamma=params['gamma'],colsample_bytree=params['colsample_bytree'],tree_method='gpu_hist',
                       predictor='gpu_predictor')
    estimators.append(('XGBClassifier',clf))
    
    ensamble = VotingClassifier(estimators,voting='soft',weights=[params['weight_mlp'],params['weight_clf']])
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=7)
    
    score = cross_val_score(ensamble, X_train_scaled, y_train, scoring='balanced_accuracy',cv=kfold).mean()
    print("ACC {:.5f} params {}".format(score, params))
    
    K.clear_session()
    return 1-score

best = fmin(fn=objective,space=param,algo=tpe.suggest,max_evals=40)
print("Hyperopt estimated optimum {}".format(best))
  
best_params = {'batch_size': 128, 'colsample_bytree': 0.8733739570686117, 'dropout1': 0.3, 'dropout2': 0.2, 'dropout3': 0.2, 'epochs': 50, 'gamma': 0, 'learning_rate': 0.09792689449963456, 'max_depth': 8, 'n_estimators': 378, 'weight_clf': 0.7025673520496984, 'weight_mlp': 0.11686999132233321}

def final_model(p,x,y):
    mlp = Sequential()
    mlp.add(Dense(512,input_dim=X_train_scaled.shape[1],activation='relu'))
    mlp.add(Dropout(rate=p['dropout1']))

    mlp.add(Dense(256,activation='relu'))
    mlp.add(Dropout(rate=p['dropout2']))

    mlp.add(Dense(128,activation='relu'))
    mlp.add(Dropout(rate=p['dropout3']))

    mlp.add(Dense(1,activation='sigmoid'))

    mlp.compile(loss='binary_crossentropy',optimizer='adam',metrics=['binary_accuracy'])
    mlp.fit(x,y,epochs=p['epochs'],batch_size=p['batch_size'],verbose=0)
    
    clf = XGBClassifier(booster='gbtree',verbosity=0,max_depth=p['max_depth'],learning_rate=p['learning_rate'],
                        n_estimators=p['n_estimators'],gamma=p['gamma'],colsample_bytree=p['colsample_bytree'],tree_method='gpu_hist',
                       predictor='gpu_predictor')
    clf.fit(x,y)
    
    return mlp, clf
  
final_mlp, final_clf = final_model(best_params,X_train_scaled,y_train)

X_test_scaled = scaler.transform(X_test)
mlp_prob_predictions = np.asarray([1-value[0] for value in final_mlp.predict_proba(X_test_scaled)])
clf_prob_predictions = np.asarray([1-value[0] for value in final_clf.predict_proba(X_test_scaled)])
final_prob_predictions = (best_params['weight_mlp']*mlp_prob_predictions + best_params['weight_clf']*
                          clf_prob_predictions)/(best_params['weight_mlp']+best_params['weight_clf'])

final_predictions = [round(value) for value in final_prob_predictions]


p = precision_score(y_test,final_predictions)
r = recall_score(y_test,final_predictions)
f1= f1_score(y_test,final_predictions)
acc = accuracy_score(y_test,final_predictions)
auc = roc_auc_score(y_test,final_prob_predictions)
print('Precision: {} Recall: {} F1: {} Accuracy: {} AUC: {} '.format(p,r,f1,acc,auc))
tn, fp, fn, tp = confusion_matrix(y_test,final_predictions).ravel()
print('TP: {} TN: {} FP: {} FN: {}'.format(tp,tn,fp,fn))

