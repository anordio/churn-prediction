# churn-prediction
Ensemble XGBoost/MLP for churn prediction [0.98 F1 score, 0.99 AUC] 
(XGBoost, MLP, Ensemble methods, Feature Engineering, Hyperparameters optimization)
I omitted the complete Feature Engineering part due to the privacy of data.
Preprocessing: Handling numerical / categorical / datatime features according to my knowledge of the field (B2B)., -> StandardScaler

Hyperparameters optimization was done using Hyperopt. This was used also for choosing the weights of the linear combination of the output probabilities of the MLP and XGBoost methods.

