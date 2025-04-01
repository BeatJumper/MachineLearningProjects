import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

# 1. 读取数据
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')
ans_df = pd.read_csv('ans.csv')

# 2. 处理缺失值
for col in train_df.columns:
    if train_df[col].dtype == 'object':
        train_df[col].fillna(train_df[col].mode()[0], inplace=True)
        test_df[col].fillna(test_df[col].mode()[0], inplace=True)
    else:
        train_df[col].fillna(train_df[col].mean(), inplace=True)
        test_df[col].fillna(test_df[col].mean(), inplace=True)

# 3. 特征编码
categorical_features = train_df.select_dtypes(include=['object']).columns
encoder = LabelEncoder()
for col in categorical_features:
    train_df[col] = encoder.fit_transform(train_df[col])
    test_df[col] = encoder.transform(test_df[col])

# 4. 归一化数值特征
num_features = ['age', 'fnlwgt', 'education_num', 'capital_gain', 'capital_loss', 'hours_per_week']
scaler = StandardScaler()
train_df[num_features] = scaler.fit_transform(train_df[num_features])
test_df[num_features] = scaler.transform(test_df[num_features])

# 5. 划分训练集与验证集
X = train_df.drop(columns=['income'])
y = LabelEncoder().fit_transform(train_df['income'])
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. 训练 XGBoost 模型并进行超参数调优
xgb = XGBClassifier(objective='binary:logistic', use_label_encoder=False, eval_metric='logloss')
params = {
    'max_depth': [4, 6, 8],
    'learning_rate': [0.01, 0.1, 0.2],
    'n_estimators': [100, 200, 300]
}
grid = GridSearchCV(xgb, param_grid=params, cv=3, scoring='accuracy', n_jobs=-1)
grid.fit(X_train, y_train)

# 7. 评估最佳模型
best_model = grid.best_estimator_
y_pred = best_model.predict(X_val)
accuracy = accuracy_score(y_val, y_pred)
f1 = f1_score(y_val, y_pred)
auc = roc_auc_score(y_val, y_pred)
print(f'Validation Accuracy: {accuracy:.4f}, F1-score: {f1:.4f}, AUC-ROC: {auc:.4f}')

# 8. 在测试集上进行预测
test_predictions = best_model.predict(test_df)
pd.DataFrame({'Predicted': test_predictions}).to_csv('predictions.csv', index=False)
print('Predictions saved to predictions.csv')