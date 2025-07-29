import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

for dirname, _, filenames in os.walk(r'D:\Data_Analysis\Machine_predictive_maintenance_classification\machine-predictive-maintenance-classification'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

sample_submission = pd.read_csv(r'D:\Data_Analysis\Machine_predictive_maintenance_classification\machine-predictive-maintenance-classification\sample_submission.csv')

df = pd.read_csv(r'D:\Data_Analysis\Machine_predictive_maintenance_classification\machine-predictive-maintenance-classification\predictive_maintenance.csv')

# print(df.columns)
#  UDI(고유 ID), Product ID(제품 식별자), 'Type'(제품 타입), 'Air temperature [K]'(공기 온도),'Process temperature [K]'(공정 온도),
#  'Rotational speed [rpm]'(회전 속도), 'Torque [Nm]'(토크,회전 힘),'Tool wear [min]'(공구 마모 시간), 'Target'(고장 발생 여부, Binary Label), 'Failure Type'(고장 유형)
# print(df.describe)
# print(df[['Target']])
# print(df['Target'].value_counts())

#단위 까지 표시하기에는 치기 힘드니 다시 조정
df.columns = ['UDI', 'Product ID', 'Type', 'Air temperature', 'Process temperature', 'Rotational speed', 'Torque', 'Tool wear', 'Target', 'Failure Type']
# print(df)


#test 데이타 분리 해줌 20% random_state는 같은 숫자(randomstate=1 끼리 똑같음)끼리 항상 똑같이 데이터를 나눠줌
train,test = train_test_split(df, test_size=0.20, random_state=1)


# print(train.shape)
# print(test.shape)
# print(train.groupby(['Type'])['Target'].mean())
# print(test.head())

# 머신러닝은 문자열은 인식을 못하니 숫자로 변경
train['Type'] = train['Type'].replace(['M','L','H'], [1,2,3])
test['Type'] = test['Type'].replace(['M','L','H'], [1,2,3])


# index는 필요없으니 버림
train = train.drop(['Product ID','UDI'], axis=1)
test = test.drop(['Product ID','UDI'], axis=1)


#print(train['Failure Type'].value_counts())
# 결측치가 너무 많아서 날려버림
train = train.drop(['Failure Type'], axis=1)
test = test.drop(['Failure Type'], axis=1)


#학습을 위해서 예측해야될 값과 학습시킬 값 분리
y = train['Target']
X = train.drop(['Target'],axis=1)

#train data를 학습 시킬 데이터와 검증할 데이터로 나눔
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.30, random_state=1)


#모델 호출 요즘 LGBM이 최고
model = LGBMClassifier()
model.fit(X_train, y_train)
pred_train = model.predict(X_train)
print((pred_train == y_train).mean()) #학습 데이터라 100%


pred_valid = model.predict(X_valid)
print((pred_valid == y_valid).mean()) # 검증데이터 98.3%


#테스트 진행
y_test = test['Target']
X_test = test.drop(['Target'],axis=1)
pred_test = model.predict(X_test)
print((pred_test == y_test).mean()) #테스트 데이터 98.6%



