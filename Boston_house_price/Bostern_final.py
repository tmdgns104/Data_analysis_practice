import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from scipy import stats
from scipy.stats import norm, skew
from subprocess import check_output
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from lightgbm import LGBMRegressor

# ============================ 초기 설정 ============================
warnings.filterwarnings('ignore')
sns.set_style('darkgrid')
sns.set_palette("muted")
pd.set_option('display.float_format', lambda x: '{:.3f}'.format(x))
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# ============================ 데이터 불러오기 ============================
folder = "D:\\Data_Analysis\\Boston_house_price\\house-prices-advanced-regression-techniques\\"
train = pd.read_csv(os.path.join(folder, 'train.csv'))
test = pd.read_csv(os.path.join(folder, 'test.csv'))
sample_submission = pd.read_csv(os.path.join(folder, 'sample_submission.csv'))

# ============================ 전처리 ============================
train.drop("Id", axis=1, inplace=True)
test.drop("Id", axis=1, inplace=True)

# 이상치 제거
train = train[(train['GrLivArea']<4000) | (train['SalePrice']>300000)].reset_index(drop=True)

# 결측치 비율 계산
train_na = (train.isnull().sum() / len(train)) * 100
test = test[[i for i in train_na[train_na < 99.9].index if 'SalePrice' != i]]

# 로그 변환 (주석 처리됨)
# train["SalePrice"] = np.log1p(train["SalePrice"])

# Pool, Misc, Fence 등의 결측치는 'None'으로 처리
cols_none = ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu']
for col in cols_none:
    train[col] = train[col].fillna("None")
    test[col] = test[col].fillna("None")

# LotFrontage는 Neighborhood의 중앙값으로 채움
train_lf_dict = train.groupby("Neighborhood")["LotFrontage"].median().to_dict()
test_lf_dict = test.groupby("Neighborhood")["LotFrontage"].median().to_dict()
train.loc[train['LotFrontage'].isnull(), 'LotFrontage'] = train['Neighborhood'].map(train_lf_dict)
test.loc[test['LotFrontage'].isnull(), 'LotFrontage'] = test['Neighborhood'].map(test_lf_dict)

# Garage 관련 결측치 처리
garage_cols_none = ['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond']
garage_cols_zero = ['GarageYrBlt', 'GarageArea', 'GarageCars']
for col in garage_cols_none:
    train[col] = train[col].fillna('None')
    test[col] = test[col].fillna('None')
for col in garage_cols_zero:
    train[col] = train[col].fillna(0)
    test[col] = test[col].fillna(0)

# Basement 관련 결측치 처리
bsmt_cols_zero = ['BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath']
bsmt_cols_none = ['BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2']
for col in bsmt_cols_zero:
    train[col] = train[col].fillna(0)
    test[col] = test[col].fillna(0)
for col in bsmt_cols_none:
    train[col] = train[col].fillna('None')
    test[col] = test[col].fillna('None')

# 그 외 결측치 처리
train['MasVnrType'] = train['MasVnrType'].fillna('None')
test['MasVnrType'] = test['MasVnrType'].fillna('None')
train['MasVnrType'] = train['MasVnrType'].fillna(0)
test['MasVnrType'] = test['MasVnrType'].fillna(0)

for col in ['MSZoning', 'Electrical', 'KitchenQual', 'Exterior1st', 'Exterior2nd', 'SaleType']:
    train[col] = train[col].fillna(train[col].mode()[0])
    test[col] = test[col].fillna(test[col].mode()[0])

train.drop(['Utilities'], axis=1, inplace=True)
test.drop(['Utilities'], axis=1, inplace=True)

train['Functional'] = train['Functional'].fillna('Typ')
test['Functional'] = test['Functional'].fillna('Typ')

train['MSSubClass'] = train['MSSubClass'].fillna("None")
test['MSSubClass'] = test['MSSubClass'].fillna("None")

# ============================ 범주형 처리 ============================
# 문자열을 문자형으로 변환
train['MSSubClass'] = train['MSSubClass'].astype(str)
test['MSSubClass'] = test['MSSubClass'].astype(str)
train['OverallCond'] = train['OverallCond'].astype(str)
test['OverallCond'] = test['OverallCond'].astype(str)
train['YrSold'] = train['YrSold'].astype(str)
train['MoSold'] = train['MoSold'].astype(str)
test['YrSold'] = test['YrSold'].astype(str)
test['MoSold'] = test['MoSold'].astype(str)

# 라벨 인코딩 적용
cols = ['FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond',
        'ExterQual', 'ExterCond','HeatingQC', 'PoolQC', 'KitchenQual', 'BsmtFinType1',
        'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope',
        'LotShape', 'PavedDrive', 'Street', 'Alley', 'CentralAir', 'MSSubClass', 'OverallCond',
        'YrSold', 'MoSold']
for c in cols:
    lbl = LabelEncoder()
    lbl.fit(list(train[c].values) + list(test[c].values))
    train[c] = lbl.transform(list(train[c].values))
    test[c] = lbl.transform(list(test[c].values))

# 숫자형 열도 인코딩 (중복 제거된 열 리스트)
obj_cols = list(set(train.dtypes[train.dtypes == "object"].index))
for c in obj_cols:
    lbl = LabelEncoder()
    lbl.fit(list(train[c].values) + list(test[c].values))
    train[c] = lbl.transform(list(train[c].values))
    test[c] = lbl.transform(list(test[c].values))

# ============================ 피처 엔지니어링 ============================
#전체 면적 계산
train['TotalSF'] = train['TotalBsmtSF'] + train['1stFlrSF'] + train['2ndFlrSF']
test['TotalSF'] = test['TotalBsmtSF'] + test['1stFlrSF'] + test['2ndFlrSF']

# ============================ 모델 학습 ============================
y = train['SalePrice']
X = train.drop(['SalePrice'], axis=1)
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.3, random_state=0)

model = LGBMRegressor()
model.fit(X_train, y_train)

# ============================ 예측 및 평가 ============================
pred_train = model.predict(X_train)
pred_valid = model.predict(X_valid)

# 원래 스케일로 복원
def safe_rmsle(y_true, y_pred):
    return np.sqrt(np.mean((np.log1p(y_pred) - np.log1p(y_true)) ** 2))

# 예측 평가
print("RMSLE (Train):", safe_rmsle(y_train, pred_train))
print("RMSLE (Valid):", safe_rmsle(y_valid, pred_valid))

# ============================ 예측 시각화 ============================
temp_df = pd.DataFrame({'Actual': y_valid, 'Predicted': pred_valid}).reset_index(drop=True)
plt.figure(figsize=(12,6))
plt.plot(temp_df['Actual'], color='gray', label='Actual')
plt.plot(temp_df['Predicted'], color='red', label='Predicted')
plt.legend()
plt.title("Validation Set: Actual vs Predicted")
plt.show()

# ============================ 결과 저장 ============================
pred_test = model.predict(test)
sample_submission['SalePrice'] = pred_test
sample_submission.to_csv("sample_submission_final.csv", index=False)
