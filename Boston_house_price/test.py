import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt  # Matlab-style plotting
import seaborn as sns
import warnings
from scipy import stats
from scipy.stats import norm, skew #for some statistics
from subprocess import check_output
import os

from Machine_predictive_maintenance_classification.test import sample_submission


def ignore_warn(*args, **kwargs):
    pass


#Seaborn 그래프 기본 색깔로 설정
color = sns.color_palette()
#Seaborn 스타일을 'darkgrid'로 설정 (어두운 배경 + 격자)
sns.set_style('darkgrid')

#모든 경고 메시지를 무시
warnings.warn = ignore_warn #ignore annoying warning (from sklearn and seaborn)




#DataFrame/Series를 출력할 때 부동소수점 숫자의 표시 형식을 소수전 3째 자리로 지정
pd.set_option('display.float_format', lambda x: '{:.3f}'.format(x)) #Limiting floats output to 3 decimal points



folder = "D:\\Data_Analysis\\Boston_house_price\\house-prices-advanced-regression-techniques\\"
#print(os.listdir(folder))

train = pd.read_csv(os.path.join(folder + 'train.csv'))
test = pd.read_csv(os.path.join(folder + 'test.csv'))
sample_submission = pd.read_csv(os.path.join(folder + 'sample_submission.csv'))


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


##display the first five rows of the train dataset.
# print(train.head(5))

train = train.drop("Id", axis = 1)
test = test.drop("Id", axis = 1)

#오른쪽 하단에서 매우 큰 GrLivArea를 가진 두 개의 낮은 가격을 볼 수 있습니다. 이 값은 큰 값을 나타냅니다. 따라서 안전하게 삭제할 수 있습니다.
#두 항목의 산전도 그래프를 그림, 아웃 라이어 들이 보이니 지움
# plt.figure(figsize=[8,5])
# plt.scatter(train['GrLivArea'], train['SalePrice'])
# plt.ylabel('SalePrice', fontsize=13)
# plt.xlabel('GrLivArea', fontsize=13)
# plt.show()

#GrLivArea 가 4000 미만 이거나 SalePrice가 300000 큰 영역만 으로 train dataframe으로 재정의 후 인덱스 다시 달아줌/ 나머진 아웃라이어로 판단
train = train[(train['GrLivArea']<4000) | (train['SalePrice']>300000)].reset_index(drop=True)

# plt.figure(figsize=[8,5])
# plt.scatter(train['GrLivArea'], train['SalePrice'])
# plt.ylabel('SalePrice', fontsize=13)
# plt.xlabel('GrLivArea', fontsize=13)
# plt.show()



#이상값 제거는 항상 안전합니다. 이 두 개는 매우 크고 매우 나쁘기 때문에 삭제하기로 결정했습니다(매우 저렴한 가격에 매우 넓은 영역).
#훈련 데이터에는 다른 이상값이 더 있을 수 있습니다.
# 그러나 테스트 데이터에도 이상값이 있는 경우 모두 제거하면 모델에 나쁜 영향을 미칠 수 있습니다.
# 그렇기 때문에 모든 이상값을 제거하는 대신 일부 이상값에 대한 모델을 강력하게 만들 것입니다.
# 이 코드의 모델링 부분을 참조하시면 됩니다.

#정규성분포를 나타냄, 좌측으로 치우쳐져 있어서 log를 씌어서 중앙으로 이동시면 좋음
# sns.distplot(train['SalePrice'])
# plt.show()

#정규성(정규분포를 따르는지) 확인을 위한 Q-Q plot (Quantile-Quantile plot)
#특정 분포(보통 정규분포)와 비교하여 정규성을 시각적으로 평가하는 그래프
# x축: 이론적(정규분포) 분위수
# y축: 실제 데이터의 분위수
# 점: 데이터의 실제 분포를 나타냄
# 직선: "완벽한 정규분포라면" 따라야 할 기준선
#           패턴	                          해석	                                      의미
# 점들이 직선을 따라 잘 정렬됨 = 	          ✅ 정규분포에 가까움	                          정규성 만족
# S자 형태로 굽음 (끝에서 위·아래로 벌어짐)	  ❌ 꼬리가 두꺼움 (fat tails)	              정규성 위반 (왜도/첨도 있음)
# 위로 벌어짐	                              오른쪽 꼬리(고값)이 길다 (positive skew)	      로그 변환 필요
# 아래로 벌어짐	                          왼쪽 꼬리(저값)가 길다 (negative skew)	      정규화 필요
# 중간은 직선, 양 끝은 벗어남	              중심은 정규적, 극단값은 이상치일 수 있음	          극단값 처리 고려
# stats.probplot(train['SalePrice'], plot=plt)
# plt.show()
#현재 위로 벌어진 형태로 로그 변환이 필요해 보임

#로그로 변환
# train["SalePrice"] = np.log1p(train["SalePrice"])


#분포가 중앙으로 이동했음
# sns.distplot(train['SalePrice'])
# plt.show()
# # 직선 형태로 변화
# stats.probplot(train['SalePrice'], plot=plt)
# plt.show()
#이제 왜곡이 수정되어 데이터가 보다 정상적으로 분포된 것처럼 보입니다.
#제출시 다시 exponential 함수를 씌워서 다시 원상태로 복귀 시켜야됨



train_na = (train.isnull().sum() / len(train)) * 100
# print(train_na)    #null 값이 몇퍼센트가 존재하는지 확인

# print(train_na[train_na < 99.9].index)
train_na99 = train[train_na[train_na < 99.9].index]
# print(train.head())


test = test[[i for i in train_na[train_na < 99.9].index if 'SalePrice' != i]]
# print(test.head())

#print(train)
#Correlation map to see how features are correlated with SalePrice
#상관 관계, 두 항목들이 그래프를 그리는지 볼수 잇는 표 , 1 or -1 일수록 상관관계가 높음
# corrmat = train.select_dtypes(include=['number']).corr()
# plt.subplots(figsize=(12,9))
# sns.heatmap(corrmat, vmax=0.9, square=True)
# plt.show()


#결측치들을 NONE 값으로 채워줌
train["PoolQC"] = train["PoolQC"].fillna("None")
test["PoolQC"] = test["PoolQC"].fillna("None")
train["MiscFeature"] = train["MiscFeature"].fillna("None")
test["MiscFeature"] = test["MiscFeature"].fillna("None")
train["Alley"] = train["Alley"].fillna("None")
test["Alley"] = test["Alley"].fillna("None")
train["Fence"] = train["Fence"].fillna("None")
test["Fence"] = test["Fence"].fillna("None")
train["FireplaceQu"] = train["FireplaceQu"].fillna("None")
test["FireplaceQu"] = test["FireplaceQu"].fillna("None")

#A 별로 B의 중앙값을 구해라
#rain.groupby(["A"])["B"].median()
train_Neighborhood_LotFrontage_median_dict = train.groupby(["Neighborhood"])["LotFrontage"].median().to_dict()
#print(train_Neighborhood_LotFrontage_median_dict)

#train df 중 LostFrontage가 null값인 곳에 위 딕셔너리를 사용해서 각각의 중앙 값들을 채워 넣어줌
train.loc[train[train['LotFrontage'].isnull()].index,'LotFrontage'] = train.loc[train[train['LotFrontage'].isnull()].index,'Neighborhood'].map(train_Neighborhood_LotFrontage_median_dict)


test_Neighborhood_LotFrontage_median_dict = test.groupby(["Neighborhood"])["LotFrontage"].median().to_dict()
test.loc[test[test['LotFrontage'].isnull()].index,'LotFrontage'] = test.loc[test[test['LotFrontage'].isnull()].index,'Neighborhood'].map(test_Neighborhood_LotFrontage_median_dict)


for col in ['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond']:
    train[col] = train[col].fillna('None')
    test[col] = test[col].fillna('None')

for col in ['GarageYrBlt', 'GarageArea', 'GarageCars']:
    train[col] = train[col].fillna(0)
    test[col] = test[col].fillna(0)


for col in ['BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath']:
    train[col] = train[col].fillna(0)
    test[col] = test[col].fillna(0)


for col in ['BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2']:
    train[col] = train[col].fillna('None')
    test[col] = test[col].fillna('None')



train['MasVnrType'] = train['MasVnrType'].fillna('None')
test['MasVnrType'] = test['MasVnrType'].fillna('None')
train['MasVnrType'] = train['MasVnrType'].fillna(0)
test['MasVnrType'] = test['MasVnrType'].fillna(0)

#최빈값으로 채워 넣어줌
train['MSZoning'] = train['MSZoning'].fillna(train['MSZoning'].mode()[0])
test['MSZoning'] = test['MSZoning'].fillna(test['MSZoning'].mode()[0])


train = train.drop(['Utilities'], axis=1)
test = test.drop(['Utilities'], axis=1)



train["Functional"] = train["Functional"].fillna("Typ")
test["Functional"] = test["Functional"].fillna("Typ")

train['Electrical'] = train['Electrical'].fillna(train['Electrical'].mode()[0])
test['Electrical'] = test['Electrical'].fillna(test['Electrical'].mode()[0])


train['KitchenQual'] = train['KitchenQual'].fillna(train['KitchenQual'].mode()[0])
test['KitchenQual'] = test['KitchenQual'].fillna(test['KitchenQual'].mode()[0])


train['Exterior1st'] = train['Exterior1st'].fillna(train['Exterior1st'].mode()[0])
train['Exterior2nd'] = train['Exterior2nd'].fillna(train['Exterior2nd'].mode()[0])

test['Exterior1st'] = test['Exterior1st'].fillna(train['Exterior1st'].mode()[0])
test['Exterior2nd'] = test['Exterior2nd'].fillna(train['Exterior2nd'].mode()[0])


train['SaleType'] = train['SaleType'].fillna(train['SaleType'].mode()[0])
test['SaleType'] = test['SaleType'].fillna(test['SaleType'].mode()[0])

train['MSSubClass'] = train['MSSubClass'].fillna("None")
test['MSSubClass'] = test['MSSubClass'].fillna("None")



train_na = train.isnull().sum()
# print(train_na[train_na > 0].index)
#'MasVnrArea'

test_na = test.isnull().sum()
# print(test_na[test_na > 0].index)
#'MasVnrArea'

#MSSubClass=The building class
train['MSSubClass'] = train['MSSubClass'].apply(str)
test['MSSubClass'] = test['MSSubClass'].apply(str)


#Changing OverallCond into a categorical variable
train['OverallCond'] = train['OverallCond'].astype(str)
test['OverallCond'] = test['OverallCond'].astype(str)


#Year and month sold are transformed into categorical features.
train['YrSold'] = train['YrSold'].astype(str)
train['MoSold'] = train['MoSold'].astype(str)
test['YrSold'] = test['YrSold'].astype(str)
test['MoSold'] = test['MoSold'].astype(str)


from sklearn.preprocessing import LabelEncoder
cols = ['FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond',
        'ExterQual', 'ExterCond','HeatingQC', 'PoolQC', 'KitchenQual', 'BsmtFinType1',
        'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope',
        'LotShape', 'PavedDrive', 'Street', 'Alley', 'CentralAir', 'MSSubClass', 'OverallCond',
        'YrSold', 'MoSold']

# process columns, apply LabelEncoder to categorical features
for c in cols:
    lbl = LabelEncoder()
    lbl.fit(list(train[c].values) + list(test[c].values))
    train[c] = lbl.transform(list(train[c].values))
    test[c] = lbl.transform(list(test[c].values))

# shape
print('Shape train: {}'.format(train.shape))

train['TotalSF'] = train['TotalBsmtSF'] + train['1stFlrSF'] + train['2ndFlrSF']
test['TotalSF'] = test['TotalBsmtSF'] + test['1stFlrSF'] + test['2ndFlrSF']



from sklearn.preprocessing import LabelEncoder

cols = list(train.dtypes[train.dtypes == "object"].index) + list(train.dtypes[train.dtypes == "object"].index)

# process columns, apply LabelEncoder to categorical features
for c in cols:
    lbl = LabelEncoder()
    lbl.fit(list(train[c].values) + list(test[c].values))
    train[c] = lbl.transform(list(train[c].values))
    test[c] = lbl.transform(list(test[c].values))

# shape
print('Shape train: {}'.format(train.shape))


y = train['SalePrice']
X = train.drop(['SalePrice'],axis=1)

from sklearn.model_selection import train_test_split
X_train, X_valid, y_train, y_valid = train_test_split(X, y, random_state=0, test_size=0.3)


from lightgbm import LGBMRegressor

# LGBMClassifier 모델 선언 후 Fitting
model = LGBMRegressor()
model.fit(X_train, y_train)


pred_train = model.predict(X_train)
y_train = np.expm1(y_train)
pred_train = np.expm1(pred_train)
(pred_train == y_train).mean()




from sklearn.metrics import mean_squared_error
def rmsle(y, y_pred):
    return np.sqrt(mean_squared_error(y, y_pred))
rmsle(pred_train,y_train)


pred_valid = model.predict(X_valid)
y_valid = np.expm1(y_valid)
pred_valid = np.expm1(pred_valid)

rmsle(y_valid, pred_valid)


import matplotlib.pyplot as plt ########

temp_df = pd.DataFrame(y_train)
temp_df['pred'] = pred_train
temp_df = temp_df.reset_index(drop=True)

plt.figure(figsize = (100,8))    #########
plt.plot(temp_df['SalePrice'],color='gray',alpha=0.9)
plt.plot(temp_df['pred'],color='red',alpha=0.9)





import matplotlib.pyplot as plt ########

temp_df = pd.DataFrame(y_valid)
temp_df['pred'] = pred_valid
temp_df = temp_df.reset_index(drop=True)

plt.figure(figsize = (100,8))    #########
plt.plot(temp_df['SalePrice'],color='gray',alpha=0.9)
plt.plot(temp_df['pred'],color='red',alpha=0.9)

pred_test = model.predict(test)

sample_submission['SalePrice'] = pred_test

sample_submission.to_csv("sample_submission_final.csv",index=False)










