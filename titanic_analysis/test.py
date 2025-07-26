import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import warnings

# 2. Load and check data
#  2.1 Load data
for dirpath, dirnames, filenames in os.walk('D:\\Data_Analysis\\titanic_analysis\\titanic'):
    for filename in filenames:
        print(os.path.join(dirpath, filename))


sns.set(style='white', context='notebook', palette='deep')
warnings.filterwarnings("ignore")



# Load data
##### Load train and Test set

train = pd.read_csv(r"D:\Data_Analysis\titanic_analysis\titanic\train.csv")
test = pd.read_csv(r"D:\Data_Analysis\titanic_analysis\titanic\test.csv")
gender_submission = pd.read_csv(r"D:\Data_Analysis\titanic_analysis\titanic\gender_submission.csv")
print(train.shape, test.shape)

#  2.1 Null값 또는 비어있는 값 서치¶
#print(train.isnull().sum()) #null 이 아닌 부분 Age, Cabin, Embarked , survived는 생존 여부
#print(test.isnull().sum())  # Age, Fare, Cabin
# 나이와 객실 특성은 결측값의 중요한 부분을 차지합니다.
# 생존 여부의 결측값은 조인 테스트 데이터셋에 해당합니다 (생존 여부 열은 test dataset에 존재하지 않습니다.

### Summarize data
# Summarie and statistics
#print(train.describe())



# 3. Feature analysis
#  3.1 Numerical values

#train[["Survived","SibSp","Parch","Age","Fare"]] # 지정한 컬럼("Survived","SibSp","Parch","Age","Fare")항목만 보고 싶음
#train[["Survived","SibSp","Parch","Age","Fare"]].corr() 특정 컬럼만으로 상관계수를 판단
# Correlation matrix between numerical values (SibSp Parch Age and Fare values) and Survived
#sns.heatmap(train[["Survived","SibSp","Parch","Age","Fare"]].corr(),annot=True, fmt = ".2f", cmap = "coolwarm")
#sns.heatmap을 통해서 아까 파악한 상관계수로 그래프를 그림 상관게수 행렬,숫자로 표시,소수점 2자리로 표시,(푸른색=음,붉은색=양)
#plt.show()
#상기 그래프는 지표들간의 연관성을 나타냄 그중 가장 survived와 연관성이 있는 건 Fare 였음
#Fare = 티켓값 이 높을수록 생존률이 높은걸 알수 있음
# 오직 운임(Fare) 특성만이 생존 확률과 유의미한 상관 관계를 가지는 것으로 보입니다.
# 다른 특성들이 유용해보이지 않습니다. 이러한 특성들은 뭔가 다른 세분화 작업을 하면 생존과 연관될 수 있습니다. 이를 확인하기 위해 이러한 특성들을 자세히 탐색해야 합니다.




# Parch 특성과 Survived 간의 관계 탐색

# g = sns.catplot(x="SibSp", y="Survived", data=train, kind="bar", height=6, palette="muted")
# # x축설정,y축설정,막대그래프,그래프높이지정(인치),색상지정(부드러운색)
# g.despine(left=True)
# # 왼쪽태두리(축)를 제거
# g.set_ylabels("Survival")

#Sibsp 와 Survived의 상관관계를 막대그래프로 보여줌
# 형제/배우자가 많은 승객들은 생존 확률이 적어 보입니다.
# 혼자 탑승한 승객들(0 SibSP) 또는 두 명과 함께 탑승한 경우(SibSP 1 또는 2) 생존 확률이 높습니다.
# 이 관찰은 매우 흥미로운데, 우리는 이러한 카테고리를 설명하는 새로운 특성을 고려할 수 있습니다(feature engineering에 참고).

# Parch 특성과 Survived 간의 관계 탐색
# g = sns.catplot(x="Parch", y="Survived", data=train, kind="bar", height=6, palette="muted")
# g.despine(left=True)
# g.set_ylabels("Survival")
# plt.show()

# 작은 가족들은 Parch 값이 0인 혼자인 경우보다, Parch 값이 3 또는 4인 중간 규모 가족 및 Parch 값이 5 또는 6인 큰 규모 가족들보다 생존할 기회가 더 많습니다.
# 3명의 부모/자녀를 가진 승객들의 생존율에 큰 표준 편차가 있습니다.





# Explore Age vs Survived
# g = sns.FacetGrid(train, col='Survived')
# g = g.map(sns.distplot, "Age")
# plt.show()
#생존한 사람과 생존하지 못한 사람을 나이에 따른 분포를 보여줌
# 나이 분포는 꼬리 분포인 것 같으며, 아마 가우시안 분포일 수 있습니다.
# 생존한 그룹과 생존하지 못한 그룹에서 나이 분포가 다른 것을 알 수 있습니다. 실제로, 생존한 젊은 승객들에 해당하는 피크가 있습니다. 또한 60-80세 승객들은 생존률이 적은 것을 볼 수 있습니다.
# 그래서 "Age"가 "Survived"과 상관관계가 없더라도, 우리는 승객들의 나이 범주에 따라 더 많거나 더 적은 생존 기회가 있다는 것을 볼 수 있습니다.
# 매우 어린 승객들이 더 많은 생존 기회를 가지고 있는 것으로 보입니다.




# # Explore Age distibution
# g = sns.kdeplot(train["Age"][(train["Survived"] == 0) & (train["Age"].notnull())], color="Red", shade = True)
# g = sns.kdeplot(train["Age"][(train["Survived"] == 1) & (train["Age"].notnull())], ax =g, color="Blue", shade= True)
# # ax=g 가 기존 그래프 위에 겹쳐 그리겠다는 의미
# g.set_xlabel("Age")
# g.set_ylabel("Frequency")
# g = g.legend(["Not Survived","Survived"])
# plt.show()
# 두 개의 밀도를 겹쳐 놓을 때, 우리는 분명히 아기와 매우 어린 아이들에 해당하는 (0과 5 사이) 뾰족한 정점을 명확히 볼 수 있습니다.


train["Fare"].isnull().sum()

#Fill Fare missing values with the median value
#null 값을 중앙값으로 채우기로함
# 하나의 결측치가 있기 때문에, 해당 값을 중앙값으로 채우기로 결정했습니다. 이는 예측에 중요한 영향을 미치지 않을 것입니다
train["Fare"] = train["Fare"].fillna(train["Fare"].median())
test["Fare"] = test["Fare"].fillna(train["Fare"].median())

# Explore Fare distribution
# sns.distplot(train["Fare"])
# plt.show()
#매우 비대칭 적이기 때문에 이 비대칭성을 줄이기 위해 로그 함수로 변환하는 것이 좋음
#하나의 결측치가 있기 때문에, 해당 값을 중앙값으로 채우기로 결정했습니다. 이는 예측에 중요한 영향을 미치지 않을 것입니다

# Apply log to Fare to reduce skewness distribution
train["Fare"] = np.log1p(train["Fare"])
test["Fare"] = np.log1p(test["Fare"])
# 우리가 볼 수 있듯이, 요금 분포는 매우 비대칭적입니다. 이는 모델에서 매우 높은 값들을 가중시킬 수 있으며, 스케일링을 하더라도 그렇습니다.
# 이 경우에는 이 비대칭성을 줄이기 위해 로그 함수로 변환하는 것이 좋습니다.

# # Explore Fare distribution
# sns.distplot(train["Fare"])
# plt.show()



# 3.2 Categorical values
# Sex

# 남성은 여성보다 생존 확률이 현저히 낮다는 사실은 명백합니다.
# 따라서 성별은 생존 예측에 중요한 역할을 할 수 있습니다.
# 타이타닉 영화(1997년)를 본 적이 있는 분들은 모두 기억하시겠지만, 대피 과정에서 이 문구가 자주 나왔습니다: '여성과 아이들 먼저'

# g = sns.barplot(x="Sex",y="Survived",data=train)
# g = g.set_ylabel("Survival Probability")
# plt.show()


train[["Sex","Survived"]].groupby('Sex').mean() # 그룹별로 평균을 냄

# 남성은 여성보다 생존 확률이 현저히 낮다는 사실은 명백합니다.
# 따라서 성별은 생존 예측에 중요한 역할을 할 수 있습니다.
# 타이타닉 영화(1997년)를 본 적이 있는 분들은 모두 기억하시겠지만, 대피 과정에서 이 문구가 자주 나왔습니다: '여성과 아이들 먼저'


# Pclass vs Survived 탐색하기
# g = sns.barplot(x="Pclass", y="Survived", data=train, ci=None, palette="muted")
# g.set_ylabel("survival")
# g.set_xlabel("pclass")
# plt.show()



# 타이타닉 데이터셋에서 Pclass와 Survived 간의 성별에 따른 탐색
# g = sns.barplot(x="Pclass", y="Survived", hue="Sex", data=train, ci=None, palette="muted")
# g.set_ylabel("survival")
# g.set_xlabel("class")
# plt.show()

# 3개의 클래스에서 승객의 생존율은 같지 않습니다. 일등석 승객들은 이차석과 삼등석 승객들보다 생존할 확률이 더 높습니다.
# 이러한 경향은 남성과 여성 승객 모두를 살펴봤을 때도 유지됩니다.




train["Embarked"].isnull().sum()
#Fill Embarked nan values of dataset set with 'S' most frequent value
train["Embarked"] = train["Embarked"].fillna("S")
# 우리는 두 개의 누락된 값이 있기 때문에, "Embarked" 변수의 가장 빈번한 값인 "S"로 그 값을 채우기로 결정했습니다.

# "Embarked"와 "Survived" 사이의 관계 탐색
# g = sns.catplot(x="Embarked", y="Survived", data=train,
# height=6, kind="bar", palette="muted")
# g.despine(left=True)
# g = g.set_ylabels("survival")
# plt.show()
# 쉐르부르(C)에서 탑승한 승객들은 생존 확률이 높아 보입니다.
# 내 가설은 쉐르부르(C)에서 온 승객들 중 일등석 승객 비율이 퀸스타운(Q)과 사우스햄튼(S)에서 온 승객들보다 높을 것이라는 것입니다.
# Pclass(승객 등급) 분포와 Embarked(승선한 항구)를 비교해 봅시다


# Pclass vs Embarked 탐색하기
# g = sns.catplot(x="Pclass", col="Embarked", data=train,
# height=6, kind="count", palette="muted")
# g.despine(left=True)
# g.set_ylabels("Count")
# plt.show()
# 실제로, 세 번째 클래스는 사우스햄튼(S)과 퀸스타운(Q)에서 탑승한 승객들에게 가장 빈번하게 나타납니다. 반면에 쉐르부르(Cherbourg) 승객들은 대부분 첫 번째 클래스에 속하며, 이들이 가장 높은 생존률을 보입니다.
# 이 시점에서, 왜 첫 번째 클래스가 더 높은 생존률을 가지는지 설명할 수 없습니다. 가설로는 첫 번째 클래스 승객들이 탈출 과정에서 우선적으로 대우받았을 가능성이 있습니다.



# 4. Filling missing Values
# 4.1 Age

# 우리가 볼 수 있듯이, 전체 데이터셋에서 Age 열에는 256개의 결측값이 있습니다.
# 생존 확률이 높은 하위 그룹(예: 어린이)이 있기 때문에, 나이 특성을 유지하고 결측값을 대체하는 것이 좋습니다.
# 이 문제를 해결하기 위해 나이와 가장 상관 관계가 있는 특성들(Sex, Parch, Pclass 및 SibSP)을 살펴보았습니다.

# Age vs Sex
# g = sns.catplot(y="Age", x="Sex", data=train, kind="box")
# plt.show()

# Age vs Sex with Pclass as hue
# g = sns.catplot(y="Age", x="Sex", hue="Pclass", data=train, kind="box")
# plt.show()

# Age vs Parch
# g = sns.catplot(y="Age", x="Parch", data=train, kind="box")
# plt.show()


# Age vs SibSp
# g = sns.catplot(y="Age", x="SibSp", data=train, kind="box")
# plt.show()

# 나이 분포는 남성과 여성 하위 모집단에서 동일해 보입니다. 따라서 성별은 나이를 예측하는 데 유용하지 않습니다.
# 그러나 1등급 승객은 2등급 승객보다 더 나이가 많으며, 2등급 승객은 3등급 승객보다 더 나이가 많습니다.
# 또한, 승객이 부모/자녀를 더 많이 가지고 있을수록 나이가 더 많으며, 승객이 형제자매/배우자를 더 많이 가지고 있을수록 더 어립니다.

# convert Sex into categorical value 0 for male and 1 for female
train["Sex"] = train["Sex"].replace("male",0).replace("female",1)
test["Sex"] = test["Sex"].replace("male",0).replace("female",1)
#머신 러닝을 하기 위해서는 문자를 숫자로 바꿔줄 필요가 있슴

train_corr = train[["Age","Sex","SibSp","Parch","Pclass"]].corr()
#g = sns.heatmap(train_corr,cmap="BrBG",annot=True)


# 5. Feature engineering
# 5.1 Name/Title
# print(train["Name"])
# # Name 에는 승객의 호칭(Initial)에 대한 정보가 포함되어 있습니다.
# # Age에서 비어있는 값을 추정해서 집어넣을 수 있을겁니다.
#
#
# print(train.isnull().sum())
train['Initial'] = train['Name'].apply(lambda x:x.split(',')[1].split('.')[0].strip())
# train_name의 column의 개개의 x들을 -> ","기준으로 나눈것의 1번째 -> "."기준으로 나눈것의 0번째 -> 앞뒤 빈칸을 제거
# print(train['Initial'])


#print(train['Initial'].value_counts())
# group_initial = train.groupby(['Initial'])
# print(group_initial)
# print(group_initial['Age'])
# print(group_initial['Age'].mean())

train_initial_age_dict = train.groupby(['Initial'])['Age'].mean().to_dict()
#이니셜로 그룹별로 묶고 그중 나이의 평균을 딕셔너리로 표현해라
# print(train_initial_age_dict)
train_age_null_index = train[train['Age'].isnull()].index
train.loc[train_age_null_index,'Age'] = train.loc[train_age_null_index,'Initial'].map(train_initial_age_dict)
#train의 나이가 빈값(row)중 나이부분(colum)을=  train의 빈곳중 이니셜로 딕셔너리랑 일치 하는 숫자로 대체



# print(test.isnull().sum())
test['Initial'] = test['Name'].apply(lambda x:x.split(',')[1].split('.')[0].strip())
test_initial_age_dict = test.groupby(['Initial'])['Age'].mean().to_dict()
test_age_null_index = test[test['Age'].isnull()].index
test.loc[test_age_null_index,'Age'] = test.loc[test_age_null_index,'Initial'].map(test_initial_age_dict)
# print(test.isnull().sum()) #호칭이 없어서 NAN으로 남은게 있음




all_initial = list(train['Initial']) + list(test['Initial'])
all_initial_dict = {u:i for i,u in enumerate(set(all_initial))}
#모든 호칭들을 모으고 중복을 없앰, 중복이 없는 호칭들에 숫자를 매김 dict형태


train['Initial'] = train['Initial'].map(all_initial_dict)
test['Initial'] = test['Initial'].map(all_initial_dict)
#train,test의 inital을 문자열에서 아까 숫자로 바꿔줌


#initial 별로 묶어서 생존륭의 평균을 구함
# print(all_initial_dict)
# print(train.groupby(['Initial'])['Survived'].mean())
# "여성과 아이들 먼저"
# 흥미로운 점은 희귀한 호칭을 가진 승객들이 생존할 확률이 더 높다는 것입니다.


# Drop Name variable
train = train.drop(['Name'], axis = 1)
# Drop Name variable
test = test.drop(['Name'], axis = 1)
#호칭은 사용했으니 이름 날려버림



# 5.2 가족 규모
# 우리는 대가족은 대피 과정에서 형제, 자매, 부모님을 찾느라 어려움을 겪을 수 있다고 상상해볼 수 있습니다.
# 따라서 저는 "Fize" (가족 규모) 특성을 만들기로 결정했습니다.
# 이 특성은 SibSp와 Parch의 합 그리고 1(해당 승객 자신을 포함)로 구성됩니다.


# Create a family size descriptor from SibSp and Parch
train["Fsize"] = train["SibSp"] + train["Parch"] + 1
# Create a family size descriptor from SibSp and Parch
test["Fsize"] = test["SibSp"] + test["Parch"] + 1
# print(train["Fsize"]) # 형재 자매수 + 부모자녀수

# g = sns.catplot(x="Fsize", y="Survived", data=train, kind="bar")
# g = g.set_ylabels("Survival")
# plt.show()


all_Embarked = list(train['Embarked']) + list(test['Embarked'])
all_Embarked_dict = {u:i for i,u in enumerate(set(all_Embarked))}
# print(all_Embarked_dict)
train['Embarked'] = train['Embarked'].map(all_Embarked_dict)
test['Embarked'] = test['Embarked'].map(all_Embarked_dict)

# 5.3 Cabin
# print(train["Cabin"].head())
# print(train["Cabin"].describe()) #데이터의 갯수(결측값 제외),종류,가장많이 나온값,top의 빈도
#

# print(train["Cabin"].isnull().sum())
# Cabin(객실) 특성 열에는 292개의 값과 1007개의 결측값이 포함되어 있습니다.
# 객실이 없는 승객들은 객실 번호 대신에 결측값(X)이 표시된다고 가정합니다.
# 선실의 첫 글자는 승객의 타이타닉호 내 예상 위치를 나타내므로, 이 정보만을 선택하여 보존하기로 결정했습니다.


train["Cabin"] = train['Cabin'].fillna('X')
# print(train[train["Cabin"] == 'X'])

train["Cabin"] = train["Cabin"].apply(lambda x:x[0])
# print(train["Cabin"].value_counts())

test["Cabin"] = test['Cabin'].fillna('X')
test["Cabin"] = test["Cabin"].apply(lambda x:x[0])
# print(test["Cabin"].value_counts())



# g = sns.catplot(x="Cabin", y="Survived", data=train, kind="bar", order=['A', 'B', 'C', 'D', 'E', 'F', 'G', 'T', 'X'])
# g = g.set_ylabels("Survival")
# plt.show()


# 승객들 중에서 객실을 소유한 사람들의 수가 적어 생존 확률은 중요한 표준 편차를 가지고 있으며, 각 데스크(석사)에 있는 승객들의 생존 확률을 구별하기 어렵습니다.
# 하지만 우리는 객실을 소유한 승객들이 객실을 소유하지 않은 승객들보다 일반적으로 생존 확률이 높다는 것을 알 수 있습니다(X).
# 특히 B, C, D, E 및 F 객실의 경우 이러한 사실이 더욱 두드러지게 나타납니다.

all_Cabin = list(train['Cabin']) + list(test['Cabin'])
all_Cabin_dict = {u:i for i,u in enumerate(set(all_Cabin))}
train['Cabin'] = train['Cabin'].map(all_Cabin_dict)
test['Cabin'] = test['Cabin'].map(all_Cabin_dict)


#5.4 Ticket
# print(train["Ticket"].head())
    # 이것은 동일한 접두사를 공유하는 티켓들이 함께 배치된 객실을 예약할 수 있음을 의미할 수 있습니다. 따라서 이로 인해 실제로 객실이 배치될 수 있습니다.
    # 동일한 접두사를 가진 티켓들은 유사한 등급과 생존율을 가질 수 있습니다.
    # 그래서 나는 티켓의 특징 열을 티켓 접두사로 대체하기로 결정했습니다. 이것이 더 유익할 수 있습니다.

train['Ticket'] = train['Ticket'].apply(lambda x:x.replace(".","").replace("/","").split(" ")[0])
test['Ticket'] = test['Ticket'].apply(lambda x:x.replace(".","").replace("/","").split(" ")[0])
# print(train['Ticket'].value_counts())


all_Ticket = list(train['Ticket']) + list(test['Ticket'])
all_Ticket_dict = {u:i for i,u in enumerate(set(all_Ticket))}

train['Ticket'] = train['Ticket'].map(all_Ticket_dict)
test['Ticket'] = test['Ticket'].map(all_Ticket_dict)



train["Pclass"] = train["Pclass"].astype("category") #데이터 타입을 범주형으로 바꿈
test["Pclass"] = test["Pclass"].astype("category") #데이터 타입을 범주형으로 바꿈
all_Pclass = list(train['Pclass']) + list(test['Pclass'])
all_Pclass_dict = {u:i for i,u in enumerate(set(all_Pclass))}

train['Pclass'] = train['Pclass'].map(all_Pclass_dict)
test['Pclass'] = test['Pclass'].map(all_Pclass_dict)

# Drop useless variables
train = train.drop(["PassengerId"],axis =1)
test = test.drop(["PassengerId"],axis = 1)

train_isnull_sum = train.isnull().sum()
print(train_isnull_sum[train_isnull_sum > 0]) #결측치가 있는 것만 보이기
test_isnull_sum = test.isnull().sum()
print(test_isnull_sum[test_isnull_sum > 0]) #결측치가 있는 것만 보이기
print(test_isnull_sum)

y = train['Survived']
X = train.drop(['Survived'],axis = 1)

from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier
X_train, X_valid, y_train, y_valid = train_test_split(X,y,random_state=30,test_size=0.3)


model = LGBMClassifier()
model.fit(X_train, y_train)
pred_train = model.predict(X_train)
print((pred_train == y_train).mean())
pred_test = model.predict(test)
gender_submission['Survived'] = pred_test
gender_submission.to_csv('final_gender_submission.csv',index=False)