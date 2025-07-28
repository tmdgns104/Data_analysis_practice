import tensorflow as tf
from tensorflow import keras
import os, re
from PIL import Image
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import efficientnet.tfkeras as efn


# print(os.listdir(r'D:\Data_Analysis\Tenssorflow\KAGGLE'))
# Kaggle -> Input이라는 폴더에 들어있는 데이터 리스트를 출력


# Train 폴더에 각 이미지 Class 별로 또다른 폴더가 있음, 그 폴더속에 이미지가 들어있음
# 전체 리스트를 얻기 위해서는 각 폴더의 리스트들을 불러서 하나의 리스트로 합쳐줘야 함
# Train 폴더 > Class 폴더 > 이미지
train_files_path = os.path.join(r'D:\Data_Analysis\Tenssorflow\KAGGLE\train')  # 폴더 경로를 입력
train_files = os.listdir(train_files_path)  # 해당 폴더 경로에 들어있는 데이터 리스트를 집어넣음
# print(train_files)

train_all_list = []  # 비어있는 List를 만듦

for train_file in train_files:  # 위에 Train 폴더의 들어있는 Class 폴더 리스트를 For문으로 돌림
    temp_path = os.path.join(train_files_path, train_file)  # Train폴더 경로 + Class 폴더명으로 새로운 경로를 만듦
    for i in os.listdir(temp_path):  # 새로운 Class 폴더 경로내에 폴더들의 리스트를 For문으로 돌림
        train_all_list.append(i)  # Class 폴더내에 파일들을 리스트에 추가시킴 --> Class 폴더를 돌며 반복

# print(len(train_all_list))
# print(train_all_list[:10])  # 각각의 Class에 있는 파일들을 하나의 리스트로 만듦

# print(pd.DataFrame(["_".join(i.split('_')[:-1]) for i in train_all_list]))
# print(pd.DataFrame(["_".join(i.split('_')[:-1]) for i in train_all_list])[0].value_counts())

# 위와 같은 이유로 Valid 폴더내의 Class 폴더내의 이미지들의 리스트르 만듦
# Valid 폴더 > Class 폴더 > 이미지

valid_files_path = os.path.join(r'D:\Data_Analysis\Tenssorflow\KAGGLE\valid')
valid_files = os.listdir(valid_files_path)

valid_all_list = []

for valid_file in valid_files:
    temp_path = os.path.join(valid_files_path, valid_file)
    for i in os.listdir(temp_path):
        valid_all_list.append(i)

# print(len(valid_all_list))
# print(valid_all_list[:10])  # 각각의 Class에 있는 파일들을 하나의 리스트로 만듦


image_files = train_all_list + valid_all_list
# print(len(image_files))
# print(image_files[:10])

# print(pd.DataFrame(["_".join(i.split('_')[:-1]) for i in image_files])[0].value_counts())






#Preprocessing
# #=== 이미지들의 Class를 정리함 ===#
class_list = []
for image_file in image_files:
    file_name = os.path.splitext(image_file)[0] #확장자 분리후 이름만 남김
    class_name = re.sub('_\d+', '', file_name) #정규표현식으로 뒤에 숫자 날림 정규표현식,뭐로 바꿀지,뭐가 바뀔지
    class_list.append(class_name)

class_list = list(set(class_list))
class_list.sort()
print(class_list)
#중복제거후 정렬 = 결국 어떤 종류의 강아지가 있는지만 남음

#=== 향후 argmax를 위한 Class 정리하기 ===#
# argmax는 딥러닝 마지막에 Softmax함수를 통해 Class별로 확률적으로 값이 나오는데 그걸 그냥 1개의 값으로 만들어 줌
# (example) 이미지 -> 딥러닝(CNN) -> Softmax -> Output1: [0.01, 0.01, 0.13, 0.6, 0.12, ... ] -> Argmax -> Output2: Class[3] ->

#=== Class Name을 Index로 Mapping하기 위한 사전작업 ===#
last_class_name = ''
class_new_list = []

for class_name in class_list:
    if last_class_name == class_name:
        pass
    else:
        class_new_list.append(class_name)
        last_class_name = class_name
#이미 중복 제거 했는데 또 하는 이유는 모르겠음


class2idx = {cls:idx for idx, cls in enumerate(class_new_list)}
#강아지 명마다 인덱스를 알려주는 딕셔너리 만듦





#=== Hyper Parameters ===#
n_train = 0

N_CLASS = len(class_list)
#클래스 수, 분류할 대상의 총 클래스 수 → 모델 출력층(neurons) 개수에 사용
N_EPOCHS = 128    # 조금 더 길게 Training
#학습 반복 횟수, 전체 학습 데이터셋을 몇 번 반복할지 결정 (크면 오래 학습함)
N_BATCH = 32
#배치 크기, 한 번에 모델에 넣는 이미지 수. 클수록 병렬 연산이 많아짐 (메모리 사용량도 증가)

N_TRAIN = len(train_all_list)
#학습 데이터 수, 학습에 사용되는 이미지 개수
N_VAL = len(valid_all_list)
#검증 데이터 수, 검증(validation)에 사용되는 이미지 개수

IMG_SIZE = 224
# 입력 이미지 크기, 대부분의 이미지 모델 (특히 VGG, ResNet)은 224×224 크기를 사용
learning_rate = 0.0001
# 학습률, 가중치를 얼마나 크게 조정할지 결정. 너무 크면 발산, 너무 작으면 느림
steps_per_epoch = N_TRAIN // N_BATCH
validation_steps = int(np.ceil(N_VAL / N_BATCH))   # validation은 남은것 까지 다 쓰기 위함








# === 학습용 이미지 전처리기 정의 ===
# 이미지 픽셀을 0~1 범위로 정규화하여 학습 안정성과 속도를 향상
train_datagen = ImageDataGenerator(rescale=1. / 255)

# === 전처리기와 디렉토리 경로를 사용하여 배치 단위 데이터 생성기 만들기 ===
train_generator = train_datagen.flow_from_directory( train_files_path,         # 클래스별 하위 폴더를 가진 이미지 폴더 경로
                                                     batch_size=N_BATCH,       # 한 번에 모델에 공급할 이미지 개수 (배치 크기)
                                                     target_size=(224, 224),   # 모든 이미지를 지정된 크기로 리사이징
                                                     class_mode='categorical'  # 다중 클래스(one-hot 인코딩) 분류용 라벨 반환
                                                    )


# === valid 이미지를 변형시킬 내용을 ImageDataGenerator에 작성해줌 ===#
valid_datagen = ImageDataGenerator(rescale=1. / 255)

# === 위에서 만든 ImageDataGenerator(변형)을 사용해서 Batch 별로 출력될 수 있게 valid_generator를 만듦 ===#
valid_generator = valid_datagen.flow_from_directory(valid_files_path,
                                                    batch_size=N_BATCH,
                                                    target_size=(224, 224),
                                                    class_mode='categorical',  # binary / categorical
                                                    )

#Modeling
#이미지 데이터 알고리즘 efn
#전이 학습(Transfer Learning)
#Keras Sequential 모델
#Conv2D(3채널) → 입력에 맞춰 기본 필터 적용 (정규화나 커스텀 특징 추출 목적일 수도)
# EfficientNetB0 (Imagenet pretrained) → 이미지넷 사전학습된 CNN 백본
# GlobalAveragePooling2D → 공간 차원 제거, 특징 벡터로 변환
# Dropout + BatchNorm → 과적합 방지 + 정규화
# Dense(N_CLASS, softmax) → 클래스 분류



efficientnet = efn.EfficientNetB0(weights='imagenet',#이미지넷에서 학습한 걸 불러넴
                                  input_shape=(IMG_SIZE, IMG_SIZE, 3),   #  ★ 최적의 Image Size를 찾아보아라
                                  include_top=False)


model = tf.keras.Sequential()

# 1. 기본 Conv2D 레이어 (입력 정리 및 채널 맞춤용)
model.add(tf.keras.layers.Conv2D(3, (3, 3),  padding='same', strides=(1, 1), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)))

# 2. EfficientNetB0 백본 모델 (Top 제외)
#Feature Extractor 역할
#include_top=False → Fully Connected 층(1000-class softmax)은 제거하고, feature map만 사용
model.add(efficientnet)

# 3. 글로벌 평균 풀링 (Flatten보다 효과적)
#2D feature map을 1D 벡터로 바꿔 Dense에 넣기 쉽게 함
# 파라미터 없음 → 가볍고 과적합 위험도 적음
model.add(tf.keras.layers.GlobalAveragePooling2D())

# 4. Dropout & BatchNormalization (정규화와 과적합 방지)
#Dropout(0.1): 학습 중 일부 뉴런 무작위 제거 → 일반화 향상
# BatchNorm: 내부 공변량 변화 감소 → 학습 안정화
model.add(tf.keras.layers.Dropout(0.1))
model.add(tf.keras.layers.BatchNormalization())

# 5. 최종 출력층: softmax 다중 분류
#N_CLASS: 분류할 클래스 개수
# 출력은 [0.05, 0.01, 0.9, ...]처럼 확률값으로 나옴
model.add(tf.keras.layers.Dense(N_CLASS, activation='softmax'))

print(model.summary())


#=== Compile을 통해 Optimizer, loss, metric을 설정해 줌 ===#
#컴파일 단계에서 옵티마이저, 손실 함수, 평가 지표를 설정
LR_INIT = 0.000001 ## 초기 학습률 (Learning Rate), 작은 학습률 = 안정적인 학습, 하지만 느릴 수 있음, 사전학습된 모델(EfficientNet 등)을 사용할 땐 1e-5나 1e-6처럼 작은 값이 일반적
model.compile(optimizer=tf.keras.optimizers.Adam(LR_INIT), #Adam Optimizer는 학습률을 자동 조절하는 강력한 옵티마이저
              loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1), #다중 클래스 분류에서 사용하는 손실 함수, label_smoothing=0.1은 정답 레이블의 확신도를 100% → 90% 정도로 줄여줌, 모델이 과도하게 특정 클래스만 예측하는 걸 방지해서 일반화 성능 향상
              metrics=['accuracy']) #학습과 평가 시 정확도(accuracy)를 보여줌

#모델의 구조와 파라미터 수를 확인하는 함수
model.summary()



#Training

#=== Training the Model ===#
history = model.fit(train_generator,             # 학습 데이터 제너레이터 (이미지 + 라벨)
                    epochs=N_EPOCHS,             # 전체 학습 반복 횟수
                    steps_per_epoch=steps_per_epoch,   # 한 에폭에 몇 배치 처리할지
                    validation_data=valid_generator,   # 검증 데이터 제너레이터
                    validation_steps=validation_steps, # 검증에서 몇 배치 평가할지
                    # callbacks=[lr_callback]     # (선택) 콜백 함수 예: 학습률 조절
)

#=== Save Trained-Model ===#
model.save('efficientnet_b0_model.h5')

#=== Load Trained Model ===#
new_model = keras.models.load_model('efficientnet_b0_model.h5')



#Inferencing
img_file_path_list = []

img_name_list = os.listdir(r'D:\Data_Analysis\Tenssorflow\KAGGLE\dogcatsample')  # 폴더 내 파일 이름 목록 가져오기
for img_name in img_name_list:
    img_file_path = os.path.join(r'D:\Data_Analysis\Tenssorflow\KAGGLE\dogcatsample', img_name)  # 전체 경로 만들기
    img_file_path_list.append(img_file_path)

print(img_file_path_list)  # 모든 이미지 경로가 담긴 리스트 출력

for img_path in img_file_path_list:
    # === Image upload 후 실행 ===#
    image = Image.open(img_path)  # 구글에서 다운받아서 raw directory 폴더에 넣고 돌리면 됨
    image = image.resize((224, 224))
    image = np.array(image)
    image = image / 255.

    plt.imshow(image)
    plt.show()
    print(image.shape)

    image = np.array([image])

    # === Predict ===#
    prediction = new_model.predict(image)
    pred_class = np.argmax(prediction, axis=-1)  # argmax를 하면 앞에서 OHE로 나온 확률에 대한 class가 나옴
    print(pred_class)
    print(class_list[int(pred_class)])
