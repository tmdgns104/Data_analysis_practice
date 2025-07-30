import tensorflow as tf
from tensorflow import keras
import os, re
from PIL import Image
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import efficientnet.tfkeras as efn

# === 이미지 경로 설정 ===
train_files_path = os.path.join(r'D:\Data_Analysis\Tenssorflow\KAGGLE\train')  # 학습 데이터 폴더
valid_files_path = os.path.join(r'D:\Data_Analysis\Tenssorflow\KAGGLE\valid')  # 검증 데이터 폴더

# === 학습/검증 이미지 리스트 구성 ===
train_all_list = [img for cls in os.listdir(train_files_path)
                  for img in os.listdir(os.path.join(train_files_path, cls))]
valid_all_list = [img for cls in os.listdir(valid_files_path)
                  for img in os.listdir(os.path.join(valid_files_path, cls))]

# === 전체 이미지 파일 리스트로부터 클래스 이름 추출 ===
image_files = train_all_list + valid_all_list
class_list = list(sorted(set(re.sub('_\d+', '', os.path.splitext(f)[0]) for f in image_files)))

# 클래스 이름 → 인덱스 매핑 딕셔너리 생성
class2idx = {cls: idx for idx, cls in enumerate(class_list)}

# === 하이퍼파라미터 설정 ===
N_CLASS = len(class_list)              # 클래스 개수
N_EPOCHS = 128                         # 에폭 수
N_BATCH = 32                           # 배치 크기
N_TRAIN = len(train_all_list)         # 학습 이미지 수
N_VAL = len(valid_all_list)           # 검증 이미지 수
IMG_SIZE = 224                         # 이미지 크기 (224x224)
steps_per_epoch = N_TRAIN // N_BATCH
validation_steps = int(np.ceil(N_VAL / N_BATCH))

# === 이미지 전처리기 정의 ===
train_datagen = ImageDataGenerator(rescale=1. / 255)
valid_datagen = ImageDataGenerator(rescale=1. / 255)

# === 제너레이터 생성 (자동 배치 처리) ===
train_generator = train_datagen.flow_from_directory(train_files_path,
                                                    batch_size=N_BATCH,
                                                    target_size=(IMG_SIZE, IMG_SIZE),
                                                    class_mode='categorical')

valid_generator = valid_datagen.flow_from_directory(valid_files_path,
                                                    batch_size=N_BATCH,
                                                    target_size=(IMG_SIZE, IMG_SIZE),
                                                    class_mode='categorical')

# # === EfficientNetB0 (사전학습된 백본) 불러오기 ===
# efficientnet = efn.EfficientNetB0(weights='imagenet',
#                                   input_shape=(IMG_SIZE, IMG_SIZE, 3),
#                                   include_top=False)
#
# # === 모델 구성 ===
# model = tf.keras.Sequential()
# model.add(tf.keras.layers.Conv2D(3, (3, 3), padding='same', activation='relu',
#                                  input_shape=(IMG_SIZE, IMG_SIZE, 3)))  # RGB 채널 정렬용
# model.add(efficientnet)                                     # 특성 추출기
# model.add(tf.keras.layers.GlobalAveragePooling2D())         # 2D → 1D 변환
# model.add(tf.keras.layers.Dropout(0.1))                     # 과적합 방지
# model.add(tf.keras.layers.BatchNormalization())             # 정규화
# model.add(tf.keras.layers.Dense(N_CLASS, activation='softmax'))  # 클래스 예측
#
# # === 모델 컴파일 ===
# model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-6),
#               loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
#               metrics=['accuracy'])
#
# # === 학습 ===
# model.fit(train_generator,
#           epochs=N_EPOCHS,
#           steps_per_epoch=steps_per_epoch,
#           validation_data=valid_generator,
#           validation_steps=validation_steps)
#
# # === 모델 저장 ===
# model.save('efficientnet_b0_model.h5')

# === 모델 불러오기 ===
new_model = keras.models.load_model(r'D:\Data_Analysis\Tenssorflow\efficientnet_b0_model.h5')

# === 예측할 이미지 경로 리스트 준비 ===
sample_folder = r'D:\Data_Analysis\Tenssorflow\KAGGLE\dogcatsample'
img_file_path_list = [os.path.join(sample_folder, name) for name in os.listdir(sample_folder)]

# === 이미지 예측 수행 ===
for img_path in img_file_path_list:
    image = Image.open(img_path).convert("RGB")       # RGBA 이미지 → RGB 변환
    image = image.resize((224, 224))                  # 리사이즈
    image = np.array(image) / 255.0                   # 정규화
    image = np.expand_dims(image, axis=0)             # 배치 차원 추가: (1, 224, 224, 3)

    # 예측 수행
    prediction = new_model.predict(image)
    pred_class = np.argmax(prediction, axis=-1)

    # 시각화 및 출력
    plt.imshow(image[0])
    plt.axis('off')
    plt.title(f"Predicted: {class_list[int(pred_class)]}")
    plt.show()
    print("예측 클래스 번호:", pred_class[0])
    print("예측 클래스 이름:", class_list[int(pred_class)])
