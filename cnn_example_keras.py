"""
CNN 간단 예제 - Keras/TensorFlow 버전
MNIST 손글씨 숫자 인식 (0-9)

실행 방법:
pip install tensorflow numpy matplotlib
python cnn_example_keras.py
"""

import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers

print("="*50)
print("CNN 간단 예제 - MNIST 손글씨 숫자 인식")
print("="*50)

# 1. 데이터 로드
print("\n[1단계] 데이터 로딩...")
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
print(f"학습 데이터: {x_train.shape}, 레이블: {y_train.shape}")
print(f"테스트 데이터: {x_test.shape}, 레이블: {y_test.shape}")

# 2. 데이터 전처리
print("\n[2단계] 데이터 전처리...")
# 28x28 이미지를 28x28x1로 reshape (채널 추가)
x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0

# 레이블을 one-hot encoding
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)
print("전처리 완료!")

# 3. CNN 모델 구축
print("\n[3단계] CNN 모델 구축...")
model = keras.Sequential([
    # 첫 번째 합성곱 블록
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),

    # 두 번째 합성곱 블록
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    # 세 번째 합성곱 블록
    layers.Conv2D(64, (3, 3), activation='relu'),

    # 완전 연결 층
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax')
])

# 모델 구조 출력
model.summary()

# 4. 모델 컴파일
print("\n[4단계] 모델 컴파일...")
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# 5. 모델 학습
print("\n[5단계] 모델 학습 시작...")
print("(3 epochs만 학습 - 빠른 데모용)")
history = model.fit(
    x_train, y_train,
    batch_size=128,
    epochs=3,
    validation_split=0.1,
    verbose=1
)

# 6. 모델 평가
print("\n[6단계] 모델 평가...")
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print(f"\n테스트 정확도: {test_acc*100:.2f}%")
print(f"테스트 손실: {test_loss:.4f}")

# 7. 예측 예시
print("\n[7단계] 실제 예측 테스트...")
# 테스트 데이터에서 랜덤으로 5개 선택
indices = np.random.choice(len(x_test), 5, replace=False)

plt.figure(figsize=(15, 3))
for i, idx in enumerate(indices):
    # 예측
    prediction = model.predict(x_test[idx:idx+1], verbose=0)
    predicted_digit = np.argmax(prediction)
    true_digit = np.argmax(y_test[idx])
    confidence = np.max(prediction) * 100

    # 시각화
    plt.subplot(1, 5, i+1)
    plt.imshow(x_test[idx].reshape(28, 28), cmap='gray')
    plt.title(f'예측: {predicted_digit}\n정답: {true_digit}\n확률: {confidence:.1f}%')
    plt.axis('off')

    print(f"샘플 {i+1}: 정답={true_digit}, 예측={predicted_digit}, "
          f"확률={confidence:.1f}% {'✓' if predicted_digit==true_digit else '✗'}")

plt.tight_layout()
plt.savefig('cnn_predictions.png', dpi=100, bbox_inches='tight')
print("\n결과 이미지 저장: cnn_predictions.png")

# 8. 학습 곡선 시각화
print("\n[8단계] 학습 곡선 저장...")
plt.figure(figsize=(12, 4))

# 정확도
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='학습 정확도')
plt.plot(history.history['val_accuracy'], label='검증 정확도')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('모델 정확도')
plt.grid(True)

# 손실
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='학습 손실')
plt.plot(history.history['val_loss'], label='검증 손실')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('모델 손실')
plt.grid(True)

plt.tight_layout()
plt.savefig('cnn_training_history.png', dpi=100, bbox_inches='tight')
print("학습 곡선 저장: cnn_training_history.png")

print("\n" + "="*50)
print("실행 완료!")
print("="*50)
print("\n생성된 파일:")
print("- cnn_predictions.png : 예측 결과")
print("- cnn_training_history.png : 학습 곡선")
