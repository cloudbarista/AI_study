# CNN 간단 예제 실행 가이드

이 폴더에는 CNN(합성곱 신경망)을 간단히 실행해볼 수 있는 예제 코드가 포함되어 있습니다.

## 파일 구성

1. **CNN_core_principles.md** - CNN의 핵심 원리 이론 설명
2. **cnn_example_keras.py** - Keras/TensorFlow 버전 실습 코드
3. **cnn_example_pytorch.py** - PyTorch 버전 실습 코드

## 빠른 시작

### 1. Keras/TensorFlow 버전 (추천 - 더 간단함)

#### 설치
```bash
pip install tensorflow numpy matplotlib
```

#### 실행
```bash
python cnn_example_keras.py
```

#### 예상 소요 시간
- CPU: 약 3-5분
- GPU: 약 1-2분

#### 출력 결과
- 콘솔에 학습 과정 출력
- `cnn_predictions.png` - 예측 결과 이미지
- `cnn_training_history.png` - 학습 곡선 그래프

---

### 2. PyTorch 버전

#### 설치
```bash
pip install torch torchvision numpy matplotlib
```

#### 실행
```bash
python cnn_example_pytorch.py
```

#### 예상 소요 시간
- CPU: 약 3-5분
- GPU: 약 1-2분

#### 출력 결과
- 콘솔에 학습 과정 출력
- `cnn_predictions_pytorch.png` - 예측 결과 이미지
- `cnn_training_history_pytorch.png` - 학습 곡선 그래프
- `cnn_mnist_model.pth` - 학습된 모델 파일

---

## 예제 설명

### 데이터셋: MNIST
- 손글씨 숫자 이미지 (0-9)
- 학습 데이터: 60,000장
- 테스트 데이터: 10,000장
- 이미지 크기: 28×28 픽셀 (흑백)

### 모델 구조
```
입력 (28×28×1)
   ↓
Conv2D (32 필터, 3×3) + ReLU
   ↓
MaxPooling (2×2)
   ↓
Conv2D (64 필터, 3×3) + ReLU
   ↓
MaxPooling (2×2)
   ↓
Conv2D (64 필터, 3×3) + ReLU
   ↓
Flatten
   ↓
Dense (64) + ReLU + Dropout(0.5)
   ↓
Dense (10) + Softmax
   ↓
출력 (10개 클래스 확률)
```

### 예상 성능
- 3 epochs 학습 후: 약 98% 정확도
- 10 epochs 학습 시: 약 99% 정확도

---

## 코드 주요 단계

### Keras 버전
1. **데이터 로드**: MNIST 데이터셋 자동 다운로드
2. **전처리**: 정규화 (0-1 범위) 및 shape 변환
3. **모델 구축**: Sequential API로 CNN 구성
4. **컴파일**: Adam 옵티마이저, Cross-Entropy 손실
5. **학습**: 3 epochs, 배치 크기 128
6. **평가**: 테스트 데이터로 정확도 측정
7. **예측**: 랜덤 샘플 5개로 예측 테스트
8. **시각화**: 결과 및 학습 곡선 저장

### PyTorch 버전
1. **모델 정의**: nn.Module 클래스로 CNN 구현
2. **데이터 로드**: DataLoader 사용
3. **학습 루프**: 수동 구현 (더 많은 제어)
4. **평가**: 별도 평가 함수
5. **모델 저장**: state_dict 형태로 저장

---

## 커스터마이징 가이드

### 더 많이 학습하고 싶다면
```python
# epochs 수 변경
epochs=3  →  epochs=10
```

### 더 빠르게 실행하고 싶다면
```python
# 학습 데이터 일부만 사용
x_train = x_train[:10000]  # 처음 10,000개만
y_train = y_train[:10000]
```

### 더 복잡한 모델을 만들고 싶다면
```python
# 합성곱 층 추가
layers.Conv2D(128, (3, 3), activation='relu'),
layers.MaxPooling2D((2, 2)),
```

### 다른 데이터셋으로 실험하고 싶다면
- **Fashion-MNIST**: 옷 이미지 분류
  ```python
  keras.datasets.fashion_mnist.load_data()
  ```
- **CIFAR-10**: 컬러 이미지 (비행기, 자동차, 새 등)
  ```python
  keras.datasets.cifar10.load_data()
  ```

---

## 문제 해결

### GPU 메모리 부족
```python
# 배치 크기 줄이기
batch_size=128  →  batch_size=64
```

### 설치 오류 (TensorFlow)
```bash
# CPU 버전만 설치
pip install tensorflow-cpu
```

### 설치 오류 (PyTorch)
```bash
# CPU 버전만 설치
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### ImportError 발생
```bash
# 패키지 재설치
pip install --upgrade tensorflow numpy matplotlib
```

---

## 학습 자료

더 자세한 이론은 다음 문서를 참고하세요:
- **CNN_core_principles.md** - CNN 핵심 원리 상세 설명

---

## 다음 단계

이 예제를 완료했다면 다음을 시도해보세요:

1. **전이 학습 (Transfer Learning)**
   - 사전 학습된 모델 사용 (ResNet, VGG 등)
   - 더 적은 데이터로 높은 성능

2. **데이터 증강 (Data Augmentation)**
   - 회전, 이동, 확대/축소
   - 과적합 방지

3. **객체 탐지 (Object Detection)**
   - YOLO, Faster R-CNN
   - 이미지 내 여러 객체 찾기

4. **세그멘테이션 (Segmentation)**
   - U-Net, DeepLab
   - 픽셀 단위 분류

---

## 참고 자료

- [TensorFlow 공식 문서](https://www.tensorflow.org/)
- [PyTorch 공식 문서](https://pytorch.org/)
- [Keras 예제 모음](https://keras.io/examples/)
- [CS231n 강의](http://cs231n.stanford.edu/)

---

**문의사항이나 버그 발견 시 Issue를 등록해주세요!**
