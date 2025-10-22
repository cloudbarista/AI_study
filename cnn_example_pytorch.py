"""
CNN 간단 예제 - PyTorch 버전
MNIST 손글씨 숫자 인식 (0-9)

실행 방법:
pip install torch torchvision numpy matplotlib
python cnn_example_pytorch.py
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

print("="*50)
print("CNN 간단 예제 - MNIST 손글씨 숫자 인식 (PyTorch)")
print("="*50)

# GPU 사용 가능 여부 확인
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\n사용 디바이스: {device}")

# 1. CNN 모델 정의
print("\n[1단계] CNN 모델 정의...")
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()

        # 첫 번째 합성곱 블록: 1채널 -> 32채널
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)  # 28x28 -> 14x14

        # 두 번째 합성곱 블록: 32채널 -> 64채널
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2)  # 14x14 -> 7x7

        # 세 번째 합성곱 블록: 64채널 -> 64채널
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU()

        # 완전 연결 층
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 7 * 7, 64)
        self.relu4 = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        # 합성곱 블록들
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.relu3(self.conv3(x))

        # 완전 연결 층
        x = self.flatten(x)
        x = self.relu4(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

model = SimpleCNN().to(device)
print(model)

# 파라미터 수 계산
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\n총 파라미터 수: {total_params:,}")
print(f"학습 가능 파라미터 수: {trainable_params:,}")

# 2. 데이터 로드
print("\n[2단계] 데이터 로딩...")
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

print(f"학습 데이터: {len(train_dataset)}개")
print(f"테스트 데이터: {len(test_dataset)}개")

# 3. 손실 함수와 옵티마이저
print("\n[3단계] 손실 함수와 옵티마이저 설정...")
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 4. 학습 함수
def train_epoch(model, device, train_loader, optimizer, criterion):
    model.train()
    train_loss = 0
    correct = 0
    total = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()

    return train_loss / len(train_loader), 100. * correct / total

# 5. 평가 함수
def evaluate(model, device, test_loader, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()

            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

    return test_loss / len(test_loader), 100. * correct / total

# 6. 모델 학습
print("\n[4단계] 모델 학습 시작...")
print("(3 epochs만 학습 - 빠른 데모용)\n")

num_epochs = 3
history = {'train_loss': [], 'train_acc': [], 'test_loss': [], 'test_acc': []}

for epoch in range(num_epochs):
    train_loss, train_acc = train_epoch(model, device, train_loader, optimizer, criterion)
    test_loss, test_acc = evaluate(model, device, test_loader, criterion)

    history['train_loss'].append(train_loss)
    history['train_acc'].append(train_acc)
    history['test_loss'].append(test_loss)
    history['test_acc'].append(test_acc)

    print(f'Epoch {epoch+1}/{num_epochs}:')
    print(f'  학습 - Loss: {train_loss:.4f}, Accuracy: {train_acc:.2f}%')
    print(f'  테스트 - Loss: {test_loss:.4f}, Accuracy: {test_acc:.2f}%')

# 7. 최종 평가
print(f"\n[5단계] 최종 테스트 정확도: {test_acc:.2f}%")

# 8. 예측 예시
print("\n[6단계] 실제 예측 테스트...")
model.eval()

# 테스트 데이터에서 샘플 추출
examples = []
with torch.no_grad():
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        pred = output.argmax(dim=1)

        # 랜덤으로 5개 선택
        indices = np.random.choice(len(data), 5, replace=False)
        for idx in indices:
            img = data[idx].cpu().numpy().squeeze()
            true_label = target[idx].item()
            pred_label = pred[idx].item()
            confidence = torch.softmax(output[idx], dim=0).max().item() * 100
            examples.append((img, true_label, pred_label, confidence))
        break

# 시각화
plt.figure(figsize=(15, 3))
for i, (img, true_label, pred_label, confidence) in enumerate(examples):
    plt.subplot(1, 5, i+1)
    plt.imshow(img, cmap='gray')
    plt.title(f'예측: {pred_label}\n정답: {true_label}\n확률: {confidence:.1f}%')
    plt.axis('off')

    print(f"샘플 {i+1}: 정답={true_label}, 예측={pred_label}, "
          f"확률={confidence:.1f}% {'✓' if pred_label==true_label else '✗'}")

plt.tight_layout()
plt.savefig('cnn_predictions_pytorch.png', dpi=100, bbox_inches='tight')
print("\n결과 이미지 저장: cnn_predictions_pytorch.png")

# 9. 학습 곡선 시각화
print("\n[7단계] 학습 곡선 저장...")
plt.figure(figsize=(12, 4))

# 정확도
plt.subplot(1, 2, 1)
plt.plot(history['train_acc'], label='학습 정확도', marker='o')
plt.plot(history['test_acc'], label='테스트 정확도', marker='s')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.title('모델 정확도')
plt.grid(True)

# 손실
plt.subplot(1, 2, 2)
plt.plot(history['train_loss'], label='학습 손실', marker='o')
plt.plot(history['test_loss'], label='테스트 손실', marker='s')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('모델 손실')
plt.grid(True)

plt.tight_layout()
plt.savefig('cnn_training_history_pytorch.png', dpi=100, bbox_inches='tight')
print("학습 곡선 저장: cnn_training_history_pytorch.png")

# 10. 모델 저장
print("\n[8단계] 모델 저장...")
torch.save(model.state_dict(), 'cnn_mnist_model.pth')
print("모델 저장: cnn_mnist_model.pth")

print("\n" + "="*50)
print("실행 완료!")
print("="*50)
print("\n생성된 파일:")
print("- cnn_predictions_pytorch.png : 예측 결과")
print("- cnn_training_history_pytorch.png : 학습 곡선")
print("- cnn_mnist_model.pth : 학습된 모델")
