# 🧠 DeepLearning Study

딥러닝 논문을 직접 읽고, 대표적인 CNN 아키텍처들을 PyTorch로 **스크래치부터 직접 구현**한 저장소입니다.  
학습용 코드로, 각 모델은 논문 기반으로 작성되었고 주요 구성 요소들을 클래스 단위로 분리하였습니다.

---

## 📅 구현 목록 및 논문 링크

| 모델명 | 구현 날짜 | 논문 링크 |
|--------|-----------|-----------|
| ✅ VGG | 2025.07.31 | [Very Deep Convolutional Networks for Large-Scale Image Recognition (2014)](https://arxiv.org/abs/1409.1556) |
| ✅ ResNet (BasicBlock) | 2025.08.04 | [Deep Residual Learning for Image Recognition (2015)](https://arxiv.org/abs/1512.03385) |
| ✅ DenseNet (with transition layer) | 2025.08.06 | [Densely Connected Convolutional Networks (2016)](https://arxiv.org/abs/1608.06993) |
| ✅ Xception (with separable conv) | 2025.08.07 | [Xception: Deep Learning with Depthwise Separable Convolutions (2017)](https://arxiv.org/abs/1610.02357) |
| ✅ MobileNet (with depthwise separable conv) | 2025.08.07 | [MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications (2017)](https://arxiv.org/abs/1704.04861) |
| ✅ ShuffleNet (with channel shuffling) | 2025.08.08 | [ShuffleNet: An Extremely Efficient Convolutional Neural Network for Mobile Devices (2017)](https://arxiv.org/abs/1707.01083) |
| ✅ ResNeXt (with cardinality) | 2025.08.11 | [Aggregated Residual Transformations for Deep Neural Networks (2017)](https://arxiv.org/abs/1611.05431) |
| ✅ Squeeze-and-Excitation (with SE block) | 2025.08.11 | [Squeeze-and-Excitation Networks (2018)](https://arxiv.org/abs/1709.01507) |
| ✅ MobileNetV2 (with invert residual) | 2025.08.12 | [MobileNetV2: Inverted Residuals and Linear Bottlenecks (2018)](https://arxiv.org/abs/1801.04381) |
| ✅ DarkNet53 | 2025.08.12 | [YOLOv3: An Incremental Improvement (2018)](https://arxiv.org/abs/1804.02767) |
| ✅ GhostNet | 2025.08.13 | [GhostNet: More Features from Cheap Operations (2019)](https://arxiv.org/abs/1911.11907) |
| ✅ CSPNet (with partial dense block)| 2025.08.14 | [CSPNet: A New Backbone that can Enhance Learning Capability of CNN (2019)](https://arxiv.org/abs/1911.11929) |
| ✅ EfficientPNet (wit MBconv) | 2025.08.18 | [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks (2019)](https://arxiv.org/abs/1905.11946) |
| ✅ MobileNetV3 | 2025.08.18 | [Searching for MobileNetV3 (2019)](https://arxiv.org/abs/1905.02244) |
| ✅ NFNet (with scaled weight standardization conv)| 2025.08.21 | [High-Performance Large-Scale Image Recognition Without Normalization (2021)](https://arxiv.org/abs/2102.06171) |
---

## 🛠️ 구현 시 특징

- 모든 모델은 PyTorch의 기본 모듈만 사용
- 공통 구조를 `Block`, `Model` 단위로 모듈화
- CIFAR-10 기반으로 훈련 가능하도록 설계
- 학습 및 테스트는 `train.py`로 분리

---


## 📌 참고

- 프레임워크: PyTorch 2.x
- 데이터셋: CIFAR-10
- 실행 환경: Python 3.10, CUDA 11 이상

---

## 🙋‍♀️ Author

- 이름: 이희재  
- GitHub: [@Heejae-L](https://github.com/Heejae-L)