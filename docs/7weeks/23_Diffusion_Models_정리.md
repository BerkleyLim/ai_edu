# Diffusion Models and Score Matching 정리

## 개요

이 문서는 Diffusion Models와 Score Matching에 대한 포괄적인 정리입니다. 강의자료를 바탕으로 핵심 개념들을 상세히 설명합니다.

## 목차

1. [Score Matching](#score-matching)
2. [Score-based Generative Models](#score-based-generative-models)  
3. [Denoising Diffusion Probabilistic Models](#denoising-diffusion-probabilistic-models)
4. [Guided Diffusion Models](#guided-diffusion-models)

---

## Score Matching

### 비정규화 통계 모델 (Non-normalized Statistical Models)

딥러닝에서 확률 분포를 모델링할 때 다음과 같은 형태를 자주 사용합니다:

```
p(x; θ) = (1/Z(θ)) * e^(-E(x;θ))
```

여기서:
- `E(x; θ)`: 에너지 함수 또는 비정규화 확률 모델
- `Z(θ) = ∫ e^(-E(x;θ)) dx`: 정규화 상수 (normalizing constant)

**핵심 문제**: `Z(θ)`는 대부분 해석적으로 계산하기 어렵습니다.

### 학습 방법들

1. **Pseudo-likelihood estimation** (Besag, 1994)
2. **Contrastive divergence** (Hinton, 2002)  
3. **Score matching** (Hyvärinen, 2005)

### Stein Score Function

Score function은 다음과 같이 정의됩니다:

```
s(x; θ) = ∇x log p(x; θ)
```

**중요한 특성**: Score function은 정규화 상수에 의존하지 않습니다!

```
s(x; θ) = ∇x log(1/Z(θ) * e^(-E(x;θ))) 
        = -∇x log Z(θ) - ∇x E(x; θ) 
        = -∇x E(x; θ)
```

### Score Matching의 두 가지 형태

#### 1. Explicit Score Matching

실제 데이터의 score function `∇x log q(x)`를 직접 매칭:

```
arg min_θ E_q(x)[1/2 ||s(x; θ) - ∇x log q(x)||²]
```

**장점**: 정규화 상수 계산 불필요  
**단점**: 실제 데이터의 score function을 알아야 함

#### 2. Implicit Score Matching

Hyvärinen이 증명한 동등한 형태:

```
arg min_θ E_q(x)[1/2 ||s(x; θ)||² + tr(∇x s(x; θ))]
```

**장점**: 데이터만으로 학습 가능  
**단점**: 고차원에서 확장성 문제 (Jacobian 계산 필요)

### 확장성 문제

Implicit score matching은 `tr(∇x s(x; θ))` 계산 때문에 d번의 backpropagation이 필요합니다:

```
tr(∇x s(x; θ)) = Σ_i ∂s_θ,i(x)/∂x_i
```

고차원 데이터에서는 계산 비용이 매우 높아집니다.

### Denoising Score Matching

**해결책**: 노이즈가 추가된 분포에서 score matching 수행

```
q_σ(x̃) = ∫ q(x)q_σ(x̃|x)dx
```

Vincent가 증명한 동등 관계:

```
1/2 E_q_σ(x̃)[||s_θ(x̃) - ∇x̃ log q_σ(x̃)||²] = 
1/2 E_x~q(x),z~N(0,I)[||s_θ(x + σz) + z/σ||²] + const
```

**장점**: 훨씬 더 확장 가능  
**단점**: 정확한 데이터 분포가 아닌 노이즈 분포를 근사

---

## Score-based Generative Models

### 기본 아이디어

1. **학습**: Score function `s_θ(x) ≈ ∇x log p(x)` 학습
2. **생성**: Langevin dynamics로 샘플 생성

### Langevin Dynamics

확률 분포에서 샘플링하는 MCMC 방법:

```
x_{t+1} = x_t + (ε/2) * ∇x log q(x_t) + √ε * z_t
```

여기서:
- `ε`: step size
- `z_t ~ N(0, I)`: 가우시안 노이즈
- `ε → 0, T → ∞`이면 `x_T`의 분포가 `q(x)`에 수렴

### 학습된 모델로 샘플링

```
x_{t+1} = x_t + (ε/2) * s_θ(x_t) + √ε * z_t
```

### 문제점: 저밀도 영역에서의 부정확성

Score matching은 데이터가 많은 고밀도 영역에서만 정확합니다. 저밀도 영역에서는 부정확한 score function을 학습하게 됩니다.

### 해결책: Multi-Scale Noise Perturbations

여러 노이즈 레벨을 사용한 학습:

```
σ_1 > σ_2 > ... > σ_L
```

목적함수:
```
Σ_{i=1}^L λ_i E_q_σi(x)[||s_θ(x, σ_i) - ∇x log q_σi(x)||²]
```

보통 `λ_i = σ_i²`로 설정합니다.

### Noise Conditional Score Networks (NCSN)

노이즈 레벨에 따라 조건부 score function을 학습:

```
s_θ(x, σ)
```

Denoising score matching 목적함수:
```
1/(2L) Σ_{i=1}^L λ_i E_x∈q(x),x̃~N(x,σ_i²I)[||s_θ(x̃, σ_i) + (x̃-x)/σ_i²||²]
```

### Annealed Langevin Dynamics

각 노이즈 레벨에서 순차적으로 샘플링:

```
x_{t+1}^i = x_t^i + (σ_i²/2σ_L²) * s_θ(x_t^i, σ_i) + √(σ_i²/σ_L²) * z_t^i
```

여기서 `x_0^{i+1} = x_T^i`로 이어집니다.

---

## Stochastic Differential Equations (SDE) 접근법

### 연속 시간으로의 확장

유한한 노이즈 레벨에서 무한한 노이즈 레벨로 확장하면 확률 과정(stochastic process)이 됩니다.

### Forward SDE

```
dx(t) = f(x(t), t) dt + g(t) dw(t)
```

여기서:
- `f(·, t)`: drift coefficient
- `g(·)`: diffusion coefficient  
- `w(t)`: Wiener process

### Reverse SDE

Anderson의 정리에 의해 역과정도 SDE로 표현 가능:

```
dx(t) = [f(x(t), t) - g(t)² ∇x log p_t(x)] dt + g(t) dw̄(t)
```

### 학습과 생성

1. **학습**: 시간 의존적 score model `s_θ(x, t) ≈ ∇x log p_t(x)` 학습
2. **생성**: Reverse SDE로 샘플 생성

**훈련 목적함수**:
```
arg min_θ E_t~Unif[0,T] {λ(t) E_x(0)~p_0(x) E_x(t)~p_{0t}(x(t)|x(0)) [||s_θ(x(t), t) - ∇_{x(t)} log p_{0t}(x(t)|x(0))||²]}
```

**샘플링**: Euler-Maruyama 방법
```
x ← x - σ²(t)s_θ(x, t)Δt + σ(t)z
```

---

## Denoising Diffusion Probabilistic Models (DDPM)

### 기본 구조

DDPM은 두 가지 프로세스로 구성됩니다:

#### 1. Forward Diffusion Process (고정)
점진적으로 노이즈 추가:

```
q(x_t|x_{t-1}) = N(x_t|√(1-β_t)x_{t-1}, β_t I)
```

#### 2. Reverse Denoising Process (학습)  
노이즈를 제거하여 데이터 생성:

```
p_θ(x_{t-1}|x_t) ≈ q(x_{t-1}|x_t, x_0)
```

### 핵심 통찰

`x_0`가 주어졌을 때 `q(x_{t-1}|x_t, x_0)`는 계산 가능한 가우시안 분포:

```
q(x_{t-1}|x_t, x_0) = N(x_{t-1}|μ̃_t(x_t, x_0), β̃_t I)
```

여기서:
```
μ̃_t(x_t, x_0) = (1/√α_t)(x_t - (1-α_t)/√(1-ᾱ_t) * ε_t)
β̃_t = (1-ᾱ_{t-1})/(1-ᾱ_t) * β_t
```

그리고:
- `α_t = 1 - β_t`
- `ᾱ_t = ∏_{i=1}^t α_i`

### Reparameterization Trick

가우시안 노이즈 ε_t를 직접 예측하도록 재매개화:

```
μ_θ(x_t, t) = (1/√α_t)(x_t - (1-α_t)/√(1-ᾱ_t) * ε_θ(x_t, t))
```

### 간소화된 목적함수

```
E_t~[T],x_0,ε_t [||ε_t - ε_θ(√ᾱ_t x_0 + √(1-ᾱ_t) ε_t, t)||²]
```

### 알고리즘

**훈련**:
1. `x_0 ~ q(x_0)` 샘플링
2. `t ~ Uniform({1, ..., T})` 샘플링  
3. `ε ~ N(0, I)` 샘플링
4. `||ε - ε_θ(√ᾱ_t x_0 + √(1-ᾱ_t) ε, t)||²`에 대해 gradient descent

**샘플링**:
1. `x_T ~ N(0, I)`에서 시작
2. `t = T, ..., 1`에 대해:
   ```
   x_{t-1} = (1/√α_t)(x_t - (1-α_t)/√(1-ᾱ_t) * ε_θ(x_t, t)) + σ_t z
   ```

---

## Guided Diffusion Models

### 문제 상황

기본 diffusion model은 무작위 샘플만 생성합니다. 특정 조건(예: "테이블 위의 사과")에 따른 생성이 필요합니다.

### Guidance의 기본 아이디어

가우시안 분포의 평균을 특정 방향으로 이동:

```
μ_θ(x_t) → μ_θ(x_t|y) + "something"
```

### 1. Classifier-based Guidance

**과정**:
1. 노이즈 샘플에서 작동하는 분류기 `p_φ(y|x)` 학습
2. 역확산 과정에서 분류기 gradient 사용:

```
x_{t-1} ~ N(μ_θ(x_t) + s * Σ_θ(x_t) * ∇_{x_t} log p_φ(y|x_t), Σ_θ(x_t))
```

여기서 `s`는 guidance scale입니다.

**장점**: 이산 레이블에 효과적  
**단점**: 별도의 분류기 학습 필요

### 2. CLIP-based Guidance

텍스트 조건을 위해 CLIP 모델 활용:

**CLIP 모델**:
- 이미지 인코더와 텍스트 인코더를 공동 학습
- 대조 학습으로 multi-modal embedding space 학습
- `N×N` 가능한 (image, text) 쌍 중 실제 쌍 예측

**Guidance**:
```
x_{t-1} ~ N(μ_θ(x_t) + s * Σ_θ(x_t) * ∇_{x_t}⟨f(x_t), g(y)⟩, Σ_θ(x_t))
```

여기서 `f`, `g`는 각각 이미지, 텍스트 인코더입니다.

### 3. Classifier-free Guidance

**핵심 아이디어**: 별도의 guidance 모델 없이 조건부/무조건부 모델을 함께 학습

**학습 과정**:
1. 조건부 diffusion model `p_θ(x|y)` 학습
2. 훈련 중 조건 정보 `y`를 랜덤하게 제거 → 무조건부 생성도 학습

**추론 과정**:
```
ε̃_θ(x_t, t, y) = ε_θ(x_t, t, ∅) + s * (ε_θ(x_t, t, y) - ε_θ(x_t, t, ∅))
```

**해석**: 무조건부 예측에서 조건부 예측 방향으로 이동

### Guidance Scale의 효과

- **낮은 s**: 다양하지만 품질이 낮은 이미지
- **높은 s**: 고품질이지만 다양성이 낮은 이미지

품질과 다양성 사이의 트레이드오프가 존재합니다.

---

## 실제 응용 사례

### GLIDE
- OpenAI에서 개발한 텍스트 조건부 이미지 생성 모델
- CLIP guidance와 classifier-free guidance 비교 연구
- 고해상도 이미지 생성 및 편집 기능

### DALL-E 2  
- CLIP latent space에서의 계층적 텍스트 조건부 이미지 생성
- 더 높은 품질과 텍스트 일치도 달성

---

## 주요 개념 정리

### Score Function
- 확률 분포의 로그 gradient: `∇x log p(x)`
- 정규화 상수에 무관한 중요한 특성
- 확률 분포의 국소적 기울기 정보 제공

### Denoising의 중요성
- 저밀도 영역에서의 score 추정 문제 해결
- 다양한 노이즈 레벨에서의 robust한 학습
- 실제 구현에서 핵심적인 기술

### SDE vs DDPM
- **SDE**: 연속 시간, 수학적으로 우아함, 이론적 완성도
- **DDPM**: 이산 시간, 구현 단순함, 실용적 효과

### Guidance의 핵심
- 조건부 생성을 위한 필수 기술
- 분포의 평균을 원하는 방향으로 조정
- 품질-다양성 트레이드오프 제어

---

## 결론

Diffusion model은 현재 가장 강력한 생성 모델 중 하나로, score matching의 이론적 기반 위에 세워진 실용적인 프레임워크입니다. 핵심은 다음과 같습니다:

1. **Score matching**: 정규화 상수 없이 확률 분포 학습
2. **Denoising**: 다양한 노이즈 레벨에서의 robust한 학습
3. **Diffusion process**: 점진적 노이즈 추가/제거 과정
4. **Guidance**: 조건부 생성을 위한 분포 조정

이러한 요소들이 결합되어 고품질의 이미지, 텍스트, 오디오 등 다양한 데이터 유형에서 뛰어난 생성 성능을 보여주고 있습니다.