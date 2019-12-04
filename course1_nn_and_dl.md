# Neural Networks and Deep Learning

- Neural Networks and Deep Learning の基礎
- Improving Deep Neural Networks: Hyperparameter tuning, Regularization and Optimization
- Structuring your Machine Learning project
- Convolutional Neural Networks
- Natural Language Processing: Building sequence models

## Binary Classification

ex: cat / non cat

- 64x64 RGB (0-255)
- $ X \in R^{ m \times (64 \times 64 \times 3)} $
- $ y \in \{0, 1\}^m$
- $ X \to y $

### Logistic Regression

- Given $X$ , want $\hat{y} = P(y=1|x)$
  - $0 < \hat{y} \le 1$
- $X \in R^{nx}$
- Parameters: $w \in R^{nx}$, $ b \in R$
- Output $\hat{y} = \sigma(w^Tx + b)$
- $\sigma(z) = \frac{1}{1 + e^{-z}}$
  - If $z \to large$, $\sigma(z) \approx \frac{1}{1+0} = 1$
  - If $z \to small$, $\sigma(z) \approx \frac{1}{1+\inf} \approx 0$
- Loss(error) function $L(\hat{y},y) = -(ylog\hat{y}+(1-y)log(1-\hat{y}))$
  - If $y = 1: L(\hat{y}, y) = -log\hat{y}$ ← want $\hat{y}$ large
    - $\hat{y}$はシグモイド関数だから$0 < \hat{y} < 1$で、$-log1$の時 0 になるからね
  - If $y = 0: L(\hat{y}, y) = -log(1 - \hat{y})$ ← want $\hat{y}$ small
    - 同様に$\hat{y} \approx 0 \to -log(1-0) = 0$
- Cost function $J(w, b) = \frac{1}{m} \sum_{i=1}^m L(\hat{y}^{(i)},y^{(i)})$
- Gradient Descent $minimize ~ J(w,b)$
  - repeat
    - $w := w - \alpha\frac{\partial J(w,b)}{\partial w}$
    - $b := b - \alpha\frac{\partial J(w,b)}{\partial b}$
  - $J(w,b)$は凸関数なのでパラメータの初期値を何に指定してもグローバル局所解に至る。そのため初期値は 0 にすることが多い

```py
# pythonっぽい擬似コード
J = dw1 = dw2 = db = 0
for i in range(1, m):
  z[i] = w.T * x[i] + b
  a[i] = sigmoid(z[i])
  J += - ( y[i] * log(a[i]) + (1 - y[i] * log(1 - a[i])))
  dz[i] = a[i] - y[i]

  dw1 += x1[i] * dz[i]
  dw2 += x2[i] * dz[i]
  db += dz[i]

J /= m
dw1 /= m
dw2 /= m

w1 = w1 - alpha * dw1
w2 = w2 - alpha * dw2
b = b - alpha * db
```
