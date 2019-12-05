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

#### Vectorize(1)

```py
J = db = 0
dw = np.zeros((n,1))

for i in range(1, m):
  z[i] = w[i] @ x[i] + b  # w.T * x + b
  a[i] = sigmoid(z[i])
  J += - ( y[i] * log(a[i]) + (1 - y[i] * log(1 - a[i])))
  dz[i] = a[i] - y[i]
  dw += x[i] * dz[i]
  db += dz[i]

J /= m
dw /= m
```

#### Vectorize(2)

```py
J = db = 0
dw = np.zeros((n, 1)) # (n, 1)

z = w.T @ x + b  # (1, n)@(n, m) = (1, m)
a = sigmoid(z)  # (1, m)
dz = a - y  # (1, m)

dw = (X @ dz.T) / m  # (n, m)@(m, 1) = (n, 1)
db = dz.sum() / m  # (1, m) -> (1)

w -= alpha * dw
b -= alpha * db

j += - ( y * log(a) + (1 - y * log(1 - a)))  # (1, m)
J = j.sum() / m  # (1, m) -> (1)
```

### memo

ndarray.reshape()は定数時間で処理するのでどんどん使っていい

### Logistic Regression Cost function

If $y = 1: p(y|x) = \hat{y}$
If $y = 0: p(y|x) = 1 - \hat{y}$

$p(y|x) = \hat{y}^y(1-\hat{y})^{(1-y)}$
$\log p(y|x) = \log \hat{y}^y(1-\hat{y})^{(1-y)}$
$ \qquad = y\log\hat{y} + (1-y)\log(1-\hat{y})$

$p(\text{labels in train set}) = \prod_{i=1}^m p(y^{(i)}|x^{(i)})$
$\log p(\text{labels in train set}) = \log \prod_{i=1}^m p(y^{(i)}|x^{(i)})$
$\qquad = \sum_{i=1}^m \log p(y^{(i)}|x^{(i)})$
$\qquad = -\sum_{i=1}^m L(\hat{y}^{(i)}, y^{(i)})$
$Cost: J(w, b) = 1/m \sum_{i=1}^m L(\hat{y}^{(i)}, y^{(i)}) $

んー、ちょっとよくわからないですね
最尤なんとかが関係してるらしい

### NN

$x \in R^{3 \times m} \to a^{[1]} \in R^{4 \times m}$
$W^{[1]} \in R^{4 \times 3}, b^{[1]} \in R^{4 \times 1}$

$z^{[1]} = W^{[1]}x + b^{[1]}$
$a^{[1]} = \sigma(z^{[1]})$

$z^{[2]} = W^{[2]}a^{[1]} + b^{[2]}$
$a^{[2]} = \sigma(z^{[2]})$

### Gradient descent for NN

Parameters:
$ \quad w^{[1]} \in R^{n1,n0},b^{[1]} \in R^{n1,1},w^{[2]} \in R^{n2,n1},b^{[2]} \in R^{n2,1}$

Cost function:
$\quad J(w^{[1]}, b^{[1]}, w^{[2]},b^{[2]} = \frac{1}{m} \sum_{i=1}^m L(\hat{y}, y)$

Gradient descent:
$\quad Repeat \{$
$\qquad dw^{[1]} = \frac{\partial J}{\partial dw^{[1]}}$
$\qquad db^{[1]} = \frac{\partial J}{\partial db^{[1]}}$
...
$\qquad w^{[1]} := w^{[1]} - \alpha  dw^{[1]}$
$\qquad b^{[1]} := b^{[1]} - \alpha  db^{[1]}$
...
$\quad\}$

### Forward and Backward prop

Forward
$z^{[1]} = w^{[1]}x + b^{[1]}$
$a^{[1]} = g^{[1]}(z^{[1]})$
$z^{[2]} = w^{[2]}a^{[1]} + b^{[2]}$
$a^{[2]} = g^{[2]}(z^{[2]})$

---

Backward

$dz^{[2]} = a^{[2]} - y$
$dw^{[2]} = \frac{1}{m}dz^{[2]}a^{[1]T}$
$db^{[2]} = \frac{1}{m} \text{np.sum}(dz^{[2]}, axis=1, keepdims=True)$ # (n, 1) by keepdims
$dz^{[1]} = w^{[2]T}dz^{[2]} * g^{[1]'(z^{[1]})}$
$dw^{[1]} = \frac{1}{m}dz^{[1]}x^T$
$db^{[1]} = \frac{1}{m}\text{np.sum}(dz^{[1]}, axis=1, keepdims=True)$
