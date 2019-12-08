# Improving Deep NN

## Train/dev/test sets

- 100 - 10000
  - 60/20/20 [%]
- 1,000,000
  - 98/1/1 [%]

## Bias and Variance

High Variance

- Train error 1% : Dev error 11%
- overfit

High Bias

- Train error 15% : Dev error 16%
- underfit

High Variance and Bias

- Train error 15% : Dev error 30%
- overfit and underfit

## Basic recipe for ML

1: Solve High Bias

- more bigger Network
- other NN architecture

2: Solve High Variance

- more data set
- Regularization
- other NN architecture

## Regularization

L2 Regularization of Logistic Regression

$minJ(w,b)$
$J(w,b) = \frac{1}{m}\sum_{i=1}^{m}L(\hat{y}^{[i]}, y^{[i]}) + \frac{\lambda}{2m} ||w||_2^2$
$||w||_2^2 = \sum_{j=1}^{nx}w_j^2 = w^Tw$

L1 は w を^2 しない。あまり使われない。

L2 Regularization of NN

$minJ(w^{[1]},b^{[1]},...w^{[L]}, b^{[L]})$
$J(w^{[1]},b^{[1]},...w^{[L]}, b^{[L]}) = \frac{1}{m}\sum_{i=1}^{m}L(\hat{y}^{[i]}, y^{[i]}) + \frac{\lambda}{2m} \sum_{l=1}^L||w^{[l]}||^2$
$||w^{[l]}||_F^2 = \sum_{i=1}^{n^{[l-1]}}\sum_{j=1}^{n^{[l]}}(w_{ij}^{[l]})^2$
$\uparrow Frobenius norm$

Dropout Regularization

- ノードをランダムにある程度の割合で消す
- ノードがランダムに消えるため、コスト J が予測しづらくなる。
  - J と#iter のグラフを見ても順調にコストが減少しているかどうかが判別しづらくなる

## Data augmentation

- データを加工してデータセットを増やす
  - 画像なら反転させたり拡大縮小歪みを加えたり

## Early stopping

Gradient Descent でコスト J を最小化する際、train set Error だけでなく、dev set error も計測し、dev set error が増加しはじめた(overfit による)ら学習を打ち切る手法

正則化の代わりに overfit を避けるために用いられるが、本質的には overfit とコスト J を最小化することは異なる処理なので？L2 正則化を使ったほうが良い。が、L2 正則化は λ を最適化するのに何度もネットワークで学習させないといけないのでとても時間がかかる。

## Feature Normalization

$X := (X -\mu) / \sigma^2$

**train set の $\mu \text{と} \sigma$ を test set でも利用すること**

## Vanishing/Exploding Gradients

ネットワークが深いと、重み w の掛け算のようになり、小数 \* 小数が続いて指数関数的に 0 に近づいたり、値 \* 値がどんどん大きくなったりする。これは学習を非常に遅くしたり失敗させたりする。

重み w を正規化することで、上記を避ける可能性が上がる。

$w^{[l]} = np.random.randn(shape) * np.sqrt(\frac{2}{n^{[l-1]}})$

※tanh の場合は$np.sqrt(\frac{1}{n^{[l-1]}})$
※tanh の場合は$np.sqrt(\frac{1}{n^{[l-1]}})$

## Gradient Checking

微分項の確認の為に、微分前の項に $\epsilon$ を足した場合と引いた場合で傾きを得る。Backward propagation で計算した Gradient と比較して解が合っているかを確認する

## Optimize Train speed

### Mini-batch gradient descent

train set を M 分割して cost と grad をそれぞれ計算する

### Stochastic gradient descent

Mini-batch で M 分割したあと、そのうちの一つだけを計算する

- 早いが cost が毎回必ずしも減少するとは限らない
- 絶対に収束はしない（打ち切りが必要）

### バッチサイズの決め方

- 全体の train set が m < 2000 程度なら Full-Batch
- m > 2000 なら 64, 128, 256, 512 くらいのミニバッチ
  - ハイパーパラメータなので試すしかない
- X, Y のサンプル数は CPU/GPU メモリに依存する

## Moving Average

### EMA

$V_t = \beta V_{t-1} + (1-\beta)\theta_t$
$\quad \beta = 0.9 \to \approx10 day's ~ average $
$\quad \beta = 0.98 \to \approx50 day's ~ average $
$\quad \approx \frac{1}{1-\beta}day's \quad \text{数学的な法則ではないがReasonable}$

$0.9^{10} \approx 0.35 \approx \frac{1}{e}$
$(1-\epsilon)^{1/\epsilon} \approx \frac{1}{e} \quad | \epsilon = 0.1$
→ β に 0.9 を採用すると、期間が 10(日)で重みが 1/3 に減少する
→ β に 0.98 を採用すると、期間が 50(日)で重みが 1/3 に減少する

SMA と比較すると、EMA は最新の値だけ覚えておけば良いため、メモリ効率と計算効率が良い

## Momentum

$V_{dw} = \beta V_{dw} + (1-\beta)dw$
$V_{db} = \beta V_{db} + (1-\beta)db$
$w := w - \alpha V_{dw}, b := b - \alpha V_{db}$

EMA で重み付けして Gradient Descent を行う
$(1-\beta)dw$は加速度みたいなもん

## RMSprop

$S_{dw} = \beta S_{dw} + (1-\beta)dw^2$
$S_{db} = \beta S_{db} + (1-\beta)db^2$
$w := w - \alpha \frac{dw}{\sqrt{S_{dw}}}, w := b - \alpha \frac{db}{\sqrt{S_{db}}}$

$w$は水平方向、$b$は垂直方向を示し、$w$よりも$b$の方が値が大きい。その性質を利用して二乗して割ることで垂直方向の影響を抑え、水平方向の影響を強くしている。

## Adam

Momentum + RMSprop な最適化手法
$V_{dw} = \beta_1V_{dw} + (1-\beta_1)dw$
$V_{db} = \beta_1V_{db} + (1-\beta_1)db$
$S_{dw} = \beta_2S_{dw} + (1-\beta_2)dw^2$
$S_{db} = \beta_2S_{db} + (1-\beta_2)db^2$

$V_{dw}^{corrected} = \frac{V_{dw}}{(1-\beta_1^t)}$
$V_{db}^{corrected} = \frac{V_{db}}{(1-\beta_1^t)}$
$S_{dw}^{corrected} = \frac{S_{dw}}{(1-\beta_2^t)}$
$S_{db}^{corrected} = \frac{S_{db}}{(1-\beta_2^t)}$
$w := w - \alpha \frac{V_{dw}^{corrected}}{\sqrt{S_{dw}^{corrected}}  + \epsilon}$
$b := b - \alpha \frac{V_{db}^{corrected}}{\sqrt{S_{db}^{corrected}}  + \epsilon}$

Default Params: (基本的に α のみ調整が必要)
$\alpha:$ needs to be tune
$\beta_1: 0.9 \quad (dw)$
$\beta_2: 0.999 \quad (dw^2)$
$\epsilon: 10^{-8}$
