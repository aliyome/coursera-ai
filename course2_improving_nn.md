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
