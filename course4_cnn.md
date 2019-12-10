# Convolutional Neural Networks

画像は 64x64x3 の小さいサイズでも 12288 空間の特徴量になってしまう。大きい画像なんか無理。

## 畳み込み演算(filter または kernel)

```py
# 縦線検出フィルタ,  横線検出フィルタ
1, 0,-1              1, 1, 1
1, 0,-1              0, 0, 0
1, 0,-1             -1,-1,-1
```

$f^{[l]} = $ filter size
$p^{[l]} = $ padding
$s^{[l]} = $ stride
$n_c^{[l]} = $ number of filters

Each filter is: $f^{[l]} \times f^{[l]} \times n_c^{[l-1]}$
Activations: $a^{[l]} \to n_h^{[l]} \times n_w^{[l]} \times n_c^{[l]}$
Weights: $f^{[l]} \times f^{[l]} \times n_c^{[l-1]} \times n_c^{[l]}$
bias: $n_c^{[l]}$

$Input: n_h^{[l-1]} \times n_w^{[l-1]} \times n_c^{[l-1]} $
$Output: n_h^{[l]} \times n_w^{[l]} \times n_c^{[l]} $

$n_{h,w}^{[l]} = \frac{n^{[l-1]} + 2p^{[l]} - f^{[l]}}{s^{[l]}} + 1$

## 典型的な CNN の構造

![1](./typical_cnn1.png)
![2](./typical_cnn2.png)
