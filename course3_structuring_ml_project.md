# Structuring Machine Learnig Project

## Single Number Evaluation

Positive sample が極端に少ない場合には$F_1$スコアを使って性能を測るのが優秀

## Statisificing and Optimizing Metrics

場合によっては、学習モデルを使った「実行速度」なども重要になる
例えば、精度はなるだけ高いほうが良いが「100ms 以内に」実行できること、など
その場合はコスト関数に実行速度を加えることもある

## Train/Dev/Test set

Test set と同じ比率の Dev set を使わなければ本番で良い精度で出なくてゴミになる

Dev set と Test set の特徴はなるだけ同じにしないと行けない。例えば、ユーザが投稿する写真(Test set)はボケてる写真が多いのに、Dev set でくっきり高解像度の写真を使うのはダメ

## Human level performance

- 最高の精度は 100%ではなく、ベイズエラーが現れる
- 人の精度を超えたあたりで実行時間に対して精度が上がらなくなってくる

### Avoidable bias

| Avoidable Bias  | 7%            | 0.5%              |
| --------------- | ------------- | ----------------- |
| Humans Error    | 1%            | 7.5%              |
| Traininng Error | 8%            | 8%                |
| Dev Error       | 10%           | 10%               |
| You should      | focus on bias | focus on variance |

### Improving model guiline

Human-level
↕ (Avoidable bias)
Training error
↕ (Variance)
Dev error

#### Avoidable Bias

- train longer / better optimization algorithms
  - momentum, rmsprop, adam, ...
- NN architecture / hyperparam search
  - RNN, CNN, ...

#### Variance

- More data
- Regularization
  - L2, dropout, data augumentation
- NN architecture / hyperparam search
