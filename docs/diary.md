## Diary
### 2024/05/27
#### exp

- 特徴量を追加して、CVが上がるか調べる
  - LB: 0.656のnotebookの設定
    - num_expr(=P, A): max, last, mean
    - date(=D): max, last, mean
    - str(=M): max, last
    - other(=T, L): max, last
    - count(=num_group): max, last
  - 自分のnotebookの特徴量はmax, mean, varのみ
  - lastを追加 -> 向上

    ```
    CV AUC scores for CatBoost:  [0.8079759451079872, 0.7902685957337676, 0.8074474771654386, 0.8112194702442381, 0.841776222704894]
    Maximum CV AUC score for CatBoost:  0.841776222704894
    CV AUC scores for LGBM:  [0.8167326539360416, 0.812457651692548, 0.8282550593287852, 0.8177598899208806, 0.849682173011073]
    Maximum CV AUC score for LGBM:  0.849682173011073
    ```

  - last, min, modeを追加 -> 悪化

    ```
    CV AUC scores for CatBoost:  [0.8393347546489643, 0.8002124048129938, 0.8292853934716131, 0.7924030274132088, 0.8129056509746336]
    Maximum CV AUC score for CatBoost:  0.8393347546489643
    CV AUC scores for LGBM:  [0.8465718998563141, 0.8198812006932772, 0.8395678771699157, 0.80967979366234, 0.8179445007173649]
    Maximum CV AUC score for LGBM:  0.8465718998563141
    ```

  - last, modeを追加 -> 悪化

    ```
    CV AUC scores for CatBoost:  [0.8279435831400331, 0.8188083152668536, 0.8217109521376255, 0.797130943409019, 0.8059460322746936]
    Maximum CV AUC score for CatBoost:  0.8279435831400331
    CV AUC scores for LGBM:  [0.8357554931603157, 0.8291786596028581, 0.8337319025719638, 0.8153417077171415, 0.8179539847191665]
    Maximum CV AUC score for LGBM:  0.8357554931603157
    ```

  - last, sumを追加 -> 悪化

    ```
    CV AUC scores for CatBoost:  [0.8267580367232591, 0.8173706401178576, 0.7828478459172222, 0.7944963182541015, 0.8316122158467123]
    Maximum CV AUC score for CatBoost:  0.8316122158467123
    CV AUC scores for LGBM:  [0.8414003859694055, 0.8160724133808466, 0.7999558255339615, 0.8162821140428996, 0.8391908754762207]
    Maximum CV AUC score for LGBM:  0.8414003859694055
    ```

  - last, medianを追加 -> 悪化

    ```
    CV AUC scores for CatBoost:  [0.8251788274590744, 0.8233242195654042, 0.7929653773660927, 0.8002608082795322, 0.792774158007378]
    Maximum CV AUC score for CatBoost:  0.8251788274590744
    CV AUC scores for LGBM:  [0.8262651845782981, 0.832422570491857, 0.8097452271738611, 0.8232752544953623, 0.8090323244910028]
    Maximum CV AUC score for LGBM:  0.832422570491857
    ```

  - varをstdへ -> 悪化

    ```
    CV AUC scores for CatBoost:  [0.8146160330194163, 0.7906193299116822, 0.81819608102194, 0.8130323551258799, 0.8167682781575546]
    Maximum CV AUC score for CatBoost:  0.81819608102194
    CV AUC scores for LGBM:  [0.8374520404604116, 0.8072776047613028, 0.8311722110588677, 0.8325152801710323, 0.8197270864855896]
    Maximum CV AUC score for LGBM:  0.8374520404604116
    ```

- モデル追加
  - 他のGBDT系モデル
    - https://www.kaggle.com/code/komekami/linking-writing-processes-to-writing-quality
    - xgb

    ```
    ```

  - NN系モデル
    - AutoML NN
    - https://www.kaggle.com/code/alexryzhkov/lightautoml-nn-test
- MetricHackでWEEK_NUMを用いてみる
- HPO with Optuna

#### Survey

- VotingClassifier Home Credit
  - https://www.kaggle.com/code/jonathanchan/votingclassifier-home-credit?scriptVersionId=179137452
  - 公開notebookでは、最高スコアのLB: 0.656
  - VotingClassifier > VotingRegressor
- Shakeup is all you need!!
  - https://www.kaggle.com/competitions/home-credit-credit-risk-model-stability/discussion/507556
  - 60~91sまでのWEEK_NUMをtrainから取り出し、testとすれば、Adversal Validationとできる
  - 単純な確率shiftはPublicLBでは有効だが、それ移行の期間では効かないよう
  - Adversal Validation: trainとtestの分布が異なる際に、testに似たValidaationデータを作るための手法
    - 目的変数をtrain(=1)とtest(=0)にして、2値分類を行うモデルを作る
    - 参考: https://www.acceluniverse.com/blog/developers/2020/01/kaggleadversarial-validation.html


### 2024/05/19
#### Survey

- How far can you go with cheating
  - https://www.kaggle.com/code/andreasbis/how-far-can-you-go-with-cheating/notebook
  - LB: 0.585 -> 0.653 (LB破壊)
  - WEEK_NUMを使わずとも、trainとtestの分布がそもそも違うことを利用して(?)、MetricHackを行う
    - WEEK_NUMを復元するのと、どちらがスコアが伸びるのか

  ```py
  condition=y_pred<0.978
  SHIFT = 0.0718

  df_subm.loc[condition, 'score'] = (df_subm.loc[condition, 'score'] - SHIFT).clip(0)
  ```

### 2024/05/18

- RegressorMixinやTransformerMixinも試してみる

### 2024/05/16

#### exp

- モデルの学習に時間がかかるので、train, inferenceで分ける
  - modelをsave, loadする
- 0.592のコードの変更をいれる
  - https://www.kaggle.com/code/pereradulina/credit-risk-prediction-with-lightgbm-and-catboost/notebook
  - var(分散)の値を追加
  - VotingModelをRegressionではなく、Classification

  ```
  CV AUC scores for CatBoost:  [0.8149291750320602, 0.8135111856042089, 0.8173872597673306, 0.813397177909435, 0.8191254826239982]
  Maximum CV AUC score for CatBoost:  0.8191254826239982
  CV AUC scores for LGBM:  [0.8277140797433427, 0.8288540082560015, 0.8308446622397907, 0.8251448295616548, 0.825535221459149]
  Maximum CV AUC score for LGBM:  0.8308446622397907
  ```

- EDA
  - WEEK_NUMと相関のありそうなカラムを見つける

#### Survey

- Method To Restore WEEK_NUM
  - https://www.kaggle.com/competitions/home-credit-credit-risk-model-stability/discussion/501840
  - `refreshdate_3813885D`を最大にして14日分引くと、日付を復元できる(相関0.91)

### 2024/05/14
#### EXP

- Baselineを提出

```
CV AUC scores:  [0.7434291067318943, 0.7877550255337937, 0.7343818432321955, 0.7014894607946911, 0.7278499778768225]
Maximum CV AUC score:  0.7877550255337937
CV AUC scores:  [0.764815416397035, 0.7891029895411878, 0.738460409620672, 0.726235534915644, 0.7387158304825012]
Maximum CV AUC score:  0.7891029895411878
```

#### Memo

- WEEK_NUMを復元しないと、勝負にならない
  - WEEK_NUMが復元できると、テスト期間の前半のスコアを0.02下げることで、全体のスコアを上げることができるため

#### Survey

- Is the explosion of good scores related to restoring of WEEK_NUM?
  - https://www.kaggle.com/competitions/home-credit-credit-risk-model-stability/discussion/501654
  - Metric hack again, sorryを支持するDiscussion
  - WEEK_NUMを復元して、再開前のようにmetricをhackする
    - 参考: https://www.kaggle.com/competitions/home-credit-credit-risk-model-stability/discussion/476449

- Problem with competition metric
  - metricの問題点
  - このDiscussionを受けて、コンペが一時中断された
  - 問題点: metricの88 * min(0,a)の項が大きすぎるため、最初の数週間のスコアを意図的に悪化させることで、全体のスコアを大幅に改善できる
  - 例の提示: WEEK_NUMがテスト期間の前半である場合にスコアを0.02下げることで、全体のスコアを約0.03改善できる

    ```python
    condition = df_subm['WEEK_NUM'] < (df_subm['WEEK_NUM'].max()-df_subm['WEEK_NUM'].min())/2+df_subm['WEEK_NUM'].min()
    df_subm.loc[condition, 'score'] = (df_subm.loc[condition, 'score'] - 0.02).clip(0)
    df_subm = df_subm[["case_id","score"]]
    df_subm = df_subm.set_index("case_id")
    df_subm.to_csv("submission.csv")

    print(df_subm)
    ```

- We are back - Submissions open on Monday, 11th March 2024
  - https://www.kaggle.com/competitions/home-credit-credit-risk-model-stability/discussion/482474
  - 変更点:
    - test_baseテーブルの日付に関連するカラム（date_decision、MONTH、WEEK_NUM）の変更
    - MONTHとWEEK_NUMは定数値のみを持つようになるが、これらのカラムはスクリプトの変更時間を最小限にするために残される
    - date_decisionカラムは変更され、他のデータも変換されるが、特徴量の意味は保持され、データ型も変わらない。分布に大きな変化はない
  - metric自体は変更なし

### 2024/05/07

#### Memo

- 評価関数
    - gini stability metric
        - gini scoreはWEEK_NUMに対応する予測に対して計算される
        - gini = 2 * AUC - 1
        - ax+bの線形回帰がweekly gini scoreに対して適合しすぎてしまうので、損失を与えるために、min(0, a)の項を加える
        - stability metric = mean(gini) + 88.0 * min(0, a) - 0.5 * std(residuals)

- EDA
  - https://www.kaggle.com/code/sergiosaharovskiy/home-credit-crms-2024-eda-and-submission
- Baseline
  - lgb単体 0.561
      - https://www.kaggle.com/code/aaachen/home-credit-clean-code-lightgbm
  - Ensemble 0.585
      - https://www.kaggle.com/code/shadesh/home-credit-credit-risk-model-stability-v2/notebook

#### Survey

- Metric hack again, sorry
  - https://www.kaggle.com/competitions/home-credit-credit-risk-model-stability/discussion/497167
  - testデータに対してWEEK_NUMを復元できることがわかった

- Test set has been changed?
  - https://www.kaggle.com/competitions/home-credit-credit-risk-model-stability/discussion/499505
  - 再スタート後もtestデータは変更されていない

- some tips to speed up your experiments
  - https://www.kaggle.com/competitions/home-credit-credit-risk-model-stability/discussion/498336
  - 高速で実験結果を検証するTIPS
    - fold数を減らす
      - 3にすると、1.6倍早い
    - lrを上げて、iterationを少なくする
    - 特徴量を下げる
      - すべての特徴量の組み合わせを毎回試すのではなく、小さなグループで試す
---

### 2024/03/10
#### EXP

- DCN V2実装
    - 参考になりそうなCode:
    - TODO
      - カテゴリ変数をintにする
      - 欠損値の扱い
- metricsの安定化に伴って、WEEK_NUMなどのカラムがなくなるので、再開するまで一旦様子見

#### Survey

- Similar competitions important kernels and discussions for references
  - https://www.kaggle.com/competitions/home-credit-credit-risk-model-stability/discussion/473973
  - 過去の類似コンペ(=American Express - Default Prediction)の紹介

- 日本語_特徴量の日本語訳_Japanese_translation
  - https://www.kaggle.com/code/pandaman817/japanese-translation/notebook
  - 各カラムの説明がついている

- [placeholder] novel ideas (stability prize) and experiment results
  - https://www.kaggle.com/competitions/home-credit-credit-risk-model-stability/discussion/475516
  - LLMのEncoderを用いる方法
    - column情報からLLMでユーザ情報を出力してもらい、Embeddingを作る
  - basic modelを通常のlossで学習
  - metricを修正
  - パフォーマンスの上限を得るために、train dataをoverfitさせる
  - 安定性を考慮したmodel explainabilityと考慮しないmodel explainability(時間経過に伴うexplainability)

- What is the meaning of num_group1 and num_group2 in depth1 and 2 dataset?
  - https://www.kaggle.com/competitions/home-credit-credit-risk-model-stability/discussion/476763
  - このコンペの目的は、顧客（正確には顧客のアプリケーション）のリスクの質を予測することなので、顧客とそのアプリケーションは、あなたのモデルがデフォルトの確率（クレジットスコア）を予測するためのエンティティ（case_id）です。
  - depth=0（静的属性と呼ばれるもの）は、case_id レベルで集約される属性です。例として、顧客の年齢や性別が挙げられます。
  - depth=1（静的属性と呼ばれるもの）は、クライアント/アプリケーションごとに複数のレコードを持つ属性
    - 例としては、信用情報機関に登録されている過去の申請やローンが挙げられ、各クライアントは 0 から n までのレコードを持つことができます。したがって、1つの case_id に対して複数のレコードが存在する可能性があり、それらのインデックスを作成するために num_group1 を使用します。
  - depth=2: depth=1 の属性では、より詳細な情報が得られます。例えば、過去の申込については、支払日や支払期日経過日数などの分割払いのデータが得られます。つまり、過去の各申請について、分割払い/支払いに関する0～n個のレコードを持つことができます。インデックスとして num_group2 を使用します。
  - まとめると、1人の顧客は複数の過去の申請書を持つことができ、それらの過去の申請書はそれぞれ、分割払い、支払い、期限切れ日数などに関する複数のレコードを持つことができます。
