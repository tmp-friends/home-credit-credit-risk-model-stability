
## Survey
### Home Credit - Discussion

#### Similar competitions important kernels and discussions for references

https://www.kaggle.com/competitions/home-credit-credit-risk-model-stability/discussion/473973

- 過去の類似コンペ(=American Express - Default Prediction)の紹介


#### 日本語_特徴量の日本語訳_Japanese_translation

https://www.kaggle.com/code/pandaman817/japanese-translation/notebook

- 各カラムの説明がついている

#### [placeholder] novel ideas (stability prize) and experiment results

https://www.kaggle.com/competitions/home-credit-credit-risk-model-stability/discussion/475516

- LLMのEncoderを用いる方法
  - column情報からLLMでユーザ情報を出力してもらい、Embeddingを作る

- basic modelを通常のlossで学習
- metricを修正
- パフォーマンスの上限を得るために、train dataをoverfitさせる
- 安定性を考慮したmodel explainabilityと考慮しないmodel explainability(時間経過に伴うexplainability)

#### What is the meaning of num_group1 and num_group2 in depth1 and 2 dataset?

https://www.kaggle.com/competitions/home-credit-credit-risk-model-stability/discussion/476763

このコンペの目的は、顧客（正確には顧客のアプリケーション）のリスクの質を予測することなので、顧客とそのアプリケーションは、あなたのモデルがデフォルトの確率（クレジットスコア）を予測するためのエンティティ（case_id）です。

- depth=0（静的属性と呼ばれるもの）は、case_id レベルで集約される属性です。例として、顧客の年齢や性別が挙げられます。

- depth=1（静的属性と呼ばれるもの）は、クライアント/アプリケーションごとに複数のレコードを持つ属性
  - 例としては、信用情報機関に登録されている過去の申請やローンが挙げられ、各クライアントは 0 から n までのレコードを持つことができます。したがって、1つの case_id に対して複数のレコードが存在する可能性があり、それらのインデックスを作成するために num_group1 を使用します。

- depth=2: depth=1 の属性では、より詳細な情報が得られます。例えば、過去の申込については、支払日や支払期日経過日数などの分割払いのデータが得られます。つまり、過去の各申請について、分割払い/支払いに関する0～n個のレコードを持つことができます。インデックスとして num_group2 を使用します。

まとめると、1人の顧客は複数の過去の申請書を持つことができ、それらの過去の申請書はそれぞれ、分割払い、支払い、期限切れ日数などに関する複数のレコードを持つことができます。

