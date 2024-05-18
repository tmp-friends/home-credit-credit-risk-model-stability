import tensorflow as tf


class Trainer(tf.keras.Model):
    def __init__(self, model, predictors):
        super().__init__()

        self.model = model
        self.predictors = predictors

        self.eps = 1e-9

    def call(self, inputs):
        """性能評価処理

        Args:
            inputs
        """
        unique_emb = self.model(inputs["unique_feature"])

        loss_sum = 0.0
        pos_true_positive = 0.0
        pos_false_positive = 0.0
        pos_false_negative = 0.0
        neg_true_positive = 0.0
        neg_false_positive = 0.0
        neg_false_negative = 0.0
        correct = 0.0

        # TODO: 3となっているが, case_idは一意なのでforがいらない
        for i in range(3):
            pred_emb = tf.reduce_sum(
                tf.gather(unique_emb, inputs[f"history_{i}"]), axis=1
            )
            pred_val = tf.clip_by_value(
                tf.math.sigmoid(
                    self.predictors[i](tf.nn.l2_normalize(pred_emb, axis=1))
                ),
                self.eps,
                1.0 - self.eps,
            )
            # Binary Cross Entropy
            loss = -inputs[f"label_{i}"] * tf.math.log(pred_val) - (
                1.0 - inputs[f"label_{i}"]
            ) * tf.math.log(1.0 - pred_val)
            loss_sum += tf.reduce_sum(tf.reduce_mean(loss, axis=0))

            # F1-macro
            pred_label = pred_val > 0.5
            bool_label = inputs[f"label_{i}"] == 1.0
            correct += tf.cast(
                tf.math.count_nonzero(pred_label == bool_label), "float32"
            )
            pos_true_positive += tf.cast(
                tf.math.count_nonzero(tf.math.logical_and(bool_label, pred_label)),
                "float32",
            )
            pos_false_positive += tf.cast(
                tf.math.count_nonzero(
                    tf.math.logical_and(tf.math.logical_not(bool_label), pred_label)
                ),
                "float32",
            )
            pos_false_negative += tf.cast(
                tf.math.count_nonzero(
                    tf.math.logical_and(bool_label, tf.math.logical_not(pred_label))
                ),
                "float32",
            )

            pred_label = pred_val < 0.5
            bool_label = inputs[f"label_{i}"] == 0.0
            neg_true_positive += tf.cast(
                tf.math.count_nonzero(tf.math.logical_and(bool_label, pred_label)),
                "float32",
            )
            neg_false_positive += tf.cast(
                tf.math.count_nonzero(
                    tf.math.logical_and(tf.math.logical_not(bool_label), pred_label)
                ),
                "float32",
            )
            neg_false_negative += tf.cast(
                tf.math.count_nonzero(
                    tf.math.logical_and(bool_label, tf.math.logical_not(pred_label))
                ),
                "float32",
            )

        accuracy = correct / tf.cast(tf.shape(inputs[f"label_0"])[0] * 18, "float32")
        pos_recall = pos_true_positive / tf.maximum(
            self.eps, pos_true_positive + pos_false_negative
        )
        pos_precision = pos_true_positive / tf.maximum(
            self.eps, pos_true_positive + pos_false_positive
        )
        pos_f1 = (
            2
            * pos_recall
            * pos_precision
            / tf.maximum(self.eps, pos_recall + pos_precision)
        )

        neg_recall = neg_true_positive / tf.maximum(
            self.eps, neg_true_positive + neg_false_negative
        )
        neg_precision = neg_true_positive / tf.maximum(
            self.eps, neg_true_positive + neg_false_positive
        )
        neg_f1 = (
            2
            * neg_recall
            * neg_precision
            / tf.maximum(self.eps, neg_recall + neg_precision)
        )

        return loss_sum, (pos_f1 + neg_f1) / 2.0, accuracy

    def predict_proba(self, inputs):
        unique_emb = self.model(inputs["unique_feature"])

        pred_emb = tf.reduce_sum(tf.gather(unique_emb, inputs["case_id"]), axis=1)
        pred_val = tf.clip_by_value(
            tf.math.sigmoid(self.predictors[i](tf.nn.l2_normalize(pred_emb, axis=1))),
            clip_value_min=self.eps,
            clip_value_max=1.0 - self.eps,
        )

        label = input["target"]

        return pred_val, label
