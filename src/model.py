"""Tensorflow models to be trained"""
import tensorflow as tf


class BaseAttention(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__()
        self.mha = tf.keras.layers.MultiHeadAttention(**kwargs)
        self.add = tf.keras.layers.Add()
        self.normalize = tf.keras.layers.LayerNormalization()


class GlobalSelfAttention(BaseAttention):
    def __call__(self, x):
        attn_output = self.mha(
            query=x,
            value=x,
            key=x
        )

        x = self.add([x, attn_output])
        x = self.normalize(x)

        return x


class ConvBlock(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__()

        self.conv_layer = tf.keras.layers.SeparableConv1D(**kwargs)
        self.batchnorm_layer = tf.keras.layers.BatchNormalization()

    def __call__(self, x):
        x = self.conv_layer(x)
        x = self.batchnorm_layer(x)

        return x


class GISLRPreProcess(tf.keras.layers.Layer):
    """
    Preprocessing layer for including in the tf lite model
    """

    # Subsets the points
    # Reshape the file

    def __init__(self):
        super(GISLRPreProcess, self).__init__()

    def call(self, inputs):
        x = inputs[:, :2]
        x = tf.reshape(x, (x.shape[0], x.shape[1] * 2))
        return x


class GISLRModel():
    def __init__(
            self, n_classes: int,
            input_shape=(32, 1629)
        ) -> None:

        model_input = tf.keras.Input(shape=input_shape)

        # x = GISLRPreProcess()(model_input)
        x = ConvBlock(filters=256, kernel_size=7, padding="same")(model_input)
        x = tf.keras.layers.Dense(256, activation="relu")(x)
        x = tf.keras.layers.Dropout(0.5)(x)
        x = tf.keras.layers.GlobalAveragePooling1D()(x)
        output = tf.keras.layers.Dense(n_classes, activation="softmax")(x)

        self.model = tf.keras.Model(model_input, output)

    def get_model(self):
        smoothing=0.1

        self.model.compile(
            loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=smoothing),
            optimizer=tf.keras.optimizers.Adam(learning_rate=3e-4),
            metrics=[tf.keras.metrics.CategoricalAccuracy()]
        )

        print(self.model.summary())

        return self.model


class GISLRModelv2():
    def __init__(
        self, n_classes: int,
        input_shape=(32, 1629)
    ) -> None:

        model_input = tf.keras.Input(shape=input_shape)

        x = GlobalSelfAttention(
            num_heads=4,
            key_dim=128,
            # output_shape=128,
            dropout=0.25
        )(model_input)
        x = tf.keras.layers.Dense(256, activation="relu")(x)
        x = tf.keras.layers.Dropout(0.5)(x)
        x = tf.keras.layers.GlobalAveragePooling1D()(x)
        output = tf.keras.layers.Dense(n_classes, activation="softmax")(x)

        self.model = tf.keras.Model(model_input, output)

    def get_model(self):
        self.model.compile(
            loss=tf.keras.losses.CategoricalCrossentropy(),
            optimizer=tf.keras.optimizers.Adam(learning_rate=3e-4),
            metrics=[tf.keras.metrics.CategoricalAccuracy()]
        )

        print(self.model.summary())

        return self.model


if __name__ == "__main__":
    model = GISLRModel(n_classes = 250).get_model()
