"""Transformer model utilities for financial time-series forecasting."""

from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Dict, Tuple

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


@dataclass
class TransformerConfig:
    """Hyperparameters for the Transformer encoder model."""

    sequence_length: int = 30
    num_features: int = 8
    d_model: int = 64
    num_heads: int = 4
    ff_dim: int = 128
    num_transformer_blocks: int = 2
    dropout: float = 0.1
    learning_rate: float = 1e-3
    use_learned_positional_encoding: bool = False


class PositionalEncoding(layers.Layer):
    """Adds either sinusoidal or learned positional embeddings."""

    def __init__(
        self,
        sequence_length: int,
        d_model: int,
        learned: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.sequence_length = sequence_length
        self.d_model = d_model
        self.learned = learned
        if self.learned:
            self.position_embedding = layers.Embedding(
                input_dim=sequence_length, output_dim=d_model
            )
        else:
            self.position_embedding = None
            self.positional_tensor = self._build_sinusoidal_encoding(
                sequence_length=sequence_length, d_model=d_model
            )

    @staticmethod
    def _build_sinusoidal_encoding(sequence_length: int, d_model: int) -> tf.Tensor:
        positions = np.arange(sequence_length)[:, np.newaxis]
        dims = np.arange(d_model)[np.newaxis, :]

        angle_rates = 1.0 / np.power(10000.0, (2 * (dims // 2)) / float(d_model))
        angle_rads = positions * angle_rates

        pos_encoding = np.zeros((sequence_length, d_model), dtype=np.float32)
        pos_encoding[:, 0::2] = np.sin(angle_rads[:, 0::2])
        pos_encoding[:, 1::2] = np.cos(angle_rads[:, 1::2])

        return tf.convert_to_tensor(pos_encoding[np.newaxis, ...], dtype=tf.float32)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        seq_len = tf.shape(inputs)[1]
        if self.learned and self.position_embedding is not None:
            positions = tf.range(start=0, limit=seq_len, delta=1)
            pos_embed = self.position_embedding(positions)
            pos_embed = tf.expand_dims(pos_embed, axis=0)
            return inputs + pos_embed

        return inputs + self.positional_tensor[:, :seq_len, :]

    def get_config(self) -> Dict[str, object]:
        config = super().get_config()
        config.update(
            {
                "sequence_length": self.sequence_length,
                "d_model": self.d_model,
                "learned": self.learned,
            }
        )
        return config


def _transformer_encoder_block(
    x: tf.Tensor,
    d_model: int,
    num_heads: int,
    ff_dim: int,
    dropout: float,
    block_index: int,
) -> tf.Tensor:
    head_dim = d_model // num_heads

    attention_out = layers.MultiHeadAttention(
        num_heads=num_heads,
        key_dim=head_dim,
        dropout=dropout,
        name=f"mha_{block_index}",
    )(x, x)
    attention_out = layers.Dropout(dropout, name=f"attn_dropout_{block_index}")(
        attention_out
    )
    x = layers.Add(name=f"attn_residual_{block_index}")([x, attention_out])
    x = layers.LayerNormalization(epsilon=1e-6, name=f"attn_ln_{block_index}")(x)

    ff = layers.Dense(ff_dim, activation="relu", name=f"ffn_relu_{block_index}")(x)
    ff = layers.Dropout(dropout, name=f"ffn_dropout_1_{block_index}")(ff)
    ff = layers.Dense(d_model, name=f"ffn_proj_{block_index}")(ff)
    ff = layers.Dropout(dropout, name=f"ffn_dropout_2_{block_index}")(ff)

    x = layers.Add(name=f"ffn_residual_{block_index}")([x, ff])
    x = layers.LayerNormalization(epsilon=1e-6, name=f"ffn_ln_{block_index}")(x)
    return x


def build_transformer_model(config: TransformerConfig) -> keras.Model:
    """Build a Transformer encoder model for one-step-ahead return forecasting."""
    if config.d_model % config.num_heads != 0:
        raise ValueError("d_model must be divisible by num_heads.")
    if config.num_transformer_blocks < 1 or config.num_transformer_blocks > 3:
        raise ValueError("num_transformer_blocks must be in [1, 3].")

    inputs = keras.Input(
        shape=(config.sequence_length, config.num_features), name="inputs"
    )

    x = layers.Dense(config.d_model, name="input_projection")(inputs)
    x = PositionalEncoding(
        sequence_length=config.sequence_length,
        d_model=config.d_model,
        learned=config.use_learned_positional_encoding,
        name="positional_encoding",
    )(x)

    for block_id in range(config.num_transformer_blocks):
        x = _transformer_encoder_block(
            x,
            d_model=config.d_model,
            num_heads=config.num_heads,
            ff_dim=config.ff_dim,
            dropout=config.dropout,
            block_index=block_id,
        )

    x = layers.GlobalAveragePooling1D(name="global_avg_pool")(x)
    x = layers.Dropout(config.dropout, name="head_dropout")(x)
    x = layers.Dense(max(config.ff_dim // 2, 16), activation="relu", name="head_relu")(x)
    x = layers.Dropout(config.dropout, name="head_dropout_2")(x)
    outputs = layers.Dense(1, name="forecast")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="transformer_forecaster")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config.learning_rate),
        loss="mse",
        metrics=["mae"],
    )
    return model


def train_transformer(
    model: keras.Model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int,
    batch_size: int,
    verbose: int = 0,
) -> keras.callbacks.History:
    """Train the Transformer using validation-based early stopping."""
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=12, restore_best_weights=True
        )
    ]

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=verbose,
    )
    return history


def predict_transformer(model: keras.Model, X: np.ndarray) -> np.ndarray:
    """Predict one-step-ahead returns."""
    preds = model.predict(X, verbose=0)
    return preds.reshape(-1)


def set_random_seed(seed: int) -> None:
    """Set seeds for reproducibility across NumPy, Python, and TensorFlow."""
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)


def model_summary_string(model: keras.Model) -> str:
    """Capture Keras model summary as a string for Streamlit display."""
    lines: list[str] = []
    model.summary(print_fn=lambda x: lines.append(x))
    return "\n".join(lines)
