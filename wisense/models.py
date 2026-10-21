"""LSTM-DenseNet for single-subcarrier CSI activity recognition.

Rebuilt from the config.json inside the authors' published .keras files, not
from the paper's prose -- the two differ in places that change the weights:

  * Transitions output a fixed 5 channels. The paper's "compression 0.5" is
    applied to the growth rate (int(11 * 0.5) = 5), not to the channel count
    as in DenseNet. Every dense block therefore starts from 5 channels.
  * Transitions are BN -> 1x1 conv -> avgpool, with no ReLU.
  * The stem is conv(7, stride 2) -> maxpool(3, stride 2), with no BN between.
  * The LSTM is unrolled, which TFLite Micro requires (no while-loop op).

Layer names match the originals so the published weights load into this
graph; tests/test_models.py uses that to prove the rebuild is exact.

    PAPER_SMALL  "LSTM-Embedded DenseNet"   blocks 3/6/12/8,  LSTM 32   122,383 params
    PAPER_LARGE  "LSTM-DenseNet121"         blocks 6/12/24/16, LSTM 128 437,234 params
"""

from __future__ import annotations

from dataclasses import dataclass

from wisense.csi import TIMESTEPS
from wisense.data import ACTIVITIES


@dataclass(frozen=True)
class Arch:
    blocks: tuple[int, ...]
    lstm_units: int
    growth: int = 11
    bottleneck: int = 4          # 1x1 conv width = bottleneck * growth
    transition_filters: int = 5
    dropout: float = 0.3
    params: int | None = None    # expected count, asserted in tests


PAPER_SMALL = Arch(blocks=(3, 6, 12, 8), lstm_units=32, params=122_383)
PAPER_LARGE = Arch(blocks=(6, 12, 24, 16), lstm_units=128, params=437_234)
ARCHS = {"small": PAPER_SMALL, "large": PAPER_LARGE}


def build(arch: Arch = PAPER_SMALL, timesteps: int = TIMESTEPS, channels: int = 1,
          n_classes: int = len(ACTIVITIES)):
    from tensorflow import keras
    L = keras.layers

    def conv(x, filters, k, name, strides=1):
        return L.Conv1D(filters, k, strides=strides, padding="same", use_bias=False,
                        name=name)(x)

    def bn_relu_conv(x, filters, k, prefix):
        x = L.BatchNormalization(name=f"{prefix}_batch_norm")(x)
        x = L.ReLU(name=f"{prefix}_relu")(x)
        return conv(x, filters, k, f"{prefix}_conv1d")

    inp = L.Input((timesteps, channels), name="input_layer")
    x = conv(inp, arch.growth, 7, "in_conv1d", strides=2)
    x = L.MaxPooling1D(3, strides=2, name="in_pool")(x)

    n_concat = 0
    for b, n_layers in enumerate(arch.blocks, start=1):
        for i in range(n_layers):
            p = f"dn{b}_l{i}"
            y = bn_relu_conv(x, arch.bottleneck * arch.growth, 1, f"{p}_bn")
            y = bn_relu_conv(y, arch.growth, 3, f"{p}_mc")
            y = L.Dropout(arch.dropout, name=f"{p}_mc_dropout")(y)
            x = L.Concatenate(name="concatenate" + (f"_{n_concat}" if n_concat else ""))([x, y])
            n_concat += 1
        x = L.BatchNormalization(name=f"tr{b}_tr_batch_norm")(x)
        x = conv(x, arch.transition_filters, 1, f"tr{b}_tr_conv1d")
        x = L.AveragePooling1D(2, strides=2, name=f"tr{b}_tr_avgpool")(x)

    x = L.LSTM(arch.lstm_units, return_sequences=True, unroll=True, name="lstm")(x)
    x = L.GlobalAveragePooling1D(name="global_average_pooling1d")(x)
    out = L.Dense(n_classes, activation="softmax", name="dense")(x)
    return keras.Model(inp, out, name="lstm_densenet")


def compile_paper(model, learning_rate: float = 1e-4):
    """Adam 1e-4, categorical cross-entropy -- from the published compile_config."""
    from tensorflow import keras
    model.compile(optimizer=keras.optimizers.Adam(learning_rate),
                  loss="categorical_crossentropy", metrics=["accuracy"])
    return model
