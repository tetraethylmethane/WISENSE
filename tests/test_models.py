"""The rebuilt architecture is exactly the published one.

Proof by weights: the authors' trained .keras files are loaded, their weights
copied by layer name into our graph, and both must produce identical outputs.
Skipped when TensorFlow or the authors' repository is not present.
"""

import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")

from wisense.data import RAW_DIR  # noqa: E402
from wisense.models import ARCHS, build  # noqa: E402

PUBLISHED = RAW_DIR.parents[1] / "Embedded_Model"
FILES = {"small": "small_model_complete_EMD_LSTM.keras", "large": "model_complete_synth_IMF.keras"}


@pytest.mark.parametrize("arch", ["small", "large"])
def test_parameter_count_matches_paper(arch):
    assert build(ARCHS[arch]).count_params() == ARCHS[arch].params


@pytest.mark.parametrize("arch", ["small", "large"])
def test_published_weights_load_and_agree(arch):
    path = PUBLISHED / FILES[arch]
    if not path.exists():
        pytest.skip(f"{path} not present")
    theirs = tf.keras.models.load_model(path, compile=False)
    ours = build(ARCHS[arch])
    assert [l.name for l in ours.layers] == [l.name for l in theirs.layers]
    for mine, src in zip(ours.layers, theirs.layers):
        mine.set_weights(src.get_weights())
    x = np.random.default_rng(0).uniform(0, 30, (4, 850, 1)).astype(np.float32)
    assert np.allclose(ours(x, training=False), theirs(x, training=False), atol=1e-5)
