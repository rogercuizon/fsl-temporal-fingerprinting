from types import SimpleNamespace

import numpy as np

from src.fingerprint import make_fingerprint
from src.mobile_sequence_extractor import (HAND_SLOT_FEATURES,
                                           SEQUENCE_FEATURE_COUNT,
                                           encode_two_hand_frame)


def sample_hand(offset=0.0):
    return [SimpleNamespace(x=offset + i * 0.01, y=0.2 + i * 0.02, z=i * 0.005)
            for i in range(21)]


def test_single_hand_is_encoded_in_its_handedness_slot_with_same_70_features():
    left = sample_hand()
    result = encode_two_hand_frame([left], [[SimpleNamespace(category_name="Left")]], 640, 480)
    assert result.shape == (SEQUENCE_FEATURE_COUNT,)
    assert result[2 * HAND_SLOT_FEATURES] == 1
    assert result[2 * HAND_SLOT_FEATURES + 1] == 0
    assert np.allclose(result[:70], make_fingerprint([(p.x, p.y, p.z) for p in left]))
    assert np.isfinite(result).all()


def test_both_hands_are_encoded_and_inter_hand_offset_is_retained():
    left, right = sample_hand(0.1), sample_hand(0.7)
    result = encode_two_hand_frame(
        [left, right],
        [[SimpleNamespace(category_name="Left")],
         [SimpleNamespace(category_name="Right")]],
        640, 480)
    assert result.shape == (151,)
    assert result[2 * HAND_SLOT_FEATURES:2 * HAND_SLOT_FEATURES + 2].tolist() == [1.0, 1.0]
    assert np.isclose(result[148], 0.6, atol=1e-6)
    assert np.allclose(result[73:143], make_fingerprint([(p.x, p.y, p.z) for p in right]))
    assert np.isfinite(result).all()
