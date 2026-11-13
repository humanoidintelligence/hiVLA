"""Tests for the extended transform set and norm modes."""

from __future__ import annotations

import numpy as np
import pytest

from hivla.core.transforms import (
    AsFloat,
    Normalize,
    NormStats,
    PadToDim,
    RenameKeys,
    ResizeImages,
    SelectKeys,
    SinCos,
    Slice,
    Unnormalize,
    compose,
)


def test_norm_mode_quantile_roundtrip():
    st = NormStats(q01=np.array([0.0, 10.0]), q99=np.array([100.0, 200.0]), mode="quantile")
    x = {"state": np.array([50.0, 105.0], dtype=np.float32)}
    norm, unnorm = Normalize({"state": st}), Unnormalize({"state": st})
    back = unnorm(norm(dict(x)))
    assert np.allclose(back["state"], x["state"], atol=1e-4)


def test_norm_mode_minmax_roundtrip():
    st = NormStats(min=np.array([-1.0]), max=np.array([1.0]), mode="minmax")
    x = {"a": np.array([0.25], dtype=np.float32)}
    back = Unnormalize({"a": st})(Normalize({"a": st})(dict(x)))
    assert np.allclose(back["a"], x["a"], atol=1e-5)


def test_norm_infers_mode_from_fields():
    st = NormStats(q01=np.array([0.0]), q99=np.array([10.0]))
    assert st.resolve_mode() == "quantile"


def test_pad_to_dim():
    out = PadToDim("state", 5)({"state": np.zeros((2, 3), dtype=np.float32)})
    assert out["state"].shape == (2, 5)
    assert np.all(out["state"][:, 3:] == 0.0)


def test_slice():
    out = Slice("action", 2)({"action": np.arange(6, dtype=np.float32).reshape(3, 2)})
    # slicing a 2-wide feature axis to 2 is a no-op
    assert out["action"].shape == (3, 2)
    out = Slice("action", 1)({"action": np.ones((3, 3), dtype=np.float32)})
    assert out["action"].shape == (3, 1)


def test_sincos_doubles_last_axis():
    x = np.array([[0.0, np.pi / 2]], dtype=np.float32)
    out = SinCos("state")({"state": x})
    assert out["state"].shape == (1, 4)
    assert np.allclose(out["state"][0, 0], np.sin(0.0))
    assert np.allclose(out["state"][0, 2], np.sin(np.pi / 2))


def test_select_keys():
    out = SelectKeys(["a"])({"a": 1, "b": 2})
    assert list(out) == ["a"]


def test_as_float_image_scale():
    out = AsFloat("img", scale=2.0 / 255.0, offset=-1.0)({"img": np.full((1, 1, 3), 255, np.uint8)})
    assert out["img"].dtype == np.float32
    assert np.allclose(out["img"], 1.0)


def test_compose_with_new_transforms():
    f = compose([RenameKeys({"joint_pos": "state"}), Normalize({"state": NormStats(mean=np.array([1.0]), std=np.array([2.0]))})])
    out = f({"joint_pos": np.array([3.0], dtype=np.float32)})
    assert np.allclose(out["state"], 1.0)  # (3-1)/2


def test_resize_images_roundtrip_shape():
    pytest.importorskip("PIL")
    img = np.zeros((10, 20, 3), dtype=np.uint8)
    out = ResizeImages("img", 8)({"img": img})
    assert out["img"].shape == (8, 8, 3)
