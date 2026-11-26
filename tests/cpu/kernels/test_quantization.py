#!/usr/bin/env python
"""The block-scale quantizers (MXFP8 / MXFP4 / NVFP4) in ``halo/kernels/lowp/quantization.py``."""

import contextlib
import warnings

import pytest
import torch
from torch._dynamo.exc import TorchDynamoException
from torch.overrides import TorchFunctionMode

from halo.kernels.lowp import quantization
from halo.kernels.lowp.quantization import (
    E2M1_MAX,
    cached_fake_quant,
    dequantize,
    fake_quant,
    quantize_mxfp4,
    quantize_mxfp8,
    quantize_nvfp4,
)
from tests.common.numerics import fro_rel_err

# Each format's intrinsic round-trip tolerance (fp8 ~ 8-bit, fp4 ~ 4-bit).
_TOL = {"mxfp8": 0.05, "mxfp4": 0.25, "nvfp4": 0.25}


def test_mxfp8_storage_shapes_and_dtypes():
    torch.manual_seed(0)
    x = torch.randn(64, 256) * 0.5
    q = quantize_mxfp8(x, axis=-1)
    assert q.data.shape == x.shape and q.data.dtype == torch.float8_e4m3fn
    assert q.scales.shape == (64, 256 // 32) and q.scales.dtype == torch.uint8
    assert not q.packed and q.pow2_scale


def test_nvfp4_storage_shapes_and_dtypes():
    torch.manual_seed(0)
    x = torch.randn(64, 256) * 0.5
    q = quantize_nvfp4(x, axis=-1)
    assert q.data.shape == (64, 256 // 2) and q.data.dtype == torch.uint8  # 2 nibbles / byte.
    assert q.scales.shape == (64, 256 // 16) and q.scales.dtype == torch.float8_e4m3fn
    assert q.packed and not q.pow2_scale
    # Two-level scaling: a scalar fp32 multiplier accompanies the e4m3 block scales.
    assert q.global_scale is not None and q.global_scale.numel() == 1 and q.global_scale.dtype == torch.float32


def test_mxfp4_storage_shapes_and_dtypes():
    torch.manual_seed(0)
    x = torch.randn(64, 256) * 0.5
    q = quantize_mxfp4(x, axis=-1)
    assert q.data.shape == (64, 256 // 2) and q.data.dtype == torch.uint8
    assert q.scales.shape == (64, 256 // 32) and q.scales.dtype == torch.uint8
    assert q.packed and q.pow2_scale
    assert q.global_scale is None  # MX is single-level by definition.


def test_roundtrip_within_tolerance_all_formats():
    torch.manual_seed(0)
    x = torch.randn(64, 256) * 0.5
    for fmt, quantizer in (("mxfp8", quantize_mxfp8), ("mxfp4", quantize_mxfp4), ("nvfp4", quantize_nvfp4)):
        recon = dequantize(quantizer(x))
        assert recon.dtype == torch.bfloat16
        re = fro_rel_err(recon, x)
        assert re < _TOL[fmt], f"{fmt} round-trip relerr {re:.4f} >= {_TOL[fmt]}"


def test_axis_0_quantization():
    torch.manual_seed(0)
    x = torch.randn(128, 96) * 0.3
    q = quantize_mxfp8(x, axis=0)
    assert q.scales.shape == (128 // 32, 96)
    assert fro_rel_err(dequantize(q), x) < _TOL["mxfp8"]


def test_zero_input_roundtrips_to_zero():
    # The min-clamp on amax must not turn an all-zero block into NaN.
    x = torch.zeros(32, 64)
    for quantizer in (quantize_mxfp8, quantize_mxfp4, quantize_nvfp4):
        recon = dequantize(quantizer(x))
        assert torch.isfinite(recon).all()
        assert recon.abs().max().item() == 0.0


def test_e2m1_midpoints_round_to_even_not_down():
    """``bucketize`` resolves every tie down, and exact ties are routine, which would bias every fp4 tensor toward
    zero."""
    block = torch.zeros(1, 32)
    block[0, 0] = E2M1_MAX  # Pins the block scale at 1.0.
    block[0, 1:4] = torch.tensor([0.75, 1.75, 3.5])
    block[0, 4] = -0.75  # Sign must not change which way a tie goes.
    block[0, 5:7] = torch.tensor([0.7, 0.8])  # Anti-vacuity: non-ties still round to nearest.

    recon = dequantize(quantize_mxfp4(block))

    assert recon[0, 0].item() == E2M1_MAX
    assert [recon[0, i].item() for i in (1, 2, 3)] == [1.0, 2.0, 4.0]  # Tie-down would give 0.5/1.5/3.0.
    assert recon[0, 4].item() == -1.0
    assert [recon[0, i].item() for i in (5, 6)] == [0.5, 1.0]


def test_divisibility_guard_raises():
    x = torch.randn(8, 30)  # 30 is divisible by neither 32 (mxfp8/mxfp4) nor 16 (nvfp4).
    for quantizer in (quantize_mxfp8, quantize_mxfp4, quantize_nvfp4):
        try:
            quantizer(x, axis=-1)
        except ValueError:
            continue
        raise AssertionError(f"{quantizer.__name__} should raise on a non-divisible axis")


def test_fake_quant_matches_roundtrip_and_ste_backward():
    torch.manual_seed(0)
    for fmt in ("mxfp8", "nvfp4"):
        x = (torch.randn(64, 256) * 0.3).requires_grad_(True)
        y = fake_quant(x, fmt, axis=-1)
        assert fro_rel_err(y.detach(), x.detach()) < _TOL[fmt]
        # Backward is the identity.
        (y * 2.0).sum().backward()
        assert torch.allclose(x.grad, torch.full_like(x.grad, 2.0)), "STE backward must be identity"


def test_fake_quant_preserves_input_dtype():
    x = (torch.randn(32, 64) * 0.2).to(torch.bfloat16)
    assert fake_quant(x, "mxfp8").dtype == torch.bfloat16


def test_cached_fake_quant_matches_fake_quant():
    torch.manual_seed(0)
    w = torch.nn.Parameter(torch.randn(8, 64) * 0.3)
    cached = cached_fake_quant(w, "mxfp8", axis=-1)
    direct = fake_quant(w.detach(), "mxfp8", axis=-1)
    assert torch.allclose(cached.detach(), direct, atol=1e-5)


def test_cached_fake_quant_reuses_until_version_bumps():
    torch.manual_seed(0)
    w = torch.nn.Parameter(torch.randn(8, 64) * 0.3)
    first = cached_fake_quant(w, "mxfp8", axis=-1).detach().clone()
    again = cached_fake_quant(w, "mxfp8", axis=-1).detach()
    assert torch.equal(first, again)
    with torch.no_grad():
        w.add_(1.0)
    after = cached_fake_quant(w, "mxfp8", axis=-1).detach()
    assert not torch.equal(first, after)
    assert fro_rel_err(after, w.detach()) < _TOL["mxfp8"]


def test_cached_fake_quant_is_straight_through():
    w = torch.nn.Parameter(torch.randn(8, 64) * 0.3)
    (cached_fake_quant(w, "mxfp8", axis=-1) * 3.0).sum().backward()
    assert w.grad is not None and torch.allclose(w.grad, torch.full_like(w.grad, 3.0))


class _TensorsFromHostData(TorchFunctionMode):
    """Each ``torch.tensor`` is a blocking host-to-device copy on a GPU."""

    def __init__(self):
        super().__init__()
        self.calls = 0

    def __torch_function__(self, func, types, args=(), kwargs=None):
        if func is torch.tensor:
            self.calls += 1
        return func(*args, **(kwargs or {}))


@pytest.mark.parametrize("fmt", ["mxfp4", "nvfp4"])
def test_fp4_eager_path_builds_its_tables_once(fmt):
    x = torch.randn(8, 64)
    fake_quant(x, fmt)
    with _TensorsFromHostData() as probe:
        for _ in range(3):
            fake_quant(x, fmt)
    assert probe.calls == 0


@pytest.mark.parametrize("fmt", quantization._COMPILABLE)
def test_compiled_round_trip_matches_eager_without_tracing_the_cache(fmt):
    torch._dynamo.reset()
    x = torch.randn(64, 128) * 3
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        compiled = torch.compile(quantization._block_round_trip, dynamic=True)(x, fmt, -1)
    assert torch.equal(compiled, quantization._block_round_trip(x, fmt, -1))
    assert not [w for w in caught if "lru_cache" in str(w.message)]


@contextlib.contextmanager
def _flushed_denormals():
    """CPU flush-to-zero, as a compiled CUDA graph applies it, restored on exit."""
    if not torch.set_flush_denormal(True):
        pytest.skip("this CPU cannot flush denormals")
    try:
        # The flush must reach the scale the bug came from, or the test proves nothing.
        assert torch.exp2(torch.tensor(-127.0)).item() == 0.0
        yield
    finally:
        torch.set_flush_denormal(False)


def _with_zero_blocks() -> torch.Tensor:
    """The repro from #151: one expert's first 32 rows zero, so each of its columns has an all-zero block on axis 1."""
    torch.manual_seed(0)
    w = torch.randn(4, 64, 32, dtype=torch.bfloat16) * 0.05
    w[1, :32, :] = 0
    return w


@pytest.mark.parametrize("fmt", ["mxfp8", "mxfp4", "nvfp4"])
def test_an_all_zero_block_round_trips_to_zero_under_flushed_denormals(fmt):
    """The block's exponent clamps to -127, and exp2(-127) is a subnormal a flushing graph turns into 0: dividing by it
    gave 0/0 = NaN for mxfp8 (1024 NaNs on this tensor). Under the flush the round trip has no NaN, the block is
    exactly 0, and every value equals the unflushed round trip's."""
    w = _with_zero_blocks()
    unflushed = quantization._block_round_trip(w, fmt, 1)
    with _flushed_denormals():
        flushed = quantization._block_round_trip(w, fmt, 1)
    assert not flushed.isnan().any()
    assert torch.equal(flushed[1, :32, :], torch.zeros(32, 32, dtype=flushed.dtype))
    assert torch.equal(flushed, unflushed)


@pytest.mark.parametrize("fmt", ["mxfp8", "mxfp4", "nvfp4"])
def test_an_all_zero_tensor_round_trips_to_zero_under_flushed_denormals(fmt):
    """Unlike a tensor with one zero block, an all-zero tensor flushes nvfp4's global scale (2^-137) itself to 0, so
    unguarded ``amax / 6 / global_scale`` gave 0/0 = NaN across the whole tensor."""
    x = torch.zeros(4, 64, dtype=torch.bfloat16)
    unflushed = quantization._block_round_trip(x, fmt, -1)
    with _flushed_denormals():
        flushed = quantization._block_round_trip(x, fmt, -1)
    assert torch.isfinite(flushed).all()
    assert flushed.abs().max().item() == 0.0
    assert torch.equal(flushed, unflushed)


@pytest.mark.parametrize(
    "x",
    [
        pytest.param(torch.full((4, 64), 1e-36), id="1e-36"),
        pytest.param(torch.full((4, 64), 3e-38), id="3e-38"),
        pytest.param(torch.cat([torch.full((2, 64), 2e-35), torch.zeros(2, 64)]), id="2e-35_with_zero_blocks"),
    ],
)
def test_tiny_nvfp4_tensors_round_trip_to_finite_zero_under_flushed_denormals(x):
    """Any tensor amax below ``2688 * 2^-126`` (~3.16e-35) flushes ``amax / 2688`` to 0 under FTZ, making
    ``global_scale = 0``; every block must still stay finite with no NaN or inf."""
    with _flushed_denormals():
        q = quantize_nvfp4(x, axis=-1)
        recon = dequantize(q)
    assert torch.isfinite(recon).all()
    assert recon.abs().max().item() == 0.0
    assert torch.isfinite(q.scales.float()).all()
    assert torch.isfinite(q.global_scale).all()


def test_nvfp4_unflushed_all_zero_keeps_subnormal_global_scale():
    """Unflushed arithmetic keeps ``nvfp4_global_scale(0) == 2^-137``, preserving Triton parity and the exporter's
    ``global_scale.reciprocal() == inf`` detection for all-zero weights."""
    q = quantize_nvfp4(torch.zeros(4, 64), axis=-1)
    assert q.global_scale.item() == 2.0**-137
    assert (q.data == 0).all()
    assert (q.scales.float() == 0.0).all()


@pytest.mark.parametrize("fmt", ["mxfp8", "mxfp4", "nvfp4"])
@pytest.mark.parametrize(
    ("shape", "axis"),
    [
        ((0, 64), -1),
        ((4, 0), -1),
        ((0, 0), -1),
        ((0, 64), 0),
        ((0, 3), 0),
        ((0, 17), 0),
        ((2, 0, 32), -1),
        ((0, 2, 32), -1),
        ((2, 0, 16), 1),
    ],
)
def test_empty_tensor_round_trip_and_ste_backward(fmt, shape, axis):
    """Empty activations (such as an EP rank with 0 local tokens) must round-trip and pass the STE backward cleanly."""
    block_size = quantization.FORMAT_BLOCK_SIZE[fmt]
    norm_axis = axis % len(shape)
    t = torch.empty(shape, dtype=torch.bfloat16)

    q = quantization.QUANTIZERS[fmt](t, axis=axis)
    expected_data_shape = list(shape)
    if q.packed:
        expected_data_shape[norm_axis] //= 2
    expected_scale_shape = list(shape)
    expected_scale_shape[norm_axis] //= block_size
    assert q.data.shape == tuple(expected_data_shape)
    assert q.scales.shape == tuple(expected_scale_shape)
    if fmt == "nvfp4":
        assert q.global_scale is not None and q.global_scale.shape == () and q.global_scale.dtype == torch.float32
        assert torch.isfinite(q.global_scale).all()
    else:
        assert q.global_scale is None

    recon = dequantize(q)
    assert recon.shape == shape and recon.dtype == torch.bfloat16

    x = torch.empty(shape, dtype=torch.bfloat16, requires_grad=True)
    y = fake_quant(x, fmt, axis=axis)
    assert y.shape == shape and y.dtype == torch.bfloat16
    y.sum().backward()
    assert x.grad is not None and x.grad.shape == shape and x.grad.dtype == torch.bfloat16


@pytest.mark.parametrize("shape", [(2, 0, 16), (0, 2, 16)])
def test_empty_nvfp4_16_wide_3d_round_trip(shape):
    x = torch.empty(shape, dtype=torch.bfloat16, requires_grad=True)
    y = fake_quant(x, "nvfp4", axis=-1)
    assert y.shape == shape and y.dtype == torch.bfloat16
    y.sum().backward()
    assert x.grad is not None and x.grad.shape == shape and x.grad.dtype == torch.bfloat16


@pytest.mark.parametrize("fmt", ["mxfp8", "mxfp4", "nvfp4"])
@pytest.mark.parametrize("shape", [(0, 3), (0, 17)])
def test_empty_tensor_with_non_divisible_blocked_axis_still_raises(fmt, shape):
    with pytest.raises(ValueError, match="not divisible"):
        quantization.QUANTIZERS[fmt](torch.empty(shape), axis=-1)


def _block_at_exponent(fmt: str, edge: str) -> torch.Tensor:
    """Row 1's first block on an extreme ``e8m0`` exponent, its values normal fp32 so the flush leaves the input alone:
    ``tiny`` clamps to -127 (amax below format max x 2^-127, about 2.6e-36 for mxfp8 and 3.5e-38 for mxfp4), ``inf``
    reaches 127 with values large enough to keep codes once scaled by 2^-127."""
    torch.manual_seed(0)
    x = torch.randn(4, 64)
    if edge == "tiny":
        x[1, :32] = (torch.rand(32) * 0.5 + 0.5) * 3e-38 * torch.where(torch.rand(32) < 0.5, -1.0, 1.0)
    else:
        x[1, :32] = torch.exp2(torch.rand(32) * 19 + 108) * torch.where(torch.rand(32) < 0.5, -1.0, 1.0)
        x[1, 0] = float("inf")
    return x


@pytest.mark.parametrize("edge", ["tiny", "inf"])
@pytest.mark.parametrize("fmt", ["mxfp8", "mxfp4"])
def test_an_extreme_exponent_block_keeps_its_codes_and_scale_under_flushed_denormals(fmt, edge):
    """Exponents -127 and 127 are where 2^-127 enters the scaling: CUDA's exp2(-127) is a subnormal, one ulp low, and a
    flushing graph makes it 0. The quantizer scales those blocks by normal powers of two, so its codes and scale match
    the unflushed ones. Before #185, mxfp4 divided a tiny block by the flushed 0 (all codes 0) and mxfp8 multiplied an
    inf block by it (NaN codes). Only the dequantize's 2^-127 multiply is left to the flush."""
    x = _block_at_exponent(fmt, edge)
    unflushed = quantization.QUANTIZERS[fmt](x, axis=-1)
    with _flushed_denormals():
        flushed = quantization.QUANTIZERS[fmt](x, axis=-1)
    assert torch.equal(flushed.data.view(torch.uint8), unflushed.data.view(torch.uint8))
    assert torch.equal(flushed.scales, unflushed.scales)
    assert unflushed.scales[1, 0].item() == (0 if edge == "tiny" else 254)
    # Past the inf's own element (byte), the block must keep nonzero codes, or it tests nothing.
    assert unflushed.data[1, 1:16].float().abs().max() > 0


@pytest.fixture
def fresh_compile_state(monkeypatch):
    monkeypatch.setattr(quantization, "_compiled_round_trip", None)
    monkeypatch.setattr(quantization, "_compile_failed", False)


def test_a_callers_error_leaves_compile_on(fresh_compile_state):
    """It must not switch every later quantization in the process to the eager path."""
    with pytest.raises(ValueError, match="not divisible"):
        quantization._round_trip(torch.randn(4, 33), "mxfp8", -1)
    assert quantization._compile_failed is False


def test_a_compile_failure_falls_back_to_eager(fresh_compile_state, monkeypatch):
    def fail_to_compile(*args):
        raise TorchDynamoException("simulated backend failure")

    monkeypatch.setattr(quantization, "_compiled_round_trip", fail_to_compile)
    x = torch.randn(4, 64)
    out = quantization._round_trip(x, "mxfp8", -1)
    assert quantization._compile_failed is True
    assert torch.equal(out, quantization._block_round_trip(x, "mxfp8", -1))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
