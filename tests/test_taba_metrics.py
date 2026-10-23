"""Native RGB metric layout and shape contracts."""

import pytest
import torch

from scripts.experiments.run_direct_mdrr_cica_scene import _taba_rgb_metrics


def test_noncontiguous_native_rgb_pair_uses_project_metrics(monkeypatch):
    import metrics

    # LPIPS network inference is covered by the checkpoint evaluations.
    monkeypatch.setattr(metrics, "lpips", lambda a, b, net_type: (a - b).square().mean())
    pred = torch.full((3, 32, 40), 0.5).permute(1, 2, 0)
    gt = torch.full((3, 32, 40), 0.4).permute(1, 2, 0)
    result = _taba_rgb_metrics(pred, gt)
    assert result["PSNR"] == pytest.approx(20.0, abs=1e-5)
    assert result["MSE"] == pytest.approx(0.01, abs=1e-7)
    assert 0.9 < result["SSIM"] < 1.0


def test_mismatched_rgb_pair_is_rejected():
    with pytest.raises(ValueError, match="shape mismatch"):
        _taba_rgb_metrics(torch.ones(32, 40, 3), torch.ones(31, 40, 3))
