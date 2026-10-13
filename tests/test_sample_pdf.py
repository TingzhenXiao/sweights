import numpy as np
import pytest
from numpy.testing import assert_allclose

from sweights.experimental import Cows


def test_Cows_sample_pdf_corrects_sample_density():
    n_samples = 50_000
    u = (np.arange(n_samples, dtype=float) + 0.5) / n_samples
    sample = np.sqrt(u)

    def signal_pdf(x):
        return 2.0 * np.asarray(x, dtype=float)

    def background_pdf(x):
        return np.ones_like(np.asarray(x, dtype=float))

    def mixture_norm(x):
        return np.asarray(x, dtype=float) + 0.5

    def sample_pdf(x):
        return 2.0 * np.asarray(x, dtype=float)

    cows = Cows(
        sample=sample,
        spdf=signal_pdf,
        bpdf=background_pdf,
        norm=mixture_norm,
        range=(0.0, 1.0),
        summation=True,
        sample_pdf=sample_pdf,
    )

    assert_allclose(cows._wm[0, 0], np.log(3.0), rtol=0, atol=1e-5)
