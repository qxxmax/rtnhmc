"""Checks on the DW reverse path and round-trip chain, for Table 8.

Same identity as the LJ13 side: with P_f the forward path measure and P_r the
reverse, dP_r/dP_f = exp(-W)/Z, so logmeanexp(W) over reverse paths started
from exact target samples equals -log Z. Here the check is run on an isotropic
Gaussian target, where log Z is closed form and exact samples are free.
"""
import math

import pytest
import torch

from dw_chains import forward_rollout_from, mode_l1, reverse_rollout
from neural_nhmc_dw import ContinuousNHMCKernel, ManyWellTarget, rollout


class GaussianTarget:
    """Unnormalized standard normal, so log Z is known and sampling is exact."""

    def __init__(self, dim=4, shift=0.0):
        self.dim = dim
        self.n_wells = dim // 2
        # A shift on the even coordinates breaks their sign symmetry. Without
        # it the flip layer is a symmetry of the target and of the base alike,
        # so mis-scoring it is very nearly harmless and cannot be detected.
        self.mean = torch.zeros(dim, dtype=torch.float64)
        self.mean[0::2] = shift

    def log_prob(self, x):
        return -0.5 * (x - self.mean).square().sum(1)

    def force(self, x):
        return -(x - self.mean)

    def score(self, x):
        return -(x - self.mean)

    def sample(self, n, generator=None):
        return self.mean + torch.randn(n, self.dim, generator=generator,
                                       dtype=torch.float64)

    def true_log_z(self):
        return 0.5 * self.dim * math.log(2.0 * math.pi)


NUM_STEPS = 8


def build(dim=4, num_steps=NUM_STEPS, seed=0, shift=0.0):
    torch.manual_seed(seed)
    target = GaussianTarget(dim, shift)
    kernel = ContinuousNHMCKernel(target, n_leapfrog=1, hidden=16,
                                  num_steps=num_steps,
                                  sigma_mode='analytic').double()
    return target, kernel


def build_with_sign_flips(dim=4, num_steps=NUM_STEPS, seed=0):
    """The DW runs wrap the kernel in MixedSignKernel, so the reverse path has
    to undo that layer too.

    Stage s scores its flip mask under sigmoid(logit[s]) going forward and
    sigmoid(logit[s+1]) going backward, so at the zero initialisation every
    probability is 1/2 and the two scores cancel identically. A single spike
    makes two consecutive stages disagree sharply while the rest still cancel.

    This is the end-to-end check that the layer does not break the identity;
    the convention itself is pinned down structurally further down, since a
    shifted target sharp enough to expose a mis-scoring statistically also
    inflates the work variance past the point where log Z can resolve it.
    """
    from neural_nhmc_dw import MixedSignKernel

    target, base = build(dim, num_steps, seed)
    kernel = MixedSignKernel(base, num_steps).double()
    with torch.no_grad():
        kernel.flip_logits.zero_()
        kernel.flip_logits[num_steps // 2] = 1.6
    return target, kernel


def logmeanexp_stderr(work):
    """Error bar for logmeanexp, which is not std(W)/sqrt(N).

    The estimator averages exp(W), so its error is set by the spread of exp(W)
    relative to its own mean. For work with a standard deviation around two
    the two formulas differ by an order of magnitude, and the cheaper one
    makes these identity checks look far sharper than they really are.
    """
    shifted = torch.exp(work - work.max())
    return float(shifted.std() / (shifted.mean() * math.sqrt(work.numel())))


def reverse_log_z(target, kernel, n_total=20000, chunk=2000, seed=100):
    works = []
    with torch.no_grad():
        for index in range(n_total // chunk):
            generator = torch.Generator().manual_seed(seed + index)
            x = target.sample(chunk, generator)
            _, work = reverse_rollout(kernel, target, x, NUM_STEPS)
            works.append(work)
    work = torch.cat(works)
    estimate = torch.logsumexp(work, 0) - math.log(work.numel())
    return -estimate.item(), work


def test_reverse_work_recovers_log_z():
    torch.manual_seed(1)
    target, kernel = build()
    got, work = reverse_log_z(target, kernel)
    stderr = logmeanexp_stderr(work)
    assert abs(got - target.true_log_z()) < max(6.0 * stderr, 1e-3), (
        got, target.true_log_z(), stderr)


def test_forward_and_reverse_agree():
    torch.manual_seed(2)
    target, kernel = build()
    reverse, work_r = reverse_log_z(target, kernel)
    with torch.no_grad():
        _, work_f = rollout(kernel, target, 20000, NUM_STEPS,
                            torch.device('cpu'), dtype=torch.float64)
    forward = (torch.logsumexp(-work_f, 0) - math.log(work_f.numel())).item()
    tolerance = 6.0 * math.hypot(logmeanexp_stderr(work_r),
                                 logmeanexp_stderr(-work_f))
    assert abs(reverse - forward) < max(tolerance, 1e-3), (reverse, forward)


def test_a_sampling_and_scoring_mismatch_is_caught():
    """The identity is invariant to which laws are used -- any normalized
    choice keeps the work unbiased -- so what it has to catch is a momentum
    drawn from one density and scored under another."""
    torch.manual_seed(3)
    target, kernel = build()
    good, work = reverse_log_z(target, kernel, n_total=8000)
    stderr = logmeanexp_stderr(work)

    original = kernel._log_sigma
    kernel._log_sigma = (
        lambda x, t, step, reverse: original(x, t, step, reverse)
        + (math.log(4.0) if reverse else 0.0))
    try:
        bad, _ = reverse_log_z(target, kernel, n_total=8000)
    finally:
        kernel._log_sigma = original
    assert abs(bad - good) > 10.0 * stderr, (good, bad, stderr)


def test_reverse_work_recovers_log_z_through_the_sign_flip_layer():
    """The production DW kernels carry a sign-flip wrapper; the first run of
    Table 8 died on it. The identity has to survive with the layer in place."""
    torch.manual_seed(11)
    target, kernel = build_with_sign_flips()
    got, work = reverse_log_z(target, kernel)
    stderr = logmeanexp_stderr(work)
    assert abs(got - target.true_log_z()) < max(6.0 * stderr, 1e-3), (
        got, target.true_log_z(), stderr)


def test_the_flip_layer_uses_the_two_probabilities_in_the_right_roles():
    """Checked structurally rather than through log Z.

    The statistical route cannot settle this. A sign flip of the even
    coordinates is a symmetry of the double well and of the Gaussian base, so
    mis-scoring it is nearly self-correcting; shifting the target to break the
    symmetry does make the error appear, but it inflates the work variance at
    the same rate, and at every shift the mistake stays inside the error bar
    of logmeanexp. So the convention is pinned down directly: stage s scores
    its mask under sigmoid(logit[s]) forward and sigmoid(logit[s+1]) backward,
    draws it from the backward one, and returns the position with its even
    coordinates un-flipped.
    """
    from dw_chains import _sign_flip_layer
    from neural_nhmc_dw import MixedSignKernel

    num_steps = 4
    _, base = build(4, num_steps, seed=0)
    kernel = MixedSignKernel(base, num_steps).double()
    with torch.no_grad():
        # Every stage a different probability, so no two can be confused.
        kernel.flip_logits.copy_(
            torch.tensor([-2.0, -0.5, 0.7, 1.3, 2.2], dtype=torch.float64))

    stage = 2
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0]], dtype=torch.float64).repeat(4096, 1)
    torch.manual_seed(0)
    x_prev, log_rho_f, log_rho_b = _sign_flip_layer(kernel, stage, num_steps, x)

    # The mask is recoverable from which even coordinates changed sign.
    flipped = (x_prev[:, 0::2] * x[:, 0::2] < 0).double()
    # Odd coordinates are never touched.
    assert torch.equal(x_prev[:, 1::2], x[:, 1::2])
    assert torch.equal(x_prev[:, 0::2].abs(), x[:, 0::2].abs())

    p_f = torch.sigmoid(kernel.flip_logits[stage])
    p_b = torch.sigmoid(kernel.flip_logits[stage + 1])

    def bernoulli_log_prob(mask, p):
        return (mask * torch.log(p) + (1 - mask) * torch.log(1 - p)).sum(1)

    assert torch.allclose(log_rho_f, bernoulli_log_prob(flipped, p_f), atol=1e-6)
    assert torch.allclose(log_rho_b, bernoulli_log_prob(flipped, p_b), atol=1e-6)
    # Swapping the two roles would be a different number, so the check bites.
    assert not torch.allclose(log_rho_f, log_rho_b, atol=1e-3)
    # And the mask is drawn from the backward probability, not the forward one.
    rate = flipped.mean().item()
    assert abs(rate - p_b.item()) < 0.02, (rate, p_b.item(), p_f.item())
    assert abs(rate - p_f.item()) > 0.05


def test_the_reverse_flip_matches_the_forward_step_convention():
    """The reverse layer must read the same two logits that the production
    forward step writes, so the two are compared on the same stage."""
    from dw_chains import _sign_flip_layer
    from neural_nhmc_dw import MixedSignKernel

    num_steps = 4
    _, base = build(4, num_steps, seed=0)
    kernel = MixedSignKernel(base, num_steps).double()
    with torch.no_grad():
        kernel.flip_logits.copy_(
            torch.tensor([-2.0, -0.5, 0.7, 1.3, 2.2], dtype=torch.float64))

    for stage in range(1, num_steps + 1):
        x = torch.ones(2048, 4, dtype=torch.float64)
        torch.manual_seed(stage)
        x_prev, log_rho_f, log_rho_b = _sign_flip_layer(kernel, stage,
                                                        num_steps, x)
        mask = (x_prev[:, 0::2] < 0).double()
        # forward_step reads logit[t] forward and logit[min(t+1, num_steps)]
        # backward. The last stage's clamp is the easy one to drop, so the
        # loop runs all the way to num_steps.
        p_f = torch.sigmoid(kernel.flip_logits[stage])
        p_b = torch.sigmoid(kernel.flip_logits[min(stage + 1, num_steps)])

        def bernoulli_log_prob(m, p):
            return (m * torch.log(p) + (1 - m) * torch.log(1 - p)).sum(1)

        assert torch.allclose(log_rho_f, bernoulli_log_prob(mask, p_f),
                              atol=1e-6), stage
        assert torch.allclose(log_rho_b, bernoulli_log_prob(mask, p_b),
                              atol=1e-6), stage


def test_forward_rollout_from_reproduces_the_path_measure():
    torch.manual_seed(4)
    target, kernel = build()
    works = []
    with torch.no_grad():
        for index in range(10):
            generator = torch.Generator().manual_seed(300 + index)
            u = torch.randn(2000, target.dim, generator=generator,
                            dtype=torch.float64)
            _, work = forward_rollout_from(kernel, target, u, NUM_STEPS)
            works.append(work)
    work = torch.cat(works)
    got = (torch.logsumexp(-work, 0) - math.log(work.numel())).item()
    stderr = logmeanexp_stderr(work)
    assert abs(got - target.true_log_z()) < max(6.0 * stderr, 1e-3), got


def test_mode_l1_is_zero_on_an_exactly_distributed_sample():
    """Drawing endpoints from the exact mode masses must give L1 near zero;
    a wrong sign convention on the mode index would not."""
    from eval_block_dw_nhmc import (exact_positive_probability,
                                    mode_probabilities)
    torch.manual_seed(5)
    n_wells, n = 4, 400000
    probabilities = mode_probabilities(n_wells, exact_positive_probability())
    modes = torch.multinomial(probabilities, n, replacement=True)
    signs = ((modes[:, None] >> torch.arange(n_wells)) & 1).double() * 2 - 1
    endpoints = torch.zeros(n, 2 * n_wells, dtype=torch.float64)
    endpoints[:, 0::2] = signs
    l1, observed = mode_l1(endpoints.reshape(1, n, 2 * n_wells).numpy(), n_wells)
    assert observed == 2 ** n_wells
    assert l1 < 0.01, l1


def test_pool_imh_reproduces_the_weighted_mode_masses():
    """The figure's middle panel claims the chain carries the corrected mode
    masses. That is only true if the chain over the pool converges to the
    weighted empirical measure, so it is checked directly against it."""
    import numpy as np

    from make_dw_imh_figure import mode_masses, path_imh_from_pool

    rng = np.random.default_rng(0)
    n = 20000
    pool = rng.normal(size=(n, 4))
    # A weight that strongly prefers one sign pattern, so raw and weighted
    # masses differ a lot and the test can tell them apart.
    log_weights = 2.0 * pool[:, 0] + 1.5 * pool[:, 2]

    weights = np.exp(log_weights - log_weights.max())
    weights /= weights.sum()
    chain, acceptance = path_imh_from_pool(log_weights, 400000, 40000, seed=1)

    weighted = mode_masses(pool, weights)
    sampled = mode_masses(pool[chain])
    raw = mode_masses(pool)
    for key in weighted:
        assert abs(sampled[key] - weighted[key]) < 0.02, (key, sampled, weighted)
    # And the correction is a real one, not a no-op.
    assert max(abs(raw[k] - weighted[k]) for k in weighted) > 0.2
    assert 0.0 < acceptance < 1.0


def test_mode_l1_sees_a_missing_mode():
    endpoints = torch.ones(1, 1000, 8, dtype=torch.float64)
    l1, observed = mode_l1(endpoints.numpy(), 4)
    assert observed == 1
    # All mass on the all-positive mode, whose exact mass is p+^4; the L1 is
    # then 2(1 - p+^4), about 0.99 for this target.
    assert l1 > 0.9
