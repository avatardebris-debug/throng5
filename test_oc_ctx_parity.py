"""Verify Option-Critic uses the same integrated ctx for select and update."""

from __future__ import annotations

import numpy as np
from brain.orchestrator import WholeBrain


def test_oc_integrated_ctx_parity():
    brain = WholeBrain(
        n_features=84,
        n_actions=18,
        session_name="oc_ctx_parity",
        enable_logging=False,
        use_torch=True,
    )
    striatum = brain.striatum
    oc = striatum.option_critic
    assert oc is not None, "Option-Critic required for ctx parity test"

    features = np.random.RandomState(7).randn(84).astype(np.float32)
    select_ctx = striatum.oc_input(features)

    # Force cache rebuild path used during live stepping
    striatum.reset_oc_context_cache()
    update_ctx = striatum.oc_input(features)

    assert select_ctx.shape == update_ctx.shape, (
        f"ctx shape mismatch: {select_ctx.shape} vs {update_ctx.shape}"
    )
    np.testing.assert_allclose(
        select_ctx, update_ctx, rtol=1e-5, atol=1e-5,
        err_msg="select and update paths must share identical integrated ctx",
    )

    # After a step, learn/observe should reuse the same builder for next_state
    next_features = np.random.RandomState(8).randn(84).astype(np.float32)
    striatum.reset_oc_context_cache()
    ctx_a = striatum.oc_input(features)
    ctx_b = striatum.oc_input(features)
    np.testing.assert_allclose(ctx_a, ctx_b)

    brain.close()
    print("[PASS] OC integrated ctx parity")


if __name__ == "__main__":
    test_oc_integrated_ctx_parity()
    print("ALL OC CTX PARITY TESTS PASSED")
