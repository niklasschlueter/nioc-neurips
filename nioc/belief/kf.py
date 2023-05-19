from jax import lax, numpy as jnp

from nioc.control import LQGSpec


def forward(spec: LQGSpec, Sigma0: jnp.ndarray, xhat0=None, gains=None) -> jnp.ndarray:
    def loop(P, step):
        A, F, V, W = step

        G = F @ P @ F.T + W @ W.T
        K = A @ P @ F.T @ jnp.linalg.inv(G)
        P = V @ V.T + (A - K @ F) @ P @ A.T

        return P, K

    _, K = lax.scan(loop, Sigma0,
                    (spec.A, spec.F, spec.V, spec.W))

    return K
