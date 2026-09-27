# ARENA 0.1.0 — VLA Server–Client–Adapter Framework

ARENA implements the architecture from the technical report: a GPU-hosted
Vision–Language–Action (VLA) policy server connected over HTTP to a client
that drives either an Isaac Lab / Arena simulation or a Unitree real robot.
An **Embodiment Adapter** bridges the robot-native observation/action space
and the canonical VLA space.

```
                ┌──────────────────────────┐
                │  VLA Policy Server (GPU) │
                │  /act  ← JSON over HTTP  │
                ────────────▲─────────────┘
                             │  action chunk
        observation          │
   ┌─────────────────────────┴──────────────────────────┐
   │                        Client                       │
   │   Embodiment Adapter  ⇄  Robot (Isaac Lab / Unitree)│
   └─────────────────────────────────────────────────────┘
```

## Components

| Module | Responsibility | Report §  |
|---|---|---|
| `arena.types` | Canonical `Observation` / `ActionChunk` | 2.1, 2.2 |
| `arena.config` | Config dataclasses + normalization | 4.4 |
| `arena.adapter` | Observation/Action Adapter | 4.1–4.3 |
| `arena.backends` | Pluggable policy backends | 2.3, 2.4 |
| `arena.server` | FastAPI `/act` server | 2.3 |
| `arena.client` | HTTP client + sim/real control loops | 3.1–3.3 |
| `arena.sim2real` | Domain-randomization curriculum (L1–L4) | 5.1, 5.2 |
| `arena.cli` | `arena server|sim|real` entrypoints | — |

## Installation

```bash
# Minimal (adapter, client, sim-to-real, mock backend)
pip install -e .

# With the FastAPI policy server
pip install -e ".[server]"

# With Isaac Lab / Arena environments
pip install -e ".[sim]"
```

## Quick start

Run the full pipeline without a GPU (mock backend + in-memory robot):

```bash
python demo.py
python demo.py --episodes 2 --sim2real-level 3
```

Launch a real policy server backed by a UnifoLM-VLA checkpoint:

```bash
arena server --backend unifolm_vla \
  --ckpt_path unifolm-vla/models/UnifoLM-VLA-Base1/checkpoints/<ckpt>.pt \
  --port 8777
```

Run the closed loop in simulation:

```bash
arena sim --backend mock --instruction "pick up the cube"
```

Run against a Unitree robot (requires the `unitree_deploy` client):

```bash
arena real --server_url http://127.0.0.1:8777/act --instruction "pick up the cup"
```

## Python API

```python
import numpy as np
from arena.adapter import EmbodimentAdapter
from arena.backends import build_backend
from arena.config import AdapterConfig, ServerConfig

adapter = EmbodimentAdapter(AdapterConfig(), instruction="pick up the cube")
backend = build_backend(ServerConfig(backend="mock"))

observation = adapter.encode_observation({
    "images": {"head": head_rgb, "wrist": wrist_rgb},
    "state": joint_positions,
})
actions = backend.infer([observation.to_dict()], "pick up the cube")
for action in adapter.decode_chunk(actions):
    robot.execute(action)
```

## Canonical data contract

* **Observation** — `{I_head, I_wrist, S_t, L}` (report § 2.1)
* **Action chunk** — `A_{t:t+H} = [a_t, …, a_{t+H}]` (report § 2.2)
* The server `/act` endpoint accepts
  `{"observations": [Obs, …], "instruction": str}` and returns
  `{"actions": [[…], …], "latency_s": float}`.
* `/health` performs a **real backend probe** and answers `200` when the backend
  can serve, `503` otherwise — it is not a bare liveness ping.

Numpy arrays are transported with `json_numpy`; a `{"encoded": "<json>"}`
payload is also accepted for clients without `json_numpy`. The server patches
`json` itself on startup, so no client-side setup is required.

## Backends

| Name | Description | `health()` |
|---|---|---|
| `mock` | Deterministic backend for tests/dry runs (no GPU) | always ready |
| `unifolm_vla` | Local UnifoLM-VLA checkpoint (GPU) | checkpoint loaded? |
| `openpi` | OpenPI `serve_policy.py` over HTTP | probes upstream |
| `http` | Transparent forwarding proxy | probes upstream (a `5xx` verdict counts as down) |

## Sim-to-real curriculum

`arena.sim2real` implements the randomized distribution
`E ~ P(L, T, C, M, μ, N)` with four levels of increasing
lighting / texture / camera / dynamics randomization, observation noise,
and action latency.

Coverage is explicit rather than implied:

| Factor | Applied by | Notes |
|---|---|---|
| `L` lighting, `T` texture | `randomize_image` | brightness / contrast multipliers |
| `C` camera | `randomize_camera` | crop + translate + sensor noise, in image space |
| `μ` observation noise | `randomize_state` | Gaussian, in normalized state space |
| `N` latency | `delay_action` | fixed control-step delay |
| `M` dynamics | `randomize_dynamics` | **returns** mass/friction/gain/damping scales; a simulator must apply them |

`DR-L1` is a strict identity transform (all magnitudes zero), so it reproduces
the plain simulation baseline. See `DOCUMENTATION.md` §8.3 for why dynamics is
parameter-only and what an Isaac Lab application looks like.

## Tests

```bash
python -m pytest tests/ -q
```

## License

BSD-3-Clause.