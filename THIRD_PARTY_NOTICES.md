# Third-Party Notices

TalkingHeads is released under the MIT License. Third-party packages and assets
retain their original licenses.

## Python Dependencies

The project depends on packages listed in `requirements.txt`, including JAX,
PyTorch, Transformers, Hugging Face Hub, OpenAI, FastAPI, Uvicorn, NumPy,
Pillow, PyYAML, Craftax, and ARC-AGI. Install-time dependencies are governed by
their respective upstream licenses.

Known direct dependencies checked during release preparation:

- `craftax`: MIT License.
- `arc-agi`: MIT License.

## ARC-AGI Game Files

ARC-AGI-related environment files are derived from the ARC-AGI ecosystem and
retain the upstream license notices supplied with those files.

## Visual Assets

Bundled visual assets under `play_web/external_visualization/` and
`play_web/client/assets/` are included for rendering the demo interface and
environment states. If you replace or redistribute these assets separately,
check their individual provenance and licenses.

## Runtime Data

Private runtime logs, local trajectories, API keys, and leaderboard submissions
are intentionally excluded from the public repository snapshot.
