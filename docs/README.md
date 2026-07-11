# TalkingHeads Documentation

This directory is the main place for project-level documentation that is not
tied to a single source file.

## Guides

- [ARC-AGI-3 integration](ARC_AGI_3.md): supported games, observation formats,
  UI behavior, human-helper mode, leaderboard, API endpoints, and tests.
- [App profiles and feature flags](APP_PROFILES.md): `dev` versus `demo`
  behavior, demo environment variables, and where feature gates are enforced.
- [Craftax oracle notes](CRAFTAX_ORACLE.md): legacy oracle intent routing,
  intent model training, and manual server startup.
- [MegaPrompt ARC stack](../MegaPrompt/arc_agi_prompt/README.md): ARC prompt
  config names and observation-family notes.

## Documentation Conventions

- Keep user-facing workflows in Markdown under `docs/`.
- Keep prompt-family implementation notes near the prompt family directory
  under `MegaPrompt/`.
- Document new API endpoints in the feature guide that introduces them.
- Do not include API keys, private run logs, or local secrets in docs.
