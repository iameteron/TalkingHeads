# App Profiles And Feature Flags

TalkingHeads supports two runtime app profiles from the same codebase:

```text
TALKINGHEADS_APP_PROFILE=dev
TALKINGHEADS_APP_PROFILE=demo
```

The profile is read by the backend and sent to the frontend through
`GET /api/session_config`. The frontend uses the returned feature flags to hide
controls, while the backend also enforces the same restrictions when session
config updates or WebSocket agent ticks arrive.

## Dev Profile

`dev` is the default profile. It exposes the full local/debug interface:

- API key entry in Settings;
- active-agent model and mode selection;
- expert/oracle model settings;
- ARC observation-format selection;
- ARC prompt override editing;
- agent prompt/debug modal;
- setup wizard;
- companion bench/research tools;
- ARC leaderboard and human-operator workflow.

Use this profile for local research, prompt debugging, model comparison, and
environment integration work.

## Demo Profile

`demo` is intended for public or guided demos where visitors should choose a
world, interact with the agent, and use the leaderboard, but should not edit
backend-sensitive or research/debug controls.

In demo mode these controls are hidden and ignored by the backend:

- API key entry;
- expert/oracle model settings;
- ARC observation-format selection;
- ARC prompt override editing;
- companion bench/research tools.

These features remain enabled:

- setup wizard and world/game selection;
- active-agent model selection, with Claude Sonnet 4.5 as the default;
- read-only agent prompt/debug modal;
- human operator chat;
- ARC object hints;
- ARC leaderboard submission and viewing.

## Demo Defaults

Server-side environment variables provide the demo defaults:

```text
OPENROUTER_API_KEY=...
TALKINGHEADS_APP_PROFILE=demo
TALKINGHEADS_DEMO_AGENT_MODEL=anthropic/claude-sonnet-4.5
TALKINGHEADS_DEMO_ARC_OBS_FORMAT=arc_image
```

`OPENROUTER_API_KEY` must be configured on the server. Demo users do not enter
keys through the UI.

`TALKINGHEADS_DEMO_AGENT_MODEL` is optional. If it is empty, TalkingHeads uses
`anthropic/claude-sonnet-4.5` as the demo default model.

`TALKINGHEADS_DEMO_ARC_OBS_FORMAT` applies to ARC sessions. It defaults to
`arc_image`; valid values are the same ARC MegaPrompt configs documented in
[ARC-AGI-3 integration](ARC_AGI_3.md).

## Implementation Notes

The feature flag source of truth is:

```text
play_web/server/features.py
```

The main API and WebSocket enforcement points are:

```text
play_web/server/services.py
play_web/server/app.py
```

The frontend consumes the `features` object from `/api/session_config` and
conditionally renders controls in:

```text
play_web/client/index.html
play_web/client/js/index.js
```

When adding a new debug-only feature, add a backend flag first, enforce it at
the API/WebSocket boundary, and then hide the matching frontend control.
