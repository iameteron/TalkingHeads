# exo-planet prompt section

This section tracks the prompt stack configured for exo-planet runs.

Active templates:
- `../database_formulation/craftext.yaml`
- `../reasoning_or_ask_path/craftext.yaml`
- `../reasoning_or_ask_help/craftext.yaml`
- `../no_dialog/craftext.yaml`

All listed templates are expected to follow the strict exo-planet output contract:
- one output type per tick (`<action>` or `<question>` via `ASK_OPERATOR`);
- optional `<to_database>` only with `<action>UPDATE_DATABASE</action>`;
- no legacy terms (`craftax`, `minecraft`, `crafting table`).
