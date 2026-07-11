# Craftax Oracle Notes

Craftax is the default TalkingHeads world mode. It supports the original
oracle/expert workflow: the agent can ask an operator, and the operator can be
implemented by routed expert models.

## Intent Model Training

The oracle routes player questions to either the map expert (location/block
queries) or the mechanics expert (achievements, crafting, how-to). Routing is
done by a small intent classifier.

If an intent model is available under `intent_model/oracle_intent_model`, the
oracle can use it through `config/oracle_config.yaml`. To retrain:

```bash
conda activate oracle_craftext
python train_intent.py
```

Useful options:

- `--base-model MODEL`: SetFit base model.
- `--save-path DIR`: where to save the trained model.
- `--data FILE`: JSON file with `texts` and `labels`.
- `--num-iterations N`: training iterations.
- `--batch-size N`: batch size.
- `--no-check`: skip the post-training sanity check.

## Manual Server Startup

The preferred local command is:

```bash
cd play_web
./scripts/play-serve.sh start
```

If you need the older two-terminal workflow:

```bash
conda activate oracle_craftext
cd play_web
uvicorn server:app --host 127.0.0.1 --port 8001
```

Then, in another terminal:

```bash
cd play_web
python -m http.server 8081
```

Open:

```text
http://127.0.0.1:8081/client/index.html
```

