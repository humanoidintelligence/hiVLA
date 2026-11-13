# Contributing to hiVLA

hiVLA serves many VLA model families behind one interface. Most contributions
add a config for a checkpoint, add a model family, or report how a model ran.
This page says what each kind of change needs so it can be reviewed quickly.

## Set up

```bash
git clone https://github.com/humanoidintelligence/hiVLA.git && cd hiVLA
pip install -e '.[dev]'             # add '.[server,client]' to run hivla serve
python -m pytest                    # run from the repo root; no GPU or model weights needed
ruff check src tests --select F     # the lint rules the code is held to
```

Every model family is faked in the tests, so the full suite runs on any machine.
Heavy dependencies (`torch`, `onnxruntime`, `tensorrt`, `fastapi`, the model's
own package) are imported only when a backend loads or the server starts, so
`import hivla` and most tests work without them. A test that needs an extra
should skip without it (`pytest.importorskip`).

## What a change needs

| Change | Add | Don't touch |
|---|---|---|
| New checkpoint for a supported family | A config in `src/hivla/configs/<name>.yaml` (start from a family's existing one), checkpoint path from an environment variable, a line in the family's guide | Family code |
| New model family | `src/hivla/families/<name>.py` with its `FamilyAdapter` and backend, registered in `families/__init__.py`; tests that fake the family's package; a guide `docs/families/<name>.md` (and `.zh-CN.md`) modelled on the existing ones; a row in the README's model table | `VlaPolicy`, other families |
| New checkpoint format for `generic` | A `ModelBackend` in `src/hivla/runtimes/` | Families |
| Wire format, `/act` / `/metadata` reply, `hivla.client` | Open an issue first. Robot-side callers depend on these | |

The README's [Add a new model](README.md#add-a-new-model) has the details. In short:

- The server does all model-specific work (resizing, normalization,
  tokenization, unnormalization). The client sends a plain observation and
  never changes when the model does.
- Use hiRobot's field names on the wire (`joint_targets`, `gripper`,
  `ee_targets`, ...). Map the model's own names in the family's input and
  output transforms, or with `policy.state_layout` and `policy.gripper_range`.
- Import the model's package inside the backend's `load()`, not at module
  level, and raise an `ImportError` that names what to install.
- A model that keeps state between requests sets `stateful` on its backend.
- Don't put local paths, hostnames or keys into shipped configs. Read the
  checkpoint path from an environment variable, as the built-in configs do.

## Tests

- `tests/` encodes the contracts: wire format, transforms, engine cache and
  batching, sessions, both transports. A change to one of them comes with a
  test for the new behavior.
- `tests/test_client.py` drives a live server through `hivla.client` over HTTP
  and WebSocket, JSON and msgpack. Run it after touching `protocol.py`,
  `serving/server.py` or `client.py`.
- The tests cannot run a real model. Say in the pull request what you ran
  (checkpoint, GPU, benchmark or robot, and the result), and add it to the
  family guide's "Verified" section.

## Pull requests

- One topic per pull request. Keep refactors separate from behavior changes.
- The tests pass and `ruff check src tests --select F` is clean.
- Update the docs your change makes wrong: README tables, `docs/api.md`, the
  family guides, and their Chinese versions.
- Commit messages: a short summary line, then what changed and why.

## Reporting problems

Open a GitHub issue with the model family and checkpoint, the config, the
hiVLA commit, the versions of the model's package and `torch`, and the full
error. For a wrong action, say what the robot or simulator did.

## License

hiVLA is licensed under [Apache-2.0](LICENSE). By contributing you agree that
your contribution is licensed under the same terms.
