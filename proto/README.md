# Deepslate Realtime Wire Protocol

[`realtime.proto`](realtime.proto) (package `eu.deepslate.realtime.speeq`) is the **single source of
truth** for the Deepslate realtime wire contract. Both the Python and Node/TypeScript SDKs generate
their bindings from this one file — it is never duplicated.

The file is kept **flat** at this path (rather than nested under a `deepslate/realtime/v1/...` tree) so
that both toolchains resolve it identically and the generated descriptors stay stable. It imports only
the Google well-known types `struct.proto` / `timestamp.proto`, which both toolchains resolve
automatically.

> **Contract DRY rule:** within each language, the `core` package owns all proto construction and
> parsing. Downstream SDKs (livekit, pipecat) import from `core` and never touch the generated bindings
> directly.

---

## Regenerating

When you change `realtime.proto`, regenerate **both** languages.

### Python (`grpc_tools` — no `protoc` binary required)

Run from the directory where the stubs must land so `--python_out=.` writes beside the existing files:

```bash
cd python/packages/core/src/deepslate/core/proto
../../../../../../.venv/bin/python -m grpc_tools.protoc \
  -I ../../../../../../../proto \
  -I ../../../../../../.venv/lib/python3.11/site-packages \
  --python_out=. \
  --pyi_out=. \
  realtime.proto
```

Generates `realtime_pb2.py` and `realtime_pb2.pyi` beside the source. Because `realtime.proto` sits flat
at the root of the `-I` include path, the descriptor stays `source: realtime.proto` and the emitted
module is `realtime_pb2`, imported unchanged as `from .proto import realtime_pb2`.

### Node / TypeScript (`buf` + protobuf-es)

Run from the `node/` workspace root:

```bash
pnpm run generate
# delegates to: pnpm --filter @deepslate/proto run generate
# → buf generate ../../../proto --template ../../buf.gen.yaml
#   writes node/packages/proto/src/gen/realtime_pb.ts
```

Then rebuild the generated package:

```bash
pnpm --filter @deepslate/proto run build
```

See [`node/buf.gen.yaml`](../node/buf.gen.yaml) and [`node/packages/proto`](../node/packages/proto/README.md)
for the codegen configuration.

---

## License

Apache License 2.0 — see [LICENSE](../LICENSE) for details.