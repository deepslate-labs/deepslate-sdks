# @deepslate/proto

Generated TypeScript protobuf bindings for the Deepslate realtime wire protocol.

> **Internal package.** You should not depend on this directly — it is consumed by
> [`@deepslate/core`](../core/README.md), which owns all proto construction and parsing. Downstream SDKs
> (e.g. [`@deepslate/livekit`](../livekit/README.md)) never touch these bindings.

The bindings are generated from the single shared wire contract at
[`proto/realtime.proto`](../../../proto/realtime.proto) (package `eu.deepslate.realtime.speeq`) — the
same file the Python SDK compiles against. Generation uses [`buf`](https://buf.build/) +
[`@bufbuild/protobuf`](https://github.com/bufbuild/protobuf-es) (protobuf-es v2).

---

## Regenerating

Run from the `node/` workspace root:

```bash
pnpm run generate
# equivalent to: pnpm --filter @deepslate/proto run generate
# → buf generate ../../../proto --template ../../buf.gen.yaml
#   writes src/gen/realtime_pb.ts
```

Then rebuild:

```bash
pnpm --filter @deepslate/proto run build
```

> When changing the wire protocol, edit [`proto/realtime.proto`](../../../proto/realtime.proto) and
> regenerate **both** languages. See [`proto/README.md`](../../../proto/README.md) for the combined
> Node (`buf`) and Python (`grpc_tools`) instructions.

---

## License

Apache License 2.0 — see [LICENSE](../../../LICENSE) for details.