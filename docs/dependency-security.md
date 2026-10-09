# Dependency security and trusted model files

The release requires `torch>=2.10.0,<3`. The minimum addresses both official
PyTorch advisories relevant to its `torch.load(..., weights_only=True)` path:

| Advisory | Affected versions | First fixed version |
| --- | --- | --- |
| [CVE-2025-32434 / GHSA-53q9-r3pm-6pq6](https://github.com/pytorch/pytorch/security/advisories/GHSA-53q9-r3pm-6pq6) | <=2.5.1 | 2.6.0 |
| [CVE-2026-24747 / GHSA-63cw-57p8-fm3p](https://github.com/pytorch/pytorch/security/advisories/GHSA-63cw-57p8-fm3p) | <=2.9.1 | 2.10.0 |

The later advisory describes malformed checkpoint data that can corrupt memory
and potentially execute code despite `weights_only=True`. Consequently, 2.6.0
alone is insufficient for this release's dependency floor. The upstream advisory
list was checked on 2026-10-09; this is a bounded review of the supported loading
path, not a comprehensive audit of PyTorch or its transitive dependencies.

Load only locally generated or otherwise trusted model directories. The loader
reads JSON metadata, disables NumPy object unpickling, loads tensor weights onto
CPU with `weights_only=True`, and applies strict state-dictionary matching. It
checks model format, preprocessing, labels, feature-array schemas and scales. These checks do not
authenticate the producer, comprehensively validate all artifact contents or
bound resource usage. A patched dependency is not permission to load arbitrary
untrusted checkpoints. Do not bypass package dependency checks with `--no-deps`
unless the installed dependencies already meet the declared requirements.

The tested environment and reproduction instructions are in `installation.md`.
Functional parity on the earlier PyTorch 2.5.1 environment does not establish
release security. No malicious checkpoint was executed during verification.
