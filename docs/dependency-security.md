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
checks model format, preprocessing, labels and anchor shape. These checks do not
authenticate the producer, comprehensively validate all artifact contents or
bound resource usage. A patched dependency is not permission to load arbitrary
untrusted checkpoints. Do not bypass package dependency checks with `--no-deps`
unless the installed dependencies already meet the declared requirements.

The tested environment and reproduction instructions are in `installation.md`.
Functional parity on the earlier PyTorch 2.5.1 environment does not establish
release security. No malicious checkpoint was executed during verification.

## Retained legacy dependencies

The historical main-branch requirements pin Requests 2.32.3, which falls within
the affected ranges of two independently verified moderate advisories:
[CVE-2024-47081](https://github.com/psf/requests/security/advisories/GHSA-9hjg-9r4m-mvj7)
(fixed in 2.32.4; potential netrc credential disclosure with crafted URLs) and
[CVE-2026-25645](https://github.com/psf/requests/security/advisories/GHSA-gc5v-m9x4-r6x2)
(fixed in 2.33.0; insecure temporary-file reuse when directly calling
`extract_zipped_paths()`). The old download script uses `requests.get()`; no
direct call to the latter utility was found in tracked legacy Python files.

Requests is absent from the current package's declared dependencies and the
verified release environment's runtime dependency closure. This correction does
not update legacy scripts or remediate separately installed legacy environments.
GitHub reported two moderate alerts on the default branch, but the available
connector did not expose their details. The advisories above must not be treated
as confirmed identities of those repository alerts.
