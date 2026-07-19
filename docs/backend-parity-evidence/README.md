# Backend parity evidence pack

Verification-only stage for commit pinned in nvironment.json.

This directory holds **reproducible artifacts only**. Status claims live in
STATUS.md and must map 1:1 to files here.

| Path | Contents |
| --- | --- |
| nvironment.json | commit, toolchain, GPU, drivers |
| logs/ | full test stdout/stderr, exit codes |
| 
umerical/ | CPU reference error tables |
| allback/ | static + runtime fallback audits |
| ench/ | raw CSV/JSON + SLO calculator output |
| portable/ | browser/WASM results or impossibility note |
| independent_review.md | review by non-author agent |
| STATUS.md | separate Implemented/Tested/… flags |
