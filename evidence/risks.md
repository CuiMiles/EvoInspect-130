# Risk register

| ID | Risk | Severity | Current control | Status |
|---|---|---:|---|---|
| R-001 | `.git` is a read-only placeholder, so commit provenance is unavailable | High | Record `UNAVAILABLE` and config/data/model hashes; restore a real repository before benchmark runs | Open |
| R-002 | GPU jobs may conflict with other users | Critical | Recheck compute PIDs before every launch; use only physical GPU 1 when idle; explicit `CUDA_VISIBLE_DEVICES=1`; never terminate or preempt unrelated PIDs | Controlled for this run |
| R-003 | Public benchmark evaluation coverage is incomplete | High | Direct MVTec AD archive with all 15 categories is available read-only; current trained result remains bottle-only until full-category protocols run | Open |
| R-004 | Official label, latency and output protocols are ambiguous | High | Maintain strict compatible paths and send the consolidated organizer questions | Open |
| R-005 | Environment drift can invalidate reproduction | Medium | Dedicated environment `/home/CuiMinghao/envs/evoinspect-130`; exact research dependency pins in `environment.yml`; record model/data hashes | Controlled |
| R-006 | `prompts/P1_MASTER_RESEARCH_BUILD.md` pre-session hash differs from `CHECKSUMS.sha256` | Medium | Preserve both values; do not rewrite the prompt or checksum without owner review | Open |
| R-007 | Required dataset downloads use official forms and some licenses prohibit commercial use | High | Record official licenses; require human signoff and use only external read-only raw-data storage | Open |
| R-008 | Direct archive source identity may be misattributed | High | Record direct archive SHA-256; path/link-check all members; cross-check bottle 292/292 against pinned mirror; retain human source review gate | Controlled, review pending |
