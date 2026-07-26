# SICA Corrupted-Image Policy

**Scope:** analysis of the current corrupt/missing-image handling in `AIGCLabelDataset` (`ForensicHub/tasks/aigc/datasets/label_dataset.py`, modified on this branch). **No source is modified by this document.** A companion CPU-only audit script is at `scripts/audit_corrupted_images.py`.

**Context:** OpenMMSecV2 contains corrupt/truncated LAION images and (in some environments) unreachable paths. The manifest `train.json` has 81,632 entries; some fraction cannot be decoded. How these are handled affects correctness, label balance, DDP consistency, and reproducibility.

---

## 1. Current handling (as implemented)

`AIGCLabelDataset` (the class used by all SICA configs) handles bad images in two layers:

1. **Init-time pre-filter** (`_init_dataset_path`, `prefilter=True` by default):
   - Reads the manifest, applies the `/mnt/public/` -> `/mnt/nas/public/` leading-prefix remap.
   - Two parallel `ThreadPoolExecutor(max_workers=32)` passes: (a) `os.path.isfile` (drop missing), (b) `_can_load` = `Image.open(path).convert("RGB").load()` (drop corrupt/truncated - a bare `Image.open` would NOT catch truncation; the `.load()` forces a full decode).
   - Drops missing + corrupt samples from `self.samples`; prints `n_raw -> n_valid (dropped X missing, Y corrupt)`.
   - Optionally **caches** the filtered sample list to `cache_dir/aigclabel_<sha1>.json` (atomic write via `os.replace(tmp, final)`). Cache key = hash of (manifest path, size, mtime, prefix-from, prefix-to, n_raw) - any change invalidates it.
2. **`__getitem__` safety net** (`max_load_retries=8`):
   - Tries the requested sample. On failure: tries up to 8 **random** replacement indices (skipping known-bad ones), then falls back to a **clone of the last known-good output** (`_good_output`).
   - **Never recurses on `__getitem__`** - the loop is bounded. Rate-limited warnings (first 10 only).
3. A `warnings.filterwarnings` silences the "Palette images with Transparency" PIL warning (those images decode fine; the warning would flood logs).

The **old** `AIGCCrossDataset` (not used by SICA configs) silently recursed to the next index on failure - the source of the historical infinite-loop risk.

---

## 2. Recursion risks

- **Old `AIGCCrossDataset`**: on a load failure it called itself at `index+1`. If many consecutive paths were bad (e.g., the `/mnt/public` prefix problem when unresolved), it would recurse across the whole manifest and loop - a DataLoader hang. **This is why SICA uses `AIGCLabelDataset` instead.**
- **New `AIGCLabelDataset`**: **no recursion.** The init-time prefilter removes bad samples before any `__getitem__`; the `__getitem__` safety net is a bounded `for _ in range(max_load_retries)` loop with a guaranteed fallback (`_good_output` clone). It cannot loop forever. **Verdict: recursion risk eliminated.**
- Residual: the safety net's random replacement could repeatedly hit bad indices, but it is bounded (8 tries) and always terminates via the fallback. No hang.

## 3. Dummy-image risks

The `__getitem__` safety net substitutes **another sample** when the requested one fails. This introduces two risks:

- **Label mismatch (the real risk):** a replacement returns a *different* image with its *own* label. If a real image (label 0) fails and is replaced by a fake image (label 1), the batch slot now carries a mislabeled example -> silent label noise. With `prefilter=True` (default) this only triggers on *residual* failures (transient NFS errors, files that degraded after the scan) - rare, so the noise is negligible. With `prefilter=False` it would trigger on every corrupt image -> systematic label noise. **Verdict: acceptable with prefilter ON; dangerous with prefilter OFF.**
- **Sample duplication:** the `_good_output` clone fallback duplicates the last known-good sample (image+label together, so self-consistent - no label mismatch, but a repeated example). Only reached if every live read fails (catastrophic). Negligible in practice.

## 4. Label-distribution distortion

- **Drop bias:** the prefilter *removes* corrupt/missing samples entirely. If corruption is correlated with label or domain (e.g., more corrupt images among fakes, or in one domain), the effective training distribution shifts. Example: if the Doc domain has more corrupt images, the already-small Doc slice (12,962 in train) shrinks further, worsening the IMDL-heavy/AIGC-light imbalance. The dropped counts per domain should be measured (the audit script does this).
- **Replacement bias (prefilter off):** random replacements preserve the distribution *in expectation* but add variance and the label-noise of §3.
- **Effective N:** the post-filter count `N'` (≤ 81,632) determines iterations/epoch and the LR-schedule denominator (`len(data_loader)`). A different `N'` ⇒ a different LR trajectory. `N'` must be recorded per run.
- **Test-set drop:** for evaluation, dropping corrupt test images changes the denominator and can shift a domain's metric. This must be reported transparently (§7).

## 5. Distributed (DDP) consistency

- **Deterministic filtered set:** the prefilter scans files in manifest order with deterministic checks (`isfile`, `convert("RGB").load()`); `ThreadPoolExecutor.map` preserves input order. So every rank produces the **same** `self.samples` list. The `DistributedSampler` then shards this identical list consistently across ranks. **Verdict: consistent across ranks** (with or without cache).
- **Cache race:** if no cache exists, all ranks scan concurrently and each writes the cache; `os.replace` is atomic and the content is identical, so last-writer-wins is harmless. Once cached, all ranks read the same file.
- **`__getitem__` non-determinism:** the safety net uses Python `random.randrange`, which is **not** seeded per DataLoader worker by default. Replacements are therefore non-reproducible across runs/workers. With prefilter ON, replacements are rare, so the effect on the final model is tiny - but it is a theoretical reproducibility leak. For strict bit-reproducibility, seed worker RNG or disable the safety net (accept a crash on residual failure instead).
- **Prefilter-off under DDP (avoid):** if prefilter were disabled, different ranks could hit different residual failures, yielding slightly different per-rank data and gradient inconsistency. Keep prefilter ON for DDP.

## 6. Reproducibility implications

- **Record `N'` and drop counts** (missing/corrupt, ideally per domain) with every run. Store them in the run summary (the `summarize_sica_run.py` parser captures the `[AIGCLabelDataset] ... dropped X missing, Y corrupt` line from text logs).
- **Store the cache file** (or the `bad_images.json` from the audit script) alongside the run. If the cache is lost and a re-scan finds a different set (e.g., a file was repaired, or an NFS hiccup excluded a transiently-unreadable file), `N'` changes and the run is not bit-reproducible.
- **Cache staleness:** the cache key includes manifest size/mtime, so a manifest edit invalidates it - good. But it does **not** include the image files' state; if an image file changes on disk without the manifest changing, the stale cache keeps the old verdict. Re-scan deliberately (delete the cache) after any dataset maintenance.
- **Deterministic audit:** the audit script produces a stable, sorted `bad_images.json` so the same images are flagged every time, making the dropped set reproducible and reviewable.

---

## 7. Recommended final policy

1. **Training:** keep `prefilter=True` (the default) for all SICA training. It removes bad samples deterministically, avoids per-sample warnings, and avoids the dummy-image label-noise problem. Record `N'` and per-domain drop counts in the run summary; store the prefilter cache (or `bad_images.json`) with the run.
2. **Evaluation:** run the prefilter once, **log the bad images per domain**, and report metrics on the valid subset **with the excluded count stated explicitly** (e.g., "Doc: 34,613 → 34,590 valid, 23 excluded"). Do not silently drop test images without recording the count. For the most defensible comparison to the paper, also report an "all images, corrupt assigned a neutral score" variant if the paper's protocol is unclear.
3. **Never use `AIGCCrossDataset`** (silent recursion) for SICA - always `AIGCLabelDataset`.
4. **DDP:** keep prefilter ON; the filtered set is identical across ranks. Optionally seed worker RNG (`worker_init_fn`) if bit-reproducibility is required, or accept the rare safety-net non-determinism as negligible.
5. **Reproducibility:** capture `bad_images.json` via `scripts/audit_corrupted_images.py` (small safe sample by default; full scan only on explicit request and only when training is **not** running, to avoid NAS I/O contention) and store it with the run snapshot.
6. **Cache hygiene:** after any dataset maintenance, delete the prefilter cache so the next run re-scans. Do not edit the cache by hand.
7. **Transparency:** every reported number should state the valid-sample count per domain and the total excluded, so a reader can see exactly what was evaluated.

### What still needs measuring (do not run a full scan while training)
- The exact missing/corrupt counts per manifest (the audit script's full-scan mode, run only when GPUs/NAS are idle). Until then, the policy is defined but the drop rates are unknown. The known data point: the sanity run scanned its subset without reporting large drop counts; the full `train.json` drop count is pending.
