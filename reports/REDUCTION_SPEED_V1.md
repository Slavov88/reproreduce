# Reduction Speed v1

## Objective

Begin evidence-driven reduction-speed optimization without weakening failure preservation, deterministic output, or standalone repro quality.

## Method and environment

- Baseline profiling commit: `52af4a953b4d316d0115f05aff8bca1d206c61c0`
- Optimized commit: `54bd6a24a24b652f836e25e60315b68842fa50f4`
- Host: Windows 10, Python 3.10.11
- PyTorch: `2.5.1+cu121`, CUDA available
- Command: `python -m reproreduce.bench --benchmark ... --repeats 3 --timeout 10`
- Baseline and optimized runs used separate fresh workspaces and SQLite caches.
- The expensive Inductor reduction was not rerun after optimization because a cold-start attempt already exceeded 40 minutes. Its prior completed warm-cache record remains the compiler baseline.
- Final full suite after optimization: 105 tests passed in 1,240 seconds at `54bd6a2`.

The profiler adds lightweight timings to `RunResult` and `ReductionSession` for source writing, process startup, process wait, oracle classification, SQLite lookup/write, candidate calls, and residual reducer bookkeeping. It does not change candidate acceptance.

## Baseline profile

The three-repeat baseline medians were:

| Benchmark | Wall time median | Requests | Unique executions | Cache hits | Reduced LOC |
|---|---:|---:|---:|---:|---:|
| Large exception | 173.19 s | 186 | 167 | 19 | 4 |
| Nested Python | 231.93 s | 390 | 212 | 178 | 55 |

Representative baseline time attribution:

| Component | Large exception | Nested Python |
|---|---:|---:|
| Candidate call | 168.79 s | 231.07 s |
| Subprocess wait | 167.06 s | 224.40 s |
| Process startup | 1.26 s | 6.08 s |
| Source writing | 0.17 s | 0.21 s |
| SQLite cache writes | 0.96 s | 1.16 s |
| SQLite cache lookups | 0.06 s | 0.15 s |
| Oracle classification | 0.02 s | 0.03 s |
| Residual reducer bookkeeping | 1.05 s | 1.59 s |

The dominant cost is candidate execution in isolated subprocesses, not oracle parsing or SQLite access. Representative candidate calls account for approximately 98.8% of wall time in both benchmarks; the exact share varies across repeats.

## Inductor profile

The completed v1 Inductor reduction recorded 574.64 seconds of candidate execution inside 577.84 seconds of wall time: approximately 99.4% candidate execution. A separate fresh original-fixture probe under the same environment took 19.939 seconds:

- source write: 0.001 s
- process startup: 0.011 s
- process wait: 19.927 s

This is not a reduction run, but it confirms that compiler/import work is inside the isolated candidate wait and that process startup is small for this case. No persistent worker or isolation-changing execution model was introduced.

## Cache and duplicate analysis

| Benchmark | Requested candidates | Unique executions | Cache hits | Distinct duplicated sources | Duplicate request ratio |
|---|---:|---:|---:|---:|---:|
| Large exception | 186 | 167 | 19 | 13 | 10.2% |
| Nested Python | 390 | 212 | 178 | 88 | 45.6% |
| Large Inductor | 445 | 78 fresh in warm-cache retry | 367 | not recorded in v1 | 82.5% |

The nested case confirms substantial duplicate requests, but the measured SQLite lookup cost is only 0.15 seconds. The first optimization therefore targets lookup overhead, not candidate generation.

## Optimization implemented

**LOW-RISK:** in-memory memoization of `source -> (RunResult, OracleResult)` inside one `ReductionSession`.

- Persistent SQLite checkpointing remains unchanged.
- The first request for a source still consults SQLite and/or executes the candidate.
- Repeated requests in the same reduction session return the memoized result before SQLite deserialization.
- No benchmark names or failure-specific logic were added to production code.

## Before / after

| Benchmark | Baseline | Optimized | Speedup | LOC Before/After | Fingerprint |
|---|---:|---:|---:|---:|---|
| Large exception | 173.19 s median | 179.41 s median | 0.965x | 4 / 4 | Preserved |
| Nested Python | 231.93 s median | 219.55 s median | 1.056x | 55 / 55 | Preserved |

The large-exception median regressed by 3.6%; the nested median improved by 5.3%. The repeat ranges overlap enough that this is not evidence of a broad material speedup. The nested result is the largest observed gain, but it should be treated as **OBSERVED**, not as a stable universal improvement.

## Candidate and cache effects

The optimization did **not** reduce requested candidates or unique executions:

- Large exception: 186 requests / 167 unique before and after
- Nested Python: 390 requests / 212 unique before and after

It changed the cache path:

- Large exception: 19 duplicate requests served from memory
- Nested Python: 178 duplicate requests served from memory
- Nested SQLite lookup time: 0.149 s → 0.065 s
- Large-exception SQLite lookup time: 0.059 s → 0.050 s

This is a small absolute saving because subprocess execution dominates. A future optimization must target duplicate candidate generation or candidate execution itself to produce a material gain.

## Reduction quality

**COMPUTATIONALLY VERIFIED:** across three baseline and three optimized repeats for both benchmarks:

- reduced nonblank LOC was unchanged;
- reduced-source SHA-256 was unchanged;
- candidate request and unique-execution counts were unchanged;
- configured fingerprints were preserved;
- standalone exported reproducers succeeded.

No evidence of a quality regression was observed.

## README decision

The README now prominently reports the verified large-input reductions and honestly states that large reductions currently take minutes. No optimization speed number was added to the README because the first pass produced a mixed result rather than a robust general speedup.

## Limitations and next bottleneck

1. Isolated candidate execution, including PyTorch compiler work, dominates runtime.
2. In-memory memoization cannot materially improve wall time while fresh candidate execution dominates.
3. The optimization did not reduce duplicate candidate requests; ddmin/reducer scheduling remains the next generic search-efficiency target.
4. A persistent worker was not attempted because it would weaken subprocess isolation and could leak module, allocator, or compiler state.
5. No optimized full Inductor reduction was run; the prior warm-cache baseline and direct probe are retained instead.

## Evidence status

- **COMPUTATIONALLY VERIFIED:** timing instrumentation, before/after quality checks, three-repeat Python benchmark comparisons, and full tests.
- **OBSERVED:** mixed speed results and the ranked bottlenecks above.
- **NOT TESTED:** optimized full Inductor reduction, persistent workers, and a generic ddmin scheduling change.
