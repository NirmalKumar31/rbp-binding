# Cluster & Operations Masterclass

**The companion to `RBP_BINDING_MASTERCLASS.md`. That one explains the *science* (biology, models,
results). This one explains *how we actually ran it* — the HPC cluster, every command we typed and
what it means, how to read every scrap of output, the full data-and-process flow, the validation
gate step by step, `rsync`, every error we ever hit (v1, v2, and this run) with the fix and the
why, and finally what changed across v1 → v2 → `rbp-binding`.**

> Written for someone who has never touched a supercomputer. Nothing is assumed. If a line of
> terminal output shows up in this project, it is decoded here character by character.

## Table of contents
- **Part 1 — What a cluster is** (login vs compute nodes, the scheduler, node names)
- **Part 2 — Slurm, and how to read every output** (sbatch, squeue, sacct, job states, job IDs)
- **Part 3 — The environment** (`setup_env.sh`, what `multimolecule` is, the CUDA `cu118` story)
- **Part 4 — The data & process flow, end to end** (every stage; the validate gate in full)
- **Part 5 — Manifests & job packing** (how array tasks map to real work, the QOS math)
- **Part 6 — Every command we typed, decoded**
- **Part 7 — `rsync` in depth** (flags, trailing slashes, why not scp/git, the retry loop)
- **Part 8 — Every error we hit** (v1/v2/now: symptom → cause → fix → why → prevent)
- **Part 9 — The intermediate checks after each run** (the syntax, and why we check)
- **Part 10 — Reading the results** (every table, every term)
- **Part 11 — Where everything lives** (the file map, cluster + Mac)
- **Part 12 — v1 → v2 → rbp-binding** (the core changes, technical + conceptual)
- **Appendix — glossary of every cluster/ops term**

---

# Part 1 — What a cluster is

## 1.1 The one-sentence mental model
A **cluster** is a few hundred computers (**nodes**) sharing one filesystem and one job queue. You
log into a small **login node**, but you never do heavy work there — you *describe* your work in a
script and hand it to a **scheduler**, which finds a free **compute node** and runs it for you. Our
cluster is **NEU Explorer**; its scheduler is **Slurm**.

## 1.2 Login node vs compute nodes (why this matters)
- **Login node** (your prompt said `explorer-01`, `explorer-02`): a shared front door. Fine for
  editing files, submitting jobs, small commands, `rsync`. **Not** for training models — if everyone
  ran heavy jobs here it would crawl, and admins kill such processes.
- **Compute nodes** (`c0281`, `d1017`, …): the actual workhorses. You reach them **only through the
  scheduler**, either as a batch job (`sbatch`) or an interactive session (`srun --pty`). When your
  prompt changes from `explorer-02` to something like `c0281`, you're *on a compute node*.

This is exactly why `setup_env.sh` and `download_data.sh` were run after an `srun` (they're heavier),
while `sbatch`/`squeue` were run from the login node (they just talk to the scheduler).

## 1.3 One shared filesystem
Every node sees the same `~/rbp-binding/`. That's why a job running on `d1017` can read the data a
different job wrote on `c0621`, and why you `cd ~/rbp-binding` on the login node and the compute
nodes see the identical folder. The filesystem is shared; the CPUs/GPUs are not.

## 1.4 Node names — how to read them (your question: "why do CPU nodes start with c and GPU with d?")
On Explorer the node name is a **location/generation code**, not a strict CPU/GPU label:
- `c0xxx`, `c2xxx`, `c3xxx` — general **CPU** compute nodes. Our `prep`, `cnn`, `validate`, `agg`
  jobs (the `short` partition) landed here (`c0281`, `c0619`, `c0685`, `c3015`, …).
- `d0xxx` — also served some CPU jobs in our run (`d0139`, `d0146`).
- `d1xxx` — the **V100 GPU** nodes. Our `lm` and `clinvar` jobs (the `gpu` partition) ran on `d1017`,
  `d1019`. (`d1013`/`d1022` are the 3-GPU variants.)
- `d4xxx` — the newest **H200 GPU** nodes.

So the letter roughly tracks hardware racks, but **don't trust the letter** — the *reliable* way to
know what a node is:
```bash
scontrol show node c0603   # shows CPUTot, RealMemory, and Gres=  (Gres=gpu:... means it has GPUs)
```
The scheduler assigns you *any* eligible node in the partition you requested; we asked for
`--partition=gpu`, and Slurm happened to give us `d1017`/`d1019` (both V100-SXM2 nodes).

## 1.5 Partitions and QOS (the rules of the road)
- **Partition** = a named pool of nodes with its own limits. We used two:
  - `short` — CPU nodes, generous limits (2-day max wall-time, up to 50 running jobs). Our prep/CNN
    arrays live here.
  - `gpu` — GPU nodes, 8-hour max wall-time. Our LM fine-tunes and ClinVar live here.
  - (Explorer also has `gpu-short`, `gpu-interactive`, `sharing`, etc. — we didn't need them.)
- **QOS (Quality of Service)** = per-user usage caps attached to a partition. The one that shaped our
  whole design: the **`gpu` QOS allows 8 submitted / 4 running jobs, ≤ 4 GPUs at once**. `short` is
  effectively uncapped (50 running). We saw this exactly:
  ```
  gpu        8   4   gres/gpu=4
  short          50  cpu=1024,mem=25T
  ```
  Everything about how the LM sweep is packed (Part 5) exists to fit "8 submitted / 4 running."

---

# Part 2 — Slurm, and how to read every output

## 2.1 The three ways to run something
- **`sbatch script.sbatch`** — submit a *batch* job: the scheduler queues it, runs it on a compute
  node when resources free up, and returns immediately (you get a job ID). This is 95% of what we did.
- **`srun … --pty /bin/bash`** — get an *interactive* shell on a compute node (blocks until granted,
  then you're "on" the node). We used this for `setup_env.sh` and `download_data.sh`.
- **`sbatch --wrap='…'`** — submit a one-liner as a batch job without writing a file. We used this for
  the hardware probe.

## 2.2 Anatomy of an `.sbatch` file — every `#SBATCH` directive
From `train_lm.sbatch`:
```bash
#!/bin/bash
#SBATCH --job-name=rbp-lm            # name shown in squeue
#SBATCH --partition=gpu              # which node pool
#SBATCH --array=0-7%4                # run as 8 tasks (indices 0..7), at most 4 at once
#SBATCH --gres=gpu:1                 # "generic resource": request 1 GPU per task
#SBATCH --cpus-per-task=8            # 8 CPU cores per task
#SBATCH --mem=24G                    # 24 GB RAM per task
#SBATCH --time=08:00:00              # kill the job if it runs past 8 h
#SBATCH --output=logs/lm_%A_%a.out   # where stdout+stderr go; %A=array job id, %a=task index
```
- `#!/bin/bash` — it's a bash script; the `#SBATCH` lines are special comments Slurm reads *before*
  running the body.
- `%j` (single job), `%A_%a` (array job + task) become numbers in the log filename — that's why the
  logs are `lm_8896867_0.out`, `prep_8896679_16.out`, etc.

## 2.3 Job arrays (the single most important idea)
`--array=0-19%8` means: submit **one** script but run it as **20 near-identical tasks**, each with a
different value of the environment variable `$SLURM_ARRAY_TASK_ID` (0, 1, …, 19), **at most 8 running
simultaneously** (the `%8` throttle). Inside the script we use that index to pick which protein to
work on (Part 5). This is how "one script" becomes "20 parallel jobs."

## 2.4 Reading `squeue` — decode a real line
You asked how I know things like *"cnn 8896866 — short, 0–4 R, rest pending, 20 tasks; agg — held on
Dependency."* Here's the actual output and how to read every column:
```
             JOBID PARTITION     NAME     USER ST       TIME  NODES NODELIST(REASON)
  8896866_[5-19%8]     short  rbp-cnn thirupal PD       0:00      1 (None)
         8896866_0     short  rbp-cnn thirupal  R       1:01      1 c0281
           8896868     short  rbp-agg thirupal PD       0:00      1 (Dependency)
```
Column by column:
- **JOBID** — `8896866_0` is *task 0* of array job `8896866`. `8896866_[5-19%8]` is a *summary line*
  for the still-queued tasks 5–19 (the `%8` cap is holding them). `8896868` (no `_`) is a plain
  single job.
- **PARTITION** — `short` or `gpu`. That alone tells you CPU vs GPU work.
- **NAME** — the `--job-name` (`rbp-cnn`, `rbp-agg`, `rbp-lm`, `rbp-prep`, `rbp-validate`).
- **USER** — you (`thirupal…`).
- **ST (state)** — the two-letter status. The ones we saw:
  - `PD` = **PenDing** (queued, not started),
  - `R` = **Running**,
  - `CG` = Completing, `CD` = Completed, `F` = Failed (we never saw F).
- **TIME** — how long it's been running (`1:01` = 1 min 1 s). `0:00` for pending.
- **NODES** — how many nodes (always 1 for us).
- **NODELIST(REASON)** — if running, the node name (`c0281`); if pending, the *reason in parentheses*:
  - `(None)` — just waiting its turn, no blocker.
  - `(Priority)` / `(Resources)` — waiting because others are ahead / no free node yet.
  - `(Dependency)` — **held until another job finishes** (our `afterok` chain). This is how I knew
    `agg` was "held on Dependency" — it literally said so.
  - `(JobArrayTaskLimit)` — held by the array's own `%N` throttle (e.g. tasks 5–19 waiting because 8
    are already running).

So "cnn `8896866` — short, 0–4 R, rest pending, 20 tasks" is a direct reading: partition column says
`short`, tasks `_0`.._4 show `R`, the `_[5-19%8]` summary line shows `PD (JobArrayTaskLimit)`, and
`0-19` = 20 tasks. Nothing guessed — it's all in the table.

## 2.5 Common `squeue` commands
```bash
squeue -u $USER      # your jobs
squeue --me          # same thing, newer shorthand
```

## 2.6 Reading `sacct` — the after-the-fact record
`squeue` only shows *live* jobs; once a job finishes it vanishes from `squeue`. To see what
*happened* (including finished jobs) you use `sacct` (Slurm accounting):
```bash
sacct -j 8896866,8896867,8896868 -X --format=JobID,JobName,State,ExitCode
```
- `-j <ids>` — restrict to these job IDs.
- `-X` — show only the *allocation* line per job (hide the internal `.batch`/`.extern` sub-steps),
  so an array shows one row per task instead of three.
- `--format=…` — which columns to print. Handy fields: `JobID, JobName, Partition, AllocCPUS,
  ReqMem, AllocTRES, Elapsed, State, ExitCode, NodeList, MaxRSS` (MaxRSS = peak memory, but it lives
  on the `.batch` sub-step, so drop `-X` to see it).
- **State** — `COMPLETED` (good), `FAILED`, `CANCELLED`, `TIMEOUT`.
- **ExitCode** — `0:0` means clean exit (exit code 0, signal 0). Anything else = trouble.

Why this mattered: after Phase B the queue was empty, which could mean "all done" *or* "a job failed
and its dependent was skipped." Running `sacct` and seeing **every task `COMPLETED 0:0`** is what
*proved* it was the good kind of empty.

## 2.7 `scancel` — stopping jobs
```bash
scancel 8898921             # cancel one job
scancel --name=rbp-clinvar  # cancel by job-name (all jobs with that name)
scancel -u $USER            # cancel ALL your jobs (the fresh-start hammer)
```

## 2.8 Why a frozen terminal never hurts a job
`sbatch` jobs run on compute nodes under the scheduler — **detached from your terminal**. If your SSH
freezes or drops, the job keeps running. (What "froze" earlier was an `srun --pty` command *waiting in
the queue* for a GPU — that blocks your terminal by design; Ctrl-C or a new SSH session fixes it. The
actual jobs were never affected.)

---

# Part 3 — The environment (`cluster/setup_env.sh`)

## 3.1 The script, line by line
```bash
module load python/3.13.5          # make a specific Python available (clusters use "modules")
python -m venv .venv               # create an isolated environment folder .venv/
source .venv/bin/activate          # switch into it (now "python"/"pip" mean the venv's)
pip install -U pip
pip install -r requirements.txt    # numpy, pandas, torch, scikit-learn, pyfaidx, matplotlib, ...
pip install "transformers==5.14.1" "tokenizers==0.22.2" "huggingface-hub==1.26.0" \
            "multimolecule==0.2.1" "peft==0.20.0"      # the RNA-LM stack, version-PINNED
pip uninstall -y torchao || true                       # remove a package that conflicts
pip install --force-reinstall torch --index-url https://download.pytorch.org/whl/cu118  # the fix
python -c "import torch; print('torch', torch.__version__, '| cuda_ok', torch.cuda.is_available())"
```

- **`module load`** — HPC clusters don't put every tool on your PATH by default; you "load a module"
  to get a specific version. This guarantees the *same* Python every run.
- **`venv`** — a private Python sandbox in `.venv/` so our exact package versions can't clash with the
  system or other users. `source .venv/bin/activate` enters it.
- **version pins (`==`)** — we lock exact versions so a future silent library update can't change how
  the models behave. Reproducibility.

## 3.2 What is `multimolecule`? (you wrote "multimodal" — it's `multimolecule`)
`multimolecule` is a **Python library of pretrained models for biological molecules** (RNA/DNA/
protein), built to plug into HuggingFace `transformers`. It's where our three language models come
from: `multimolecule/rnafm`, `multimolecule/rnabert`, `multimolecule/splicebert`. Think of it as "a
HuggingFace model hub, but for nucleic acids." The supporting cast:
- **`transformers`** — HuggingFace's core library (the Transformer/BERT machinery, loading weights).
- **`tokenizers`** — turns an RNA string into the integer tokens the model eats.
- **`huggingface-hub`** — downloads the pretrained weights from the internet the first time.
- **`peft`** — "Parameter-Efficient Fine-Tuning," the library that provides **LoRA** (used for RNA-FM).
- **`torch`** (PyTorch) — the deep-learning engine underneath all of it.

## 3.3 The CUDA `cu118` story (a real error we pre-empted)
GPUs are driven by **CUDA**. PyTorch ships in builds compiled against a specific CUDA version. If the
torch build is *newer* than the GPU node's driver, torch **can't initialize the GPU** and training
dies. In v1/v2 we hit exactly this — the driver reported code `12030` (= CUDA 12.3), and the default
torch wheel was built for something newer. The fix, kept in `setup_env.sh`:
```bash
pip install --force-reinstall torch --index-url https://download.pytorch.org/whl/cu118
```
This installs the **CUDA 11.8 ("cu118") build of torch**, which runs on any modern driver (11.8 ≤
12.3). The last line prints `cuda_ok True/False` as a self-check.

**Gotcha we saw:** when you run `setup_env.sh` on a **CPU** node (the `short` partition), the final
check prints `cuda_ok False` — *correctly*, because a CPU node has no GPU. Not a failure; it just
means "verify the GPU separately on a GPU node," which is why we ran a one-off GPU smoke test:
```bash
srun --partition=gpu --gres=gpu:1 --time=00:10:00 --pty bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# -> True  Tesla V100-SXM2-32GB
```

---

# Part 4 — The data & process flow, end to end

## 4.1 The pipeline as a picture
```
config/proteins.tsv  (the 16 proteins — single source of truth)
        │
        ▼
download_data.sh  ── ENCODE peaks + GRCh38 genome + GENCODE GTF + ClinVar  → data/
        │
        ▼
┌─ PHASE A (submit_data.sh) ───────────────────────────────────────────────┐
│  prep.sbatch  (short, array 0-19%8)                                       │
│     20 runs of data_prep.py  → data/processed/<P>/dataset.tsv + onehot.npz│
│     (16 primary + 4 splice_matched)                                       │
│        │  afterok                                                         │
│        ▼                                                                  │
│  validate.sbatch  (short) → eda.py (figures) then validate.py (the GATE)  │
│     >>> STOPS HERE. Human reads logs/validate_*.out. Must say PASSED.     │
└──────────────────────────────────────────────────────────────────────────┘
        │  (manual go-ahead)
        ▼
┌─ PHASE B (submit_models.sh) ─────────────────────────────────────────────┐
│  train_cnn.sbatch (short, 0-19%8)   ┐                                     │
│  train_lm.sbatch  (gpu,   0-7%4)    ┘ run in parallel                     │
│        │  afterok(cnn,lm)                                                 │
│        ▼                                                                  │
│  aggregate.sbatch (short) → aggregate.py + figures.py → results/*.tsv,*.png│
└──────────────────────────────────────────────────────────────────────────┘
        │  (after aggregate; separate to respect the gpu QOS)
        ▼
  clinvar.sbatch (gpu)  → clinvar.py + figures.py → results/clinvar_*.tsv
```

## 4.2 What each stage reads and writes
| Stage | Partition | Reads | Writes | ~time |
|---|---|---|---|---|
| `download_data.sh` | login/compute | config | `data/raw/`, `data/reference/` | minutes |
| `prep` ×20 | short | genome + GTF + peaks | `data/processed/<P>/dataset*.tsv`, `onehot*.npz` | 0.4–4 min each |
| `validate` | short | processed data + `positives_ref.tsv` | EDA figures, PASS/FAIL to log | ~20 s |
| `cnn` ×20 | short | `onehot*.npz` | `models/cnn/*.pt`, `results/metrics/*.cnn*.json` | 0.4–1.6 min |
| `lm` ×8 tasks (60 fine-tunes) | gpu | `dataset*.tsv` | `models/lm/*.pt`, `results/metrics/*.json`, `logs/util_*.csv` | 8–34 min/task |
| `aggregate` | short | `results/metrics/*.json` | `results/*.tsv`, `results/figures/*.png` | ~5 s |
| `clinvar` | gpu | ClinVar VCF + peaks + trained models | `results/clinvar_*.tsv` | ~5 min |

## 4.3 Why the phase gate (STOP at validate)
Training is the expensive part (GPU-hours). If the *data* is subtly wrong you'd waste all of it and,
worse, get a plausible-but-invalid result. So the pipeline **physically stops** after validation:
`submit_data.sh` submits only prep + validate; you read the log; only then do you run
`submit_models.sh`. "We validated" is *structural*, not a promise.

## 4.4 The validate gate, dissected (this was the new piece)
`validate.sbatch` runs two things on the **frozen** (just-built) data:
```bash
python src/eda.py        # regenerate all 11 EDA figures on the REAL data
python src/validate.py   # the integrity gate — exits non-zero on any hard failure
```
`validate.py`, step by step, for each of the 16 proteins:

1. **Files exist?** `dataset.tsv` + `onehot.npz` present (and the 4 ablation proteins'
   `*.splice_matched.*`). A missing file ⇒ a prep task silently failed. → hard fail.
2. **Class balance 1:1?** `#positives == #negatives`. Catches a broken negative sampler. → hard fail.
3. **Split obeys the rule?** Recompute `split_of(chrom)` for every row and check it equals the stored
   `split` column, **and** that no chromosome appears in more than one split. The anti-leakage
   assertion made executable. → hard fail.
4. **Enough eval data?** `val ≥ 100` and `test ≥ 100` pairs (below that the AUROC is noise). → hard fail.
5. **No cross-split leakage?** No identical `seq_rna` in two different splits (can happen via genome
   repeats). → hard fail.
6. **Reproduces v2?** Recompute a SHA-256 hash of every column *except* `split` and compare to
   `cluster/positives_ref.tsv` (the hash captured from the known-good v2 data). A **match** proves the
   fresh cluster re-extract is byte-identical to v2. → warn only (a benign difference shouldn't block
   you, but you should see it).
7. **Soft checks** (warn only): GC gap per protein; how many pairs the splice-matched sets retain.

Then it prints a per-protein table and either `=== VALIDATION PASSED ===` (exit 0) or `=== VALIDATION
FAILED ===` (exit 1) listing the hard failures. On our run every protein showed `vs_v2 = match`, all
cleared the floors, and it printed **PASSED** — the green light for Phase B.

**How the hash check works (the clever bit):** `content_hash()` reads the TSV with `dtype=str` (so
floating-point formatting can't drift between pandas versions), drops the `split` column (which
*legitimately* changed vs v2), and SHA-256s the rest. `make_reference.py` computed those hashes
*locally* from the old v2 data before we ever ran; `validate.py` recomputes them on the cluster and
compares. Identical bytes ⇒ identical hash ⇒ the extraction reproduced exactly.

**We tested the gate itself** before trusting it: locally we fed it correct data (it passed, all
`vs_v2=match`) then deliberately corrupted a split label + deleted an ablation file — it correctly
printed the three hard failures and `VALIDATION FAILED`. A gate you've never seen fail isn't a gate.

---

# Part 5 — Manifests & job packing (how array tasks map to real work)

## 5.1 The problem an array must solve
An array runs the *same* script 20 (or 8) times, differing only in `$SLURM_ARRAY_TASK_ID`. We need
task *k* to know "you handle protein X, model Y, negative-mode Z." We solve this with **manifests** —
plain text files, one job per line — generated from the config so there's a single source of truth.

## 5.2 `make_manifests.sh` output
```
20 cluster/manifest_prep.txt     # <protein>\t<negatives>          (16 primary + 4 splice_matched)
20 cluster/manifest_cnn.txt      # <protein>\t<negatives>
60 cluster/manifest_lm.txt       # <model>\t<protein>\t<negatives>  ((16+4) x 3 models)
```
Example lines:
```
manifest_prep.txt line 1:  TARDBP        primary
manifest_prep.txt line 17: PUM1          splice_matched
manifest_lm.txt   line 1:  rnafm  TARDBP primary
manifest_lm.txt   line 60: splicebert RBFOX2 splice_matched
```

## 5.3 How a prep/CNN task reads its line
`prep.sbatch` does:
```bash
read prot neg <<< "$(sed -n "$(( SLURM_ARRAY_TASK_ID + 1 ))p" cluster/manifest_prep.txt)"
python src/data_prep.py "$prot" --negatives "$neg"
```
- `SLURM_ARRAY_TASK_ID` is 0-based, file lines are 1-based → `+ 1`.
- `sed -n "Np"` prints just line N.
- `read prot neg <<< "…"` splits that line's two tab-separated fields into `$prot` and `$neg`.
- So task 0 → line 1 → `TARDBP primary`; task 16 → line 17 → `PUM1 splice_matched`. Clean 1-to-1.

## 5.4 The LM packing — fitting 60 fine-tunes into the 8/4 QOS
The gpu QOS allows only **8 submitted / 4 running** jobs, but we have **60** LM fine-tunes (48 primary
+ 12 ablation). A naive `--array=0-59` = 60 submitted jobs → rejected. So `train_lm.sbatch` uses **8
array tasks**, and each task **walks 8 lines** of the manifest:
```bash
PER=8
start=$(( SLURM_ARRAY_TASK_ID * PER + 1 ))   # task 0 -> lines 1..8, task 1 -> 9..16, ...
for i in $(seq 0 $(( PER - 1 ))); do
  entry=$(sed -n "$((start+i))p" cluster/manifest_lm.txt)
  [ -z "$entry" ] && continue                 # past line 60 -> empty -> skip
  read model prot neg <<< "$entry"
  python src/train_lm.py "$model" "$prot" --negatives "$neg"
done
```
8 tasks × 8 lines = 64 slots covering all 60 lines (last 4 slots empty, skipped). `--array=0-7%4` =
**8 submitted (= the cap), 4 running (= the cap)** — it fits *exactly*. This is the single most
important piece of "cluster engineering" in the project.

---

# Part 6 — Every command we typed, decoded

## 6.1 Navigating & inspecting
- `cd ~/rbp-binding` — change directory; `~` = your home. The prompt shows your current dir.
- `ls` / `ls -la` — list files / long format with hidden files, permissions, sizes, dates.
- `cat file` — print a whole file. `tail -f file` — print the end and *follow* it live (Ctrl-C stops).
- `grep -H "SBATCH" cluster/*.sbatch` — search for the text `SBATCH` in those files; `-H` prints the
  filename with each match. We used it to dump all `#SBATCH` directives at once.
- `wc -l file` — count lines (how we verified the manifests are 20/20/60).
- `du -sh models` — **d**isk **u**sage, **s**ummary, **h**uman-readable (→ `1.6G`).
- `column -t results/x.tsv` — pretty-print a tab file into aligned columns (readable tables).

## 6.2 Scheduler commands (covered above)
`sbatch`, `srun --pty`, `squeue --me`, `sacct -j … -X --format=…`, `scancel`, `sinfo`, `scontrol show
node`, `sacctmgr show qos`.

## 6.3 The submit scripts (what one command kicks off)
- `bash cluster/submit_data.sh` — regenerates manifests, then `sbatch prep` and `sbatch --dependency=
  afterok:<prep> validate`. Prints the two job IDs and "read the validate log."
- `bash cluster/submit_models.sh` — `sbatch cnn`, `sbatch lm`, then `sbatch --dependency=afterok:
  <cnn>:<lm> aggregate`. (ClinVar is *not* here — see Part 8's QOS bug.)
- `--parsable` (inside those scripts) makes `sbatch` print just the numeric job ID so we can capture it
  into a variable and chain the next job's `--dependency` on it.

## 6.4 The dependency chain (`afterok`)
`--dependency=afterok:8896867` means "start me only after job 8896867 finishes **successfully**." If
the dependency *fails*, the dependent is cancelled (never runs) — which is why an empty queue could
hide a failure, and why we always confirm with `sacct` (Part 2.6).

---

# Part 7 — `rsync` in depth (moving files between Mac and cluster)

## 7.1 What rsync is, and why not something else
`rsync` copies files between two machines over SSH, but **only the differences** — if a file already
exists and matches, it's skipped. That makes it **resumable and idempotent**: re-running is cheap and
safe. Why not the alternatives?
- **`scp`** — simpler but dumb: it re-copies everything every time and has no resume. Bad for 1.6 GB
  of models over a flaky link.
- **`git`** — versions *code*, not multi-GB data/models; you don't put a genome or 80 checkpoints in
  git. (We *do* use `.gitignore` to keep them out.)
- **A shared drive / Globus** — great for huge transfers, but rsync-over-SSH needs zero setup and we
  already had SSH.

## 7.2 The flags we used, decoded
```
rsync -avz --progress --partial --timeout=120 \
      -e "ssh -o ServerAliveInterval=15 -o ServerAliveCountMax=8" \
      --exclude 'X' src/ dst/
```
- **`-a`** (archive) — recurse into subfolders and preserve permissions/timestamps. The everyday
  default.
- **`-v`** (verbose) — list files as they transfer.
- **`-z`** (compress) — gzip in transit; faster over a network.
- **`--progress`** — show a live progress bar per file.
- **`--partial`** — **keep** a half-transferred file if the connection drops, so the next run
  *resumes* it instead of restarting. Crucial for our flaky link.
- **`--timeout=120`** — abort if the link stalls for 120 s (so the retry loop can kick in).
- **`-e "ssh -o ServerAliveInterval=15 -o ServerAliveCountMax=8"`** — run over SSH but send a
  keepalive "ping" every 15 s (up to 8 missed before giving up), so an idle moment doesn't get the
  connection killed.
- **`--exclude 'pattern'`** — skip matching paths (we excluded `.venv/`, `data/`, `models/` at times,
  and always the stale masterclass).

## 7.3 Trailing slashes (the classic footgun)
`rsync src/ dst/` copies the **contents** of `src` into `dst`. `rsync src dst/` copies the **folder**
`src` *into* `dst` (making `dst/src`). We used trailing slashes on both sides so
`~/rbp-binding/` contents land directly in the local `rbp-binding/`.

## 7.4 The three transfers we actually did
1. **Up (Mac → cluster), initial:** push the project up before running.
   ```bash
   rsync -avz --progress --exclude '.venv/' --exclude 'data/' \
     "…/rbp-binding/" thirupallikrishnan.n@login.explorer.northeastern.edu:~/rbp-binding/
   ```
   Excluded the local Mac `.venv` (the cluster builds its own) and `data/` (re-downloaded on the
   cluster; the local peak files were symlinks that'd break there).
2. **Up again (re-sync):** after we fixed `submit_models.sh`, a second identical rsync re-sent only the
   two changed files — proving rsync's "differences only" nature.
3. **Down (cluster → Mac):** pull results + models back, **excluding** `.venv/`, `data/` (14 GB,
   regenerable), and — critically — `docs/RBP_BINDING_MASTERCLASS.md`, because the cluster's copy was
   **stale** (uploaded before results existed); pulling the whole `docs/` would have overwritten the
   Mac's updated masterclass. This "protect the newer file" is a real gotcha.

## 7.5 The retry loop (for a dropping connection)
When the down-sync of models kept dropping (`unexpected end of file`, Part 8), we wrapped it:
```bash
until rsync -avz --partial --timeout=120 -e "ssh -o ServerAliveInterval=15 …" \
      "…:~/rbp-binding/models/" "…/models/"; do
    echo ">>> dropped — resuming in 5s"; sleep 5
done
```
- `until CMD; do …; done` — run `CMD`; if it exits non-zero (failure), run the loop body and try
  again; stop when `CMD` finally succeeds (exit 0).
- Combined with `--partial`, each retry *continues* the transfer. Scoping the source to `models/` meant
  only the missing ~1.2 GB moved. It printed `=== DONE ===` when rsync finally exited 0.

---

# Part 8 — Every error we hit (symptom → cause → fix → why → prevent)

## 8.1 CUDA driver mismatch (v1/v2) — the cu118 fix
- **Symptom:** torch imports fine but `torch.cuda.is_available()` is `False` on a GPU node, or CUDA
  init errors mentioning code `12030`.
- **Cause:** the default torch wheel is built for a newer CUDA than the node's driver (CUDA 12.3).
- **Fix:** install the **cu118** torch build (`--index-url …/whl/cu118`).
- **Why it works:** a cu118 build only needs a driver ≥ 11.8; 12.3 satisfies it.
- **Prevent:** baked into `setup_env.sh` permanently.

## 8.2 `QOSMaxSubmitJobPerUserLimit` (v2) — the array too big
- **Symptom:** `sbatch` refuses a GPU array: `QOSMaxSubmitJobPerUserLimit`.
- **Cause:** the gpu QOS caps you at **8 submitted** jobs, and **each array element counts** — a
  48-element array = 48 submitted → rejected.
- **Fix:** repack the 60 fine-tunes into **8 array tasks** that each loop over several manifest lines
  (Part 5.4), throttled `%4`.
- **Prevent:** the manifest+packing design; we now know 8/4 is the ceiling.

## 8.3 The `submit_models.sh` ClinVar bug (this run — caught *before* it bit)
- **Symptom (predicted):** the original `submit_models.sh` submitted cnn + lm + aggregate + **clinvar**.
  The lm array already uses all 8 gpu slots; clinvar would be a **9th** held gpu job → the same
  `QOSMaxSubmitJobPerUserLimit`.
- **Cause:** a held (`--dependency`) job still counts against the submit cap the moment it's submitted.
- **Fix:** removed clinvar from `submit_models.sh`; run it **separately** after aggregate frees the
  slots (`sbatch cluster/clinvar.sbatch`). Updated the doc + re-synced.
- **Why measure not patch-live:** we reasoned it out from the known 8/4 cap and fixed the script, then
  re-synced — rather than discovering it as a failed submission mid-run.

## 8.4 The frozen terminal (this run)
- **Symptom:** terminal unresponsive after `srun … --pty`; couldn't type.
- **Cause:** `srun --pty` **blocks** while it waits in the queue for a GPU to free up — it looks frozen
  but is just waiting.
- **Fix:** Ctrl-C (cancels the waiting `srun`); or open a fresh SSH session; the jobs are unaffected.
- **Prevent:** for one-off hardware probes we switched to `sbatch --wrap` (submit-and-return, writes to
  a file) instead of `srun --pty` (blocking).

## 8.5 Accidental duplicate ClinVar submit (this run)
- **Symptom:** you ran `sbatch cluster/clinvar.sbatch` twice.
- **Cause:** a re-submit.
- **Impact:** harmless — ClinVar is deterministic, so the second run recomputes the *identical* numbers
  and overwrites the same output files. (Both `8897201` and `8898104` appear in `sacct`, both
  `COMPLETED`.)
- **Prevent:** nothing needed; if you want to save GPU, `scancel --name=rbp-clinvar`.

## 8.6 rsync `unexpected end of file` / exit 255 (this run)
- **Symptom:** `rsync … error: unexpected end of file … child 90508 exited with status 255`.
- **Cause:** the **SSH connection dropped** mid-transfer (exit 255 is SSH's "connection lost"). Flaky
  network or a login-node transfer limit.
- **Fix:** the retry loop + `--partial` + SSH keepalives (Part 7.5).
- **Prevent:** for big/flaky transfers, always `--partial` + keepalives, or use a dedicated transfer
  node if the cluster has one.

## 8.7 The `UNEXPECTED` / `MISSING` model-loader warnings (this run) — **benign, but worth understanding**
When ClinVar loaded each pretrained encoder you saw big reports like:
```
[transformers] RnaFmModel LOAD REPORT
Key                              | Status
lm_head.transform.dense.weight   | UNEXPECTED
ss_head.decoder.weight           | UNEXPECTED
pooler.dense.weight              | MISSING
```
What this means: a saved model is a **`state_dict`** — a dictionary of `layer_name → weight tensor`.
When you load one set of weights into a model *object*, the loader compares key sets:
- **UNEXPECTED** = keys **in the checkpoint** but **not in our model**. The pretrained files ship the
  *pretraining heads* — `lm_head` (the masked-language-model head that predicts masked nucleotides) and
  `ss_head` (a secondary-structure head). We only want the **base encoder**, so we don't have those
  layers → they're "unexpected" and ignored.
- **MISSING** = keys **in our model** but **not in the checkpoint**, so they're **newly initialized**.
  Here `pooler.dense` — a pooling layer we don't even use (we do our own *masked mean pooling*), so its
  random init is irrelevant.
- **Both are expected and harmless in our setup.** We deliberately use only the encoder + our own
  classification head. The warning is `transformers` being transparent about key mismatches, not an
  error. (The nearby `padding='same' with even kernel` warning from the CNN is also cosmetic.)

## 8.8 zsh `no matches found` (this run, on the Mac)
- **Symptom:** `ls models/lm/*.splice_matched.pt` → `zsh: no matches found`.
- **Cause:** zsh (unlike bash) errors when a glob matches **nothing** — at that moment those files
  hadn't downloaded yet, so the pattern matched zero files.
- **Fix:** it correctly told us "0 splice checkpoints present" — i.e. the transfer was incomplete; we
  re-ran rsync. (To silence it generally: `setopt NULL_GLOB`, but here the error was informative.)

## 8.9 The `du --exclude` error on macOS (this run)
- **Symptom:** `du -sh --exclude=.venv .` → `du: illegal option -- -` / exit 64.
- **Cause:** macOS `du` (BSD) doesn't support GNU's `--exclude`. Purely a macOS-vs-Linux flag
  difference; every real check in that command had already printed fine.

---

# Part 9 — The intermediate checks after each run (and *why*)

The pattern all along was **run a stage → verify it → only then proceed**. The commands and reasons:

## 9.1 After `prep` (Phase A)
```bash
cat logs/prep_8896679_0.out     # a primary prep log
cat logs/prep_8896679_16.out    # the PUM1 splice_matched prep (watch "dropped=")
sacct -j 8896679 -X --format=JobID,State,ExitCode   # all 20 COMPLETED?
```
- **Why:** the prep logs print `pairs=… train/val/test=… GCband={…} dropped=…`. A healthy primary log
  had `dropped=0`; the splice one had `dropped=23` (tiny) — telling us splice-matching wasn't starving
  negatives *before* we relied on it.

## 9.2 The validate gate (Phase A end)
```bash
cat logs/validate_8896680.out   # must end: === VALIDATION PASSED ===
```
- **Why:** the single most important check — it's the gate. `vs_v2 = match` on all 16 confirmed the
  data reproduced v2 exactly; PASSED unlocked Phase B.

## 9.3 After training (Phase B)
```bash
sacct -j 8896866,8896867,8896868 -X --format=JobID,JobName,State,ExitCode   # all COMPLETED 0:0?
cat logs/agg_*.out                              # aggregate printed the tables?
column -t results/model_comparison.tsv          # the headline AUROCs
column -t results/ablation_splice.tsv           # the ablation
```
- **Why:** an empty queue can hide a failed job whose dependent got skipped. `sacct` all-`COMPLETED`
  is the *proof* the sweep truly finished, and confirms aggregate actually ran (not cancelled).

## 9.4 After ClinVar
```bash
cat logs/clinvar_*.out
column -t results/clinvar_summary.tsv
```

## 9.5 Final local audit (on the Mac, after pulling)
```bash
ls results/metrics/*.json | wc -l                 # expect 80
ls models/cnn/*.pt | wc -l; ls models/lm/*.pt | wc -l   # expect 20, 60
```
- **Why:** to prove the *download* was complete (this is how we caught that only 17/60 LM checkpoints
  had arrived, and later that all 80 were valid torch-zip files paired to their metric JSONs).

---

# Part 10 — Reading the results (every table, every term)

## 10.1 `results/model_comparison.tsv` — the headline
A **protein × model** grid of **test AUROC**, plus a `MEAN` row.
- **AUROC** (Area Under the ROC Curve): probability the model ranks a random *bound* window above a
  random *unbound* one. **0.5 = coin flip, 1.0 = perfect.** Bigger = better.
- **Read it:** each cell is one protein-model result; the `MEAN` row is the average over 16 proteins.
  Ours: **SpliceBERT 0.8926, RNA-FM 0.8784, CNN 0.8603, RNABERT 0.7871.**
- "Wins 13/16" = for 13 of the 16 proteins, SpliceBERT's cell is the highest in its row.

## 10.2 `results/ablation_splice.tsv` — the confound test
Columns `protein, model, splice_auroc, primary_auroc, drop` where **`drop = primary − splice`**.
- A big **positive** drop would mean "that score was riding on splice-site proximity."
- Ours: every `|drop| ≤ 0.046`, most `≤ 0.02`, many **negative** (splice-matched did *better*) → the
  models learned real binding, not a splice shortcut.

## 10.3 `results/clinvar_summary.tsv` — variant effect
Columns `stratum, n_path, n_benign, CNN, RNA-FM (LoRA), RNABERT, SpliceBERT`.
- **stratum** — which variants: `all @ sites`, `noncoding @ sites`, `coding @ sites` ("@ sites" =
  variants that fall in a real eCLIP peak).
- **n_path / n_benign** — how many Pathogenic / Benign variants in that stratum (the sample size —
  bigger = more trustworthy).
- The model columns are the **AUROC of the disruption score** (`max_shift |p(ref) − p(alt)|`) at
  telling Pathogenic from Benign.
- Ours: SpliceBERT **noncoding 0.837** (strong), coding 0.576 (modest) — a binding model has
  jurisdiction over noncoding-at-site variants, not coding ones. Honest, domain-scoped.

## 10.4 `results/gpu_perf.tsv` — cost
Per LM run: `seconds` (train time), `gpu_util_mean` (%), `gpu_mem_peak_gb`. Feeds the
`gpu_dashboard` and `acc_vs_compute` figures. Ours: RNA-FM ~398 s / 88 % / 3.3 GB vs SpliceBERT
~109 s / 58 % / 1.7 GB — the specialist is best **and** ~3.6× cheaper.

## 10.5 The four result figures
`model_comparison.png` (mean AUROC + per-protein winner), `gpu_dashboard.png` (time/util/mem per LM),
`acc_vs_compute.png` (AUROC vs training seconds — up-and-left = best), `splice_ablation.png` (the tiny
`drop` bars). See the figures walkthrough for how to read each.

---

# Part 11 — Where everything lives (the file map)

## 11.1 On the cluster (`~/rbp-binding/`)
```
config/proteins.tsv           the 16 proteins
src/*.py                      all the code
cluster/*.sbatch, *.sh        the job scripts + manifests + positives_ref.tsv
data/                         raw peaks, genome, GTF, ClinVar, + processed windows   (BIG, stays here)
models/cnn/<P>[.splice_matched].pt      20 CNN checkpoints
models/lm/<P>.<model>[.splice_matched].pt   60 LM checkpoints
results/                      metrics/*.json, *.tsv tables, figures/*.png
eda/                          frozen-data figures + summary
logs/                         every job's stdout + the util_*.csv GPU profiles
.venv/                        the cluster Python environment                         (BIG, stays here)
```

## 11.2 On the Mac (the mirror)
Same tree, **minus** `data/` and `.venv/` (huge + regenerable, left on the cluster), **plus** the
authoritative `docs/` (both masterclasses + the config exports). `models/` was pulled down (1.6 GB) so
you can archive/re-score locally.

## 11.3 Where the models live, and their names
- **CNN:** `models/cnn/TARDBP.pt` (primary), `models/cnn/PUM1.splice_matched.pt` (ablation). ~30 KB
  each (7k params).
- **LM:** `models/lm/TARDBP.splicebert.pt`, `models/lm/PUM1.rnafm.splice_matched.pt`, etc. The naming is
  `<protein>.<model>[.splice_matched].pt`. SpliceBERT/RNABERT are full weights (~80 MB / ~2 MB);
  **RNA-FM saves only the LoRA adapters + head** (small), not the frozen 100M base.
- A checkpoint is a **`state_dict`** — reload the same architecture, `load_state_dict()`, and you have
  the trained model back (that's exactly what `clinvar.py` does).

## 11.4 What's git-ignored (and why)
`.gitignore` excludes `data/`, `models/`, `results/`, `logs/`, `.venv/`, and the generated
`cluster/manifest_*.txt`. **Why:** git is for source, not multi-GB regenerables; the manifests rebuild
from config. **Kept in git:** code, config, `cluster/positives_ref.tsv` (the reproducibility
reference), docs, the EDA notebook.

---

# Part 12 — v1 → v2 → rbp-binding: the core changes

## 12.1 The one-line arc
- **v1** — "can a model predict RBP binding at all?" (a small, honest first pass).
- **v2** — "make the comparison *fair* and match the model to the biology" (SpliceBERT wins).
- **rbp-binding** — "do it *correctly*, *prove* it, and *interrogate* the result" (validation,
  reproducibility, comparability, and the ablation).

## 12.2 Side-by-side
| Dimension | v1 | v2 | **rbp-binding (now)** |
|---|---|---|---|
| Proteins | 5 | 16 | 16 |
| Split | single-chrom (test=chr1, val=chr2) | single-chrom (same) | **multi-chrom (test={chr1,chr2,chr20}, val={chr19,16,13,18})** |
| Models | CNN, RNA-FM(LoRA), RNABERT **frozen** | + **RNABERT fully fine-tuned**, + **SpliceBERT** | same 4, all tuned |
| Fair comparison? | no (RNABERT frozen) | **yes (all tuned to peak)** | yes |
| Best result | — | SpliceBERT 0.901 | SpliceBERT 0.893 (harder 3-chrom test) |
| ClinVar | ~coin-flip (~0.57) | **0.825 noncoding** (domain-scoped) | **0.837 noncoding** |
| EDA | light | light | **intense, first, as a gate** |
| Splice confound | not examined | not examined | **found in EDA + measured by ablation** |
| Validation gate | none | none | **`validate.py` hard gate before training** |
| Reproducibility | — | — | **byte-identical hash vs v2 (`positives_ref.tsv`)** |
| Job arrays | basic | QOS-packed (hardcoded `sed`) | **manifest-driven (single source of truth)** |
| Submission | one-shot | one-shot | **two-phase gate (data→validate→STOP→models)** |

## 12.3 The changes, explained

**Technical:**
- **Multi-chromosome split.** v1/v2 put the whole test on chr1; now test spreads across chr1/chr2/chr20
  and val across four chromosomes. Spreading the test set averages out any single chromosome's quirks
  → a more trustworthy generalization estimate. (It also matches a teammate's split for direct
  comparability — see conceptual, below.)
- **Manifest-driven arrays.** v2 hardcoded which protein a task handled via arithmetic on line numbers
  inside the `.sbatch`. Now `make_manifests.sh` generates `manifest_{prep,cnn,lm}.txt` from the config,
  and each task just reads its line. One source of truth; changing the panel is a config edit.
- **The validation gate + reproducibility hash.** Brand new. `validate.py` runs hard checks (balance,
  split rule, leakage, floors) and a SHA-256 comparison to v2 (`positives_ref.tsv`) *before* any GPU
  time. This is why the run could claim `vs_v2 = match` — proof, not assertion.
- **The splice-matched negative mode + `--negatives` switch.** `data_prep.py` gained a second negative
  sampler (distance-to-splice-site matched) and the training scripts a `--negatives` flag, so the same
  code produces both the primary result and the ablation.
- **Two-phase, QOS-aware submission.** `submit_data.sh` (prep→validate, then STOP) and
  `submit_models.sh` (cnn+lm→aggregate), with ClinVar separated to respect the 8/4 gpu cap.

**Conceptual:**
- **v1 → v2 was about *fairness and fit*.** v1's RNABERT was frozen, so the comparison wasn't
  apples-to-apples; v2 tuned every model to its peak. And v2 asked a real scientific question — *does
  matching the model's pretraining to the biology (SpliceBERT on pre-mRNA) beat raw size?* — and
  answered yes. It also gave ClinVar *jurisdiction* by restricting to noncoding variants at real sites,
  turning a coin-flip into 0.825.
- **v2 → rbp-binding was about *rigor and honesty*.** Three shifts: (1) **comparability** — adopt the
  teammate's exact split so numbers can sit side by side; (2) **provability** — a validation gate and a
  reproducibility hash so "the data is correct" is checked, not hoped; (3) **interrogation** — don't
  just report that SpliceBERT wins on an intron-heavy panel; *test whether that win is real or a
  splice-site artifact* via the ablation. The ablation's near-zero drops are the payoff: the result
  survived its own stress test.

**The single most important new idea:** *measure a confound instead of removing it.* EDA showed a
splice-site signal; the tempting move is to "clean" the negatives to erase it. Instead we kept the
comparable primary set **and** built a splice-matched control, so the data could *tell us* whether the
confound mattered. It didn't — and now we can say so with evidence.

---

# Appendix — Glossary of cluster/ops terms

- **HPC cluster** — many networked computers (nodes) + shared filesystem + a job scheduler.
- **Login node / compute node** — the shared front door / the workhorses you reach via the scheduler.
- **Slurm** — the scheduler. **partition** — a named pool of nodes. **QOS** — per-user usage caps.
- **`sbatch` / `srun` / `salloc`** — submit a batch job / run interactively / reserve a node.
- **job array** — one script run as many indexed tasks (`$SLURM_ARRAY_TASK_ID`); `%N` throttles concurrency.
- **`#SBATCH`** — directive lines in a batch script (`--partition`, `--gres`, `--mem`, `--time`, `--array`, `--output`).
- **`--gres=gpu:1`** — request 1 GPU ("generic resource"). **`--mem` / `--cpus-per-task`** — RAM / cores.
- **`squeue`** — live queue. **states:** `PD` pending, `R` running, `CG` completing, `CD` completed, `F` failed.
- **pending reasons:** `(None)`, `(Priority)`, `(Resources)`, `(Dependency)`, `(JobArrayTaskLimit)`.
- **`sacct`** — accounting/history (finished jobs). **State** `COMPLETED`, **ExitCode** `0:0` = clean.
- **`--dependency=afterok:ID`** — start only after job ID succeeds (the chain). **`--parsable`** — print just the job ID.
- **`scancel`** — cancel jobs (by ID, `--name`, or `-u $USER`).
- **`scontrol show node`** — a node's real specs (CPUs, RAM, `Gres`). **`sinfo`** — partition/node overview.
- **module** — `module load python/3.13.5` makes a specific tool version available.
- **venv** — an isolated Python environment (`.venv/`).
- **CUDA / cu118** — GPU compute platform / the CUDA-11.8 torch build that matches Explorer's driver.
- **multimolecule** — HuggingFace-compatible library of pretrained RNA/DNA/protein models.
- **transformers / tokenizers / huggingface-hub / peft** — model machinery / text→tokens / weight
  download / LoRA fine-tuning.
- **state_dict / checkpoint (`.pt`)** — a saved dictionary of a model's weights.
- **UNEXPECTED / MISSING** — checkpoint keys the model doesn't use / model keys the checkpoint lacks
  (newly initialized) — benign when you're using only the base encoder + your own head.
- **rsync** — differential file sync over SSH. **`-a -v -z --progress --partial --timeout`**, `-e ssh
  -o ServerAliveInterval` (keepalive), `--exclude`, trailing-slash = "contents of."
- **scp / git / Globus** — dumb copy / code versioning / big-data transfer (why we chose rsync).
- **AUROC / AUPRC** — ranking-quality metrics (0.5 = chance). **delta** — ClinVar disruption score.
- **manifest** — a text file (one job per line) that maps an array index to real work.
- **phase gate** — the deliberate STOP after validation, before training.

---

*Companion to `RBP_BINDING_MASTERCLASS.md`. Between the two, every concept, command, output, error,
and decision in this project is written down. If a line here isn't clear, that's a bug in the doc.*
