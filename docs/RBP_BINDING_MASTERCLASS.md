# RBP-Binding — The Masterclass

**A complete, from-zero explanation of the `rbp-binding` project: the biology, the data, the
preprocessing, the exploratory analysis, the models, the validation gate, the downstream
variant analysis, and the HPC cluster engineering — every concept, every file, every design
decision, and every trade-off (why *and* why-not), down to the nitty-gritty.**

> Read this top-to-bottom once and you will understand not just *what* we built but *why* every
> knob is set the way it is, what each alternative would have cost, and how the whole thing runs
> as a parallel job on a supercomputer. Nothing is hand-waved.

---

## Table of contents

- **Part I — The biology and the problem** (what an RBP is, eCLIP, the prediction task)
- **Part II — The data and where every byte comes from** (ENCODE, GENCODE, ClinVar)
- **Part III — Preprocessing, line by line** (`data_prep.py`: windows, negatives, regions, splits, splice-matching)
- **Part IV — The chromosome split and the teammate comparison** (leakage, why these chromosomes)
- **Part V — Exploratory Data Analysis** (every figure, what we observed, the splice-site discovery)
- **Part VI — The models** (CNN, RNA-FM+LoRA, RNABERT, SpliceBERT — architectures and training)
- **Part VII — The validation gate** (why we validate before training, every check)
- **Part VIII — Downstream** (aggregation, the ClinVar variant-effect analysis, figures)
- **Part IX — The cluster** (Slurm, partitions, QOS, job arrays, dependencies, the phase gate)
- **Part X — Every file, annotated** (what it is, how we got it, what's inside)
- **Part XI — What we observed and what we expect** (EDA numbers, v2 baselines)
- **Part XII — The decision log** (every fork we hit, the choice, and the roads not taken)
- **Appendix A — Glossary** · **Appendix B — Commands** · **Appendix C — The exact hyperparameters**

A note on scope: this document describes `rbp-binding`, the *correct-from-scratch* rebuild. Where it
helps, it references the earlier `rbp-v2` run whose numbers we treat as a baseline/expectation, but
`rbp-binding` is self-contained.

---

# Part I — The biology and the problem

## I.1 What is an RNA-binding protein, and why do we care?

DNA is transcribed into RNA. Between being made and being used, an RNA molecule is spliced (introns
cut out), edited, transported, stabilised or degraded, and eventually translated. It never floats
around naked — it is constantly bound by **RNA-binding proteins (RBPs)**. An RBP is a protein that
physically grips specific stretches of RNA and, by gripping there, controls what happens next:

- **TARDBP (TDP-43)** and **FUS** bind long UG-rich / GU-rich stretches and regulate splicing; both
  are central to ALS and frontotemporal dementia.
- **RBFOX2** binds the exact motif `UGCAUG` and switches exons on/off.
- **PUM1 / PUM2** (Pumilio) recognise `UGUANAUA` and destabilise target mRNAs.
- **PTBP1** binds pyrimidine (C/U) tracts and represses splicing.
- **ELAVL1 (HuR)** binds AU-rich elements (AREs) in 3′UTRs and *stabilises* mRNA.
- **QKI** binds `ACUAAC` and controls myelination-related splicing.
- **U2AF1** is part of the core spliceosome — it defines 3′ splice sites.
- **IGF2BP1/2/3, LIN28B, MATR3, TIA1, EWSR1** — each with its own binding preference and biology.

**Why predict where they bind?** Because a *change in binding* is a change in regulation, and that is
how a lot of disease works. A single-nucleotide variant that destroys an RBP's landing site can
mis-splice a gene or de-stabilise a transcript without touching the protein-coding sequence at all.
If we have a model that knows where an RBP binds *from sequence alone*, we can ask of any variant:
"does this change disrupt binding?" — which is Part VIII of this project (ClinVar).

## I.2 How do we know where an RBP binds? — eCLIP

We don't guess; there is an experiment. **eCLIP** (enhanced crosslinking and immunoprecipitation)
works like this:

1. UV-crosslink proteins to the RNA they are touching *in living cells* (a covalent bond forms
   exactly where protein meets RNA).
2. Pull down (immunoprecipitate) your one RBP of interest with an antibody, dragging its bound RNA
   with it.
3. Sequence those RNA fragments and map them back to the genome.
4. Where fragments **pile up** far above background, the protein was bound. Those pile-ups are
   called **peaks**.

A peak is a genomic interval (chromosome, start, end, strand) where the protein was reproducibly
found. ENCODE runs eCLIP for hundreds of RBPs in two cell lines (**K562**, a leukemia line, and
**HepG2**, a liver line) and publishes **reproducible peaks** — peaks confirmed across two
biological replicates (this is the IDR, "irreproducible discovery rate", step; it throws away
one-off noise). Those reproducible-peak BED files are our ground-truth **positives**.

Key honesty point: eCLIP peaks are *where the protein was observed bound in that cell line*, which
is a function of both the protein's sequence preference **and** which transcripts were expressed and
accessible. That nuance matters later (Part V, the splice-site confound).

## I.3 The prediction task, stated precisely

For **each protein separately**, we build a **binary classifier**:

> Given a 101-nucleotide RNA sequence, output the probability that *this* protein binds in the middle
> of it.

- **Per-protein**, not one model for all: RBFOX2 and PTBP1 want completely different motifs, so each
  gets its own model. 16 proteins → 16 models per architecture.
- **Binary**: bound (positive) vs not-bound (negative). We need negatives — sequences the protein
  does *not* bind — and constructing *fair* negatives is the single most important preprocessing
  decision (Part III).
- **From sequence alone**: the input is just the 101 letters (A/C/G/U). No conservation scores, no
  structure, no expression — we want to test what pure sequence can do, and we want the model to be
  usable on *any* sequence (including hypothetical variant sequences).
- **101 nt window**: wide enough to contain a motif plus context, narrow enough that the signal is
  local. Odd number so there is a well-defined centre nucleotide (position 51, index 50).

## I.4 How we score success — AUROC and AUPRC

We never look at raw accuracy (it lies when classes are imbalanced). We use two threshold-free
metrics:

- **AUROC** (Area Under the ROC Curve): the probability that a randomly chosen positive gets a higher
  score than a randomly chosen negative. 0.5 = coin flip, 1.0 = perfect. It answers "can the model
  *rank* bound above unbound?"
- **AUPRC** (Area Under the Precision-Recall Curve, a.k.a. average precision): more sensitive to
  performance on the positive class; useful when positives are rare.

Because we build a **balanced 1:1** dataset (one negative per positive), a chance model scores AUROC
≈ 0.5 and AUPRC ≈ 0.5, so both are easy to read.

---

# Part II — The data and where every byte comes from

Everything downstream is only as good as these inputs. There are four data sources; three are
downloaded on the cluster by `cluster/download_data.sh`, and the peak files are also mirrored
locally.

## II.1 The protein panel — `config/proteins.tsv`

This tiny tab-separated file is the **single source of truth** for the whole pipeline. Every script,
every job array, every manifest reads it. It has 16 rows, one per protein:

```
protein   accession     cell_line
TARDBP    ENCFF593RED   K562
FUS       ENCFF861KMV   K562
RBFOX2    ENCFF206RIM   K562
PUM2      ENCFF880MWQ   K562
IGF2BP1   ENCFF650LMV   K562
MATR3     ENCFF246EPM   K562
TIA1      ENCFF918KMT   K562
EWSR1     ENCFF607ZRF   K562
IGF2BP2   ENCFF524ZZB   K562
IGF2BP3   ENCFF886SDQ   HepG2
ELAVL1    ENCFF566LNK   K562
LIN28B    ENCFF061XNA   K562
PTBP1     ENCFF907HNN   K562
QKI       ENCFF786UOW   K562
U2AF1     ENCFF640IHY   K562
PUM1      ENCFF094MQV   K562
```

- **`protein`** — the gene symbol; used as the folder name for that protein's data and as the label
  in results.
- **`accession`** — the ENCODE file ID for that protein's reproducible-peak BED. It is the exact
  file we download; `ENCFF593RED` is not a guess, it is a specific vetted file on encodeproject.org.
- **`cell_line`** — K562 for 15 of them, HepG2 for IGF2BP3 (we picked whichever cell line had the
  better/available reproducible-peak file for that protein).

**Why these 16?** They are a *splicing- and UTR-regulation–heavy* panel with well-characterised
motifs (so we can sanity-check the model against literature), spanning disease-relevant proteins
(TDP-43, FUS), classic motif proteins (RBFOX2, QKI, Pumilio), and a few harder ones. **Why not
more?** Three candidates were dropped because their reproducible-peak files were too small to build a
trustworthy dataset: **HNRNPA1** (~199 usable pairs), **TAF15** (~545), **HNRNPC** (~516). Below a
few hundred pairs the per-protein val/test sets fall under our 100-pair floor and the metrics become
noise, so including them would *lower* the quality of the whole comparison. Sixteen well-populated
proteins beats nineteen where three are unreliable.

## II.2 The eCLIP peaks — the BED files

For each protein we have `data/raw/encode/<PROTEIN>.<ACCESSION>.bed.gz` — a gzip-compressed BED
(narrowPeak) file. Each line is one reproducible peak:

```
chrom   start   end   name   score   strand   signalValue   pValue   qValue   peak
chr1    517552  517653  .     200     -        4.2           15.3     8.1      -1
```

We only use columns 1, 2, 3, 6: **chromosome, start, end, strand**. `start`/`end` are 0-based
half-open (BED convention: the interval is `[start, end)`). The **strand** matters enormously for
RNA: the `-` strand means the RNA is the reverse complement of the reference, and an RBP motif only
makes sense read in the RNA's own 5′→3′ direction (Part III.3).

**How we got them:** `download_data.sh` reads `config/proteins.tsv` and, for each row, curls
`https://www.encodeproject.org/files/<ACC>/@@download/<ACC>.bed.gz`. Locally, some of these are
symlinks pointing at the copies already downloaded for the earlier v1 project — identical files,
just not re-downloaded to save bandwidth.

## II.3 The genome — `GRCh38.primary_assembly.genome.fa`

To turn a peak (coordinates) into a **sequence** (letters), we need the reference human genome. We
use **GRCh38 primary assembly** from **GENCODE release 45**. "Primary assembly" means the main
chromosomes (chr1…chr22, chrX, chrY, chrM) plus scaffolds, but *without* the alternate haplotype
contigs — we restrict to the standard chromosomes anyway (`STD` in the code). It is a ~3 GB FASTA
(a plain-text file of chromosome sequences). We read it with `pyfaidx`, which builds a small `.fai`
index so we can grab any sub-sequence (e.g. `chr1:517552-517653`) in O(1) without loading 3 GB into
RAM.

## II.4 The gene annotation — `gencode.v45.primary_assembly.annotation.gtf.gz`

A peak lands somewhere in the genome, but *where in a gene*? Inside an intron? A 3′UTR? The coding
sequence? That "region type" matters because we match negatives **within the same region type**
(Part III.4). The **GTF** (Gene Transfer Format) is GENCODE's annotation: for every transcript it
lists the genomic intervals of its exons, CDS (coding sequence), and UTRs. We parse it in two passes
(Part III.5) to (a) find which transcript a peak sits in and (b) know that transcript's exon/CDS/UTR
layout. Matching versions matters: we use GENCODE **v45** for both the genome and the GTF so
coordinates line up exactly.

## II.5 The disease variants — `clinvar.vcf.gz`

For the downstream analysis (Part VIII) we need known disease variants. **ClinVar** is NCBI's public
archive of human variants with clinical interpretations. It is a **VCF** (Variant Call Format) file:
each line is a variant — chromosome, position, reference allele, alternate allele, and an INFO field
with clinical significance (`CLNSIG` = Pathogenic / Benign / …) and molecular consequence (`MC` =
missense / intron / splice / …). We keep only clean single-nucleotide Pathogenic/Benign variants and
ask whether our binding models react to them. Downloaded from
`ftp.ncbi.nlm.nih.gov/pub/clinvar/vcf_GRCh38/clinvar.vcf.gz`.

---

# Part III — Preprocessing, line by line (`src/data_prep.py`)

This is the heart of the project. Garbage here poisons everything, so every step is deliberate. The
script builds the dataset for **one** protein (so it can run as one task in a job array) and writes
`data/processed/<PROTEIN>/dataset.tsv` + `onehot.npz`. Below, each stage with the *why*.

## III.1 The constants (the knobs)

```python
WIN, HALF, GC_BAND, MIN_DIST, BIN, SEED = 101, 50, 0.05, 500, 1 << 17, 7
TEST_CHR = {"chr1", "chr2", "chr20"}
VAL_CHR  = {"chr19", "chr16", "chr13", "chr18"}
SPLICE_EDGES = [50, 150, 500, 1500]
```

- **`WIN = 101`** — window length. Odd → a true centre (index 50). Long enough for a motif (4–8 nt)
  plus flanking context the models can use; short enough to keep the signal local and the language
  models fast.
- **`HALF = 50`** — half-window; a window is `[mid-50, mid+51)` = exactly 101 nt.
- **`GC_BAND = 0.05`** — a negative's GC content must be within ±5% of its positive's (Part III.4).
- **`MIN_DIST = 500`** — a negative must be ≥500 nt away from *any* peak of this protein, so we never
  accidentally label a real (just un-called) binding site as a negative.
- **`BIN = 1 << 17 = 131072`** — the bin size for a coarse spatial index (Part III.6); a performance
  trick, not a biological parameter.
- **`SEED = 7`** — the RNG seed. Everything random (which negative we pick) is deterministic given
  this seed, so the dataset is **exactly reproducible** — the basis of the validation hash (Part VII).
- **`TEST_CHR / VAL_CHR`** — the chromosome split (Part IV).
- **`SPLICE_EDGES`** — distance buckets for the splice-matched ablation (Part III.9).

## III.2 Loading peaks — `load_peaks`

Reads the gzipped BED, keeps only standard chromosomes (`STD`), and returns tuples
`(chrom, start, end, strand, mid)` where `mid = (start + end) // 2` is the peak's **midpoint**. We
centre our window on the midpoint rather than the whole peak because peaks vary in width (Part V.8)
but our model needs a fixed 101-nt input; the midpoint is our best single estimate of "where the
protein actually sat".

## III.3 Positives — extracting the window sequence — `get_seq`

```python
def get_seq(chrom, a, b, strand):
    s = str(fa[chrom][a:b]).upper()
    return revcomp(s) if strand == "-" else s
```

Grab the reference bases for `[mid-50, mid+51)`. **The strand correction is critical.** The genome
FASTA is always the `+` strand. If the peak is on the `-` strand, the RNA that the protein actually
saw is the **reverse complement** of the reference. `revcomp` reverses the string and complements
each base (A↔T, C↔G). Skipping this would feed the model the wrong strand for ~half the data and
scramble every motif. We then store the DNA sequence and its RNA version (`T→U`) — models that were
pretrained on RNA expect `U`.

We discard a window if it is not exactly 101 nt (peak too close to a chromosome end) or contains an
`N` (an unsequenced base) — an `N` is missing information and would be a fake input.

## III.4 Negatives — the matched-negative philosophy (the crux)

A classifier is only as meaningful as its negatives. If we picked negatives *randomly* from the
genome, the model could "cheat": most of the genome is AT-rich intergenic sequence, while binding
sites often sit in GC-flavoured, exonic, gene-dense regions. The model would then learn
"gene-rich vs desert", not "this protein's motif". That is a classic **shortcut** — high AUROC, zero
biology.

So for **every positive** we draw **one matched negative** that is deliberately similar in every way
*except* that the protein does not bind it:

1. **Same region type.** If the positive is in an intron, the negative is drawn from an intron; if a
   3′UTR, from a 3′UTR. This removes the "region" shortcut.
2. **GC within ±5%.** The negative's GC content must match the positive's within `GC_BAND`. This
   removes the "GC composition" shortcut — arguably the biggest one, because many RBP motifs are
   GC- or AU-flavoured and you don't want the model to win on base composition alone.
3. **≥500 nt from any peak** (`MIN_DIST`). eCLIP does not call *every* real site; a sequence 20 nt
   from a called peak is probably also bound but just wasn't called. Requiring 500 nt of distance
   makes "negative = truly unbound" much safer.
4. **Drawn from a bound transcript.** Negatives come from the same transcript as the positive first,
   then from a pool of other transcripts this protein binds — so negatives live in the same kind of
   expressed, accessible RNA as positives (not some silent gene).

The payoff: a model that beats this negative set has learned something *specific* to the protein's
grip, because everything cheap (region, GC, transcript context) was held constant. This is the
single most defensible thing about the whole pipeline — and Part V shows one place it is *not quite*
enough (splice-site proximity), which is exactly why we add the ablation.

## III.5 Region annotation — the two-pass GTF read

To know a peak's region type, we need the gene structure. Loading the whole GTF's exon table for
every transcript in the genome would be wasteful, so we do it in two passes:

- **Pass 1 — `load_gtf`:** read only `transcript` lines → remember each transcript's span
  `(chrom, start, end, strand, type)` and index it into coarse **bins** (`BIN`-sized genomic
  buckets) so we can later ask "which transcripts overlap point *p*?" quickly.
- **`find_tx`:** for a peak midpoint, look up its bin, and among transcripts overlapping that point
  pick the **longest**, preferring **protein_coding** ones (a `+10^12` score bonus). Rationale: a
  point often sits inside several overlapping transcripts; the longest protein-coding one is the most
  informative, stable choice for assigning a region.
- **Pass 2 — `load_gtf_features`:** now that we know the set of **bound** transcripts (the ones our
  peaks landed in), read `exon`/`CDS`/`UTR` lines **only for those transcripts**. Far less memory.

- **`region_of`:** given a point and its transcript's features, classify it as `CDS`, `5UTR`,
  `3UTR`, `exon` (non-coding exon of a coding gene), `ncRNA_exon`, or `intron`. The 5′-vs-3′ UTR
  decision uses the CDS extent and the strand (on `-` strand, "before the CDS" flips).

## III.6 The "forbidden zone" index — `build_forbidden` / `forbidden`

To enforce "≥500 nt from any peak" fast, we pre-expand every peak by ±500 nt and drop those
intervals into the same `BIN` bucket index. Checking whether a candidate negative window is too
close is then a quick lookup of a couple of bins instead of scanning thousands of peaks. This is the
`BIN = 131072` trick — pure performance, no effect on results.

## III.7 Finding valid negative starts — `candidate_starts`

Given a region interval `[a, b)`, this returns every window start where a negative is legal:

- the 101-nt window fits inside the interval,
- it contains **no `N`**,
- its **GC is within the band** of the positive's GC,
- it is **not forbidden** (≥500 nt from peaks).

The GC and N checks are vectorised with **prefix sums**: we build a cumulative count of G/C (and of
N) across the interval once, so the GC fraction of *any* 101-nt sub-window is an O(1) subtraction.
Without this, scanning a 50 kb intron base-by-base would be brutally slow.

## III.8 Picking the negative — `make_negative` (with GC relaxation)

We try to find a legal negative **in the positive's own transcript first**, then in up to 60 other
bound transcripts of the same region type. If nothing is found at the strict ±5% GC band, we
**relax** the band: 0.05 → 0.10 → 1.0 (the last means "GC unconstrained, take anything legal"). We
record which band was used (`relax` counter) so EDA can report how often we had to relax — a quality
signal. In practice relaxation is rare (Part V), so almost all negatives are true ±5% GC matches.

## III.9 The splice-matched negatives (the ablation) — new in `rbp-binding`

This is the one genuinely new preprocessing idea versus v2, and it exists because of an EDA finding
(Part V.9): for some intron-heavy proteins, part of "positive vs negative" turns out to be
"near a splice site vs not", because binding sites cluster near splice junctions while our ordinary
same-region negatives can sit deep in the intron. A model could then score well by detecting splice
sites rather than the RBP's motif.

To *measure* that (not remove it — see Part XII for why measuring beats removing), we build a
**second** negative set for four proteins in which negatives are **also matched on distance to the
nearest splice site**:

- **`splice_sites(feat, tid)`** — a transcript's internal exon boundaries (donors + acceptors) are
  its splice sites; we collect them (the first exon-start and last exon-end are transcript ends, not
  splice sites, so they're excluded).
- **`splice_bucket(pos, sites)`** — the distance from a position to its nearest splice site, bucketed
  by `SPLICE_EDGES = [50, 150, 500, 1500]` → bucket 0 (`<50 nt`), 1 (`50–150`), 2 (`150–500`),
  3 (`500–1500`), 4 (`≥1500`). Bucketing (rather than exact matching) is robust and interpretable.
- In splice-matched mode, a candidate negative must fall in the **same distance bucket** as its
  positive, on top of all the ordinary constraints.

This runs for **PUM1, LIN28B, U2AF1** (the splice-driven proteins) and **RBFOX2** (a clean-motif
control), writing `dataset.splice_matched.tsv` / `onehot.splice_matched.npz` alongside the primary
files. The `--negatives {primary,splice_matched}` flag selects the mode; **primary is byte-identical
to v2** because when the flag is `primary` no splice code runs and the RNG draw order is unchanged.

## III.10 One-hot encoding — `one_hot`

For the CNN (which eats numbers, not letters) we also store a **one-hot** tensor: a 4×101 matrix
where row 0=A, 1=C, 2=G, 3=T and a 1.0 marks the base at each position (an `N` becomes an all-zero
column). Saved compressed in `onehot.npz` together with the labels and the split assignment, so the
CNN never has to re-parse text.

## III.11 The output files

For each protein, `data/processed/<PROTEIN>/`:

- **`dataset.tsv`** — the human-readable table, one row per window:
  `id, label, chrom, start, end, strand, region, gc, split, seq_dna, seq_rna`.
  - `id` — e.g. `PUM1_pos_0`, `PUM1_neg_1` (deterministic order).
  - `label` — 1 positive, 0 negative.
  - `split` — `train` / `val` / `test`, computed from `chrom` (Part IV).
  - `seq_rna` — the RNA sequence the language models consume.
- **`onehot.npz`** — `X` (N×4×101 float32), `y` (labels), `split` (per-row split string) for the CNN.
- **`dataset.splice_matched.tsv` / `onehot.splice_matched.npz`** — only for the 4 ablation proteins.

Everything is deterministic: same inputs + `SEED=7` + same code ⇒ identical bytes, which is what the
validation gate checks.

---

# Part IV — The chromosome split and the teammate comparison

## IV.1 The one mistake that would ruin everything: leakage

Machine-learning models are lazy — they memorise anything they can. If two windows that are basically
the same sequence end up one in *train* and one in *test*, the model can "recall" the training one at
test time and post a fake-high score. In genomics the biggest source of this is **the genome
repeats itself and neighbouring windows overlap**. Two peaks 30 nt apart share almost their whole
101-nt window; a repeat element appears in many places. If train and test aren't cleanly separated,
your test AUROC is a lie.

## IV.2 Why split by *chromosome*

The clean fix is to split by **whole chromosomes**: every window from a chromosome goes entirely into
one of train/val/test, never split across. Because a genomic locus lives on exactly one chromosome,
two overlapping windows are guaranteed to be on the same chromosome and therefore in the same split —
no overlap leakage is even possible. It's the standard, defensible choice for this kind of task.

**Why not a random 80/10/10 split of windows?** Because random splitting scatters overlapping and
repeat-sharing windows across train and test → leakage → inflated scores. A random split *looks*
better (higher AUROC) precisely because it is cheating. We reject it.

## IV.3 The exact split we use (the teammate standard)

```
test  = { chr1, chr2, chr20 }
val   = { chr19, chr16, chr13, chr18 }
train = everything else
```

`split_of(chrom)` maps each chromosome to its split. We adopted **exactly** the split a teammate is
using, for one overriding reason: **comparability**. Two AUROCs are only comparable if they were
measured on the same test data. If we and the teammate used different splits, we couldn't put our
numbers next to theirs. A "shared but slightly sub-optimal" split beats a "private but optimal" one.

Observed proportions on our data (Part V): **64.6 % train / 15.5 % val / 19.9 % test** — a healthy,
balanced partition, and every one of the 16 proteins keeps ≥100 pairs in both val and test.

## IV.4 Why-not the alternatives (the roads not taken)

- **A leaner multi-chromosome split** (e.g. `test={chr1,chr8}, val={chr2,chr9}`, ~80/10/10). This
  keeps *more* training data and is still robust, and it was genuinely tempting. We rejected it
  **only** because it isn't the teammate's split — comparability won.
- **Single-chromosome test (`test=chr1` only)**, which the earlier v2 used. A single chromosome can be
  atypical (chr1's specific genes/composition), so its AUROC is a noisier estimate of generalisation.
  Spreading test across three chromosomes averages out chromosome-specific quirks — a better estimate.
- **Note the cost we accepted:** `chr19` is the most gene-dense chromosome and it sits in *val*, so
  val is inflated and gene-dense-biased for some proteins (EWSR1's val ends up larger than its test).
  We keep it because matching the teammate matters more than tidying this.

---

# Part V — Exploratory Data Analysis (`src/eda.py`, `notebooks/01_eda.ipynb`)

## V.1 Why EDA *before* modelling

The point of EDA here is not pretty pictures — it is to **stress-test the modelling assumptions
before we spend GPU-hours training on flawed data**. Concretely we ask: did GC-matching actually
work? Is the split balanced? Is there hidden leakage? Is there real motif signal? And crucially — is
there anything that *defies* the plan? We found exactly one such thing (V.9), and it changed the
design. Doing this after training would have meant re-running everything.

Two forms exist: `src/eda.py` (headless script → PNGs + `eda_summary.tsv`, runs on the cluster) and
`notebooks/01_eda.ipynb` (the same analysis with code + figures + narrative inline, for humans). The
preview was run on the existing windows (identical sequence content; only the split label differs),
which is valid for designing the pipeline.

## V.2 Dataset size & class balance (fig 01)

Matched pairs per protein, biggest to smallest: ELAVL1 8821, EWSR1 8786, PTBP1 7791, LIN28B 6916,
IGF2BP1 5706, TIA1 5883, PUM2 5382, MATR3 4868, TARDBP 4589, FUS 4464, IGF2BP2 3833, RBFOX2 3519,
PUM1 2658, IGF2BP3 2218, QKI 2189, **U2AF1 1230** (smallest). Every protein is exactly 1:1
positive:negative by construction. Takeaway: even the smallest protein has ~1.2k pairs — enough to
train, and the check confirms the balance assumption holds.

## V.3 Split balance — the decisive check (fig 02)

For each protein, how many pairs land in train/val/test under the new split, with a dashed 100-pair
floor. **Result: every protein clears the floor in both val and test.** Smallest evaluation sets:
U2AF1 (val 133, test 247), QKI (val 187, test 473). These are thin but usable; we **accept** them
rather than complicate the pipeline. Overall 64.6/15.5/19.9 %. This single figure is why we could
confidently proceed with the teammate's split — it *works* for all 16.

## V.4 Chromosome distribution (fig 03)

A heatmap of "% of each protein's positives on each chromosome", with the test (orange) and val
(gold) chromosomes boxed. Observations: **chr1 is hot for everyone** (biggest chromosome, most
peaks) — a large test contribution. **chr19 (val) is bright** for gene-dense-loving proteins (EWSR1,
FUS), which is why those proteins have oversized val sets. **chr13/chr18 (val) are dim** (gene-poor),
contributing little. This visualises the chr19 skew we accepted in IV.4.

## V.5 GC match (figs 04, 05)

Per-protein mean GC of positives vs matched negatives (should overlap) and the pooled GC
distributions. **Result: the two clouds sit on top of each other; the largest mean-GC gap across all
16 proteins is ≤0.010.** The GC-matching worked — the model cannot win on base composition. This
validates the single most important negative-sampling design choice (III.4).

## V.6 Nucleotide composition (fig 06)

A/C/G/U frequencies, positives vs negatives, pooled. Because negatives are GC-matched, the gross
composition is similar — any residual difference is finer-grained (the actual motif), which is what we
*want* the model to have to find.

## V.7 Region composition (fig 07)

The 5UTR/3UTR/CDS/exon/ncRNA/intron makeup per protein. Most of the panel is **intron-dominated**
(many proteins 90 %+ intronic). This is real biology (these are co-transcriptional / splicing
regulators) and it is the setup for two things: (a) why a splice-aware model has an edge, and (b) the
splice-site confound (V.9).

## V.8 Peak widths (fig 08)

Boxplots of the raw eCLIP peak widths per protein — a sanity check on the signal before we force
everything to 101 nt. Widths vary (tens to a couple hundred nt), which is exactly why we centre a
fixed window on the **midpoint** rather than trying to use the raw peak.

## V.9 Motif signal — and the discovery that changed the design (figs 09, 10)

**Fig 09 (top enriched 6-mer per protein):** for each protein we compute the 6-mer most enriched in
positives vs negatives (log2 ratio, min-count filtered) and check it against the literature motif:

- **Clean recoveries:** RBFOX2 `UGCAUG` ✓, QKI `ACUAAC` ✓, PTBP1 `UCUCUC` ✓ (pyrimidine tract),
  TARDBP `GUGUGU` (the UG-repeat), PUM2 `UGUACA` (≈ Pumilio `UGUAUA`), IGF2BP1 `GUGUGU`.
- **The surprise:** for **PUM1, LIN28B, U2AF1** the top 6-mer is **`GGUAAG` — the 5′ splice-donor
  consensus**, *not* the protein's own motif (PUM1 "should" be `UGUAUA`).

**Fig 10 (where the motif sits in the window):** if a peak marks a true binding site, its motif
should peak near the window **centre**. For the clean proteins it does (broad central hump). For
**PUM1 and LIN28B the `GGUAAG` signal is off-centre and spiky** — it appears at fixed offsets, exactly
what you'd see if the window is positioned relative to a splice junction rather than the RBP's own
grip.

**Interpretation:** these intron-heavy proteins bind *near splice sites*, and because our ordinary
negatives (same region, GC-matched) can sit deep in the intron, **part of "positive vs negative" is
"near a splice site vs not"** — a confound. It is real biology (and it explains why a splice-aware
model like SpliceBERT wins), but it means we cannot *assume* the model learned the RBP's own motif for
those proteins. **This single observation is the reason the splice-matched ablation exists** (III.9,
Part XII).

## V.10 Complexity vs redundancy (fig 11)

- **Low-complexity fraction** (windows dominated by one base or a long homopolymer): **ELAVL1 is 38 %
  low-complexity** — it binds AU-rich elements, so many of its windows are U/AU runs. U2AF1 (18 %) and
  TIA1 (9 %) are also elevated (U-rich binders). This is expected biology, not a bug.
- **Exact-duplicate windows: 0 % for every protein.** No redundancy inflation.

## V.11 The red-flag stress test

An automated pass flags: GC gap >0.03, any val/test <100 pairs, duplicate windows >5 %, weak motif
enrichment, and — importantly — **any sequence appearing in more than one split** (repeat-driven
leakage). On our data it printed **✅ no red flags**: GC matched, negatives in-band, positives spread,
motifs enriched, no leakage. That green light is what let us move to preprocessing-freeze.

**Bottom line of Part V:** the design holds; the one consequence is the splice-matched ablation on 4
proteins. Nothing else about the plan changed.

---

# Part VI — The models

We compare **four model families**, deliberately spanning four decades of ideas — a tiny from-scratch
CNN, two general RNA language models (one huge, one tiny), and a splicing specialist. The rule is
**every model is tuned to its own peak** so the comparison is fair (v2's first attempt left RNABERT
frozen, which wasn't apples-to-apples; here everyone gets a real chance).

## VI.1 The CNN baseline — `src/models/cnn.py` (DeepBind-style)

```
(B,4,101) one-hot
  → Conv1d(4→16, kernel 12, same)  → ReLU → MaxPool(4)
  → Conv1d(16→32, kernel 8, same)  → ReLU → AdaptiveMaxPool(1)
  → Flatten → Linear(32→64) → ReLU → Dropout → Linear(64→1)  → logit
```

- **~7,000 parameters** — minuscule. This is the classic **DeepBind** idea (2015): the first Conv
  layer's 16 kernels of width 12 are effectively **learned motif detectors** sliding along the
  sequence; `MaxPool`/`AdaptiveMaxPool` ask "does this motif appear *anywhere* in the window?"
  (position-invariance); the small MLP head combines motif presences into a score.
- **Why include it:** it is the honest baseline. If a 7k-parameter CNN nearly matches a 100M-parameter
  language model (it does — v2: 0.864 vs 0.876), that is a *result* — it says most of the signal is a
  local motif, not long-range context.
- **Trained on CPU** (`DEVICE="cpu"`): the model is so small that CPU is faster than paying GPU
  transfer overhead, and it keeps GPUs free for the language models.
- **Data helpers** live here too: `load_split_arrays` (slice the one-hot tensor by split),
  `make_loader`, `predict_probs` (sigmoid the logits at inference).

## VI.2 What a "language model for RNA" is

`RNA-FM`, `RNABERT`, `SpliceBERT` are **transformers pretrained on millions of RNA sequences** the way
BERT was pretrained on text: mask out some nucleotides, predict them, and in doing so learn a rich
per-nucleotide representation ("embedding"). We get them from the **`multimolecule`** library (a
HuggingFace-compatible zoo of nucleic-acid models). To turn a pretrained encoder into a binding
classifier we wrap it in `LMClassifier` (`src/train_lm.py`):

- **Tokenizer:** splits the RNA string into per-nucleotide tokens and adds special tokens
  (`cls`/`eos`/`pad`).
- **Encoder:** outputs a vector per token (the last hidden state).
- **Masked mean pooling** (`masked_mean_pool`): average the per-nucleotide vectors **excluding** the
  `cls`/`eos`/`pad` tokens, giving one fixed vector for the whole window. (We exclude specials so
  padding can't dilute the signal — a subtle but real correctness point.)
- **Head:** `Linear(hidden→128) → ReLU → Dropout(0.3) → Linear(128→1)` → a single binding logit.

## VI.3 RNA-FM + LoRA — the 100M-parameter generalist, adapted cheaply

`RNA-FM` (`multimolecule/rnafm`) is a ~100-million-parameter general RNA model. Fully fine-tuning
100M parameters per protein × 16 proteins would be enormous and prone to overfitting on a few-thousand
-example dataset. So we use **LoRA (Low-Rank Adaptation)**:

- **The idea:** freeze the giant pretrained weights; inject tiny trainable **low-rank** matrices into
  the attention `query` and `value` projections. Instead of updating a `d×d` weight, you learn two
  skinny matrices `d×r` and `r×d` (`r=8`) whose product is the update. You train **<1 %** of the
  parameters, get most of the benefit, and can't easily overfit or wreck the pretrained knowledge.
- **Config:** `r=8, alpha=16, dropout=0.05, target_modules=["query","value"]` (`apply_lora`). `alpha`
  scales the adapter's contribution; `r` is its rank/capacity.
- **Two learning rates:** the LoRA adapters train at `1e-4`, the fresh head at `1e-3` (the head starts
  from scratch so it can move faster). Weight decay `1e-2`.
- **Why include it:** it is the "big generalist" contender and the fair way to deploy a 100M model on
  small data. We save only the adapter+head weights (tiny), not the frozen 100M.

## VI.4 RNABERT — the tiny generalist, fully fine-tuned

`RNABERT` (`multimolecule/rnabert`) is ~0.5M parameters — small enough to **fully fine-tune**
end-to-end without overfitting. Encoder trains at `3e-5` (gentle, to preserve pretraining), head at
`1e-3`. Its role in the story: it is small, so where it underperforms (e.g. Pumilio motifs) it shows
the limit of a tiny model; v2 saw it jump from 0.716 (frozen) to 0.793 (fully fine-tuned) — which is
*why* we insist every model is tuned to peak.

## VI.5 SpliceBERT — the specialist, and (in v2) the winner

`SpliceBERT` (`multimolecule/splicebert`, ~20M params, fully fine-tuned) was **pretrained on
pre-mRNA** — introns *and* exons, i.e. exactly the splicing-rich sequence world our peaks live in.
Encoder `3e-5`, head `1e-3`. **The hypothesis this project tests:** matching the model's pretraining
to the biology beats raw size. In v2 it did — SpliceBERT led with mean AUROC 0.901, beating the 100M
RNA-FM (0.876) while being ~3.6× cheaper to run. Part V.9 explains *why* it wins (splice context is
part of the signal), and the ablation quantifies how much of that edge is splice-proximity.

## VI.6 Training protocol (identical across LMs) — `src/train_lm.py`

- **Optimiser:** AdamW, weight decay `1e-2`, batch size 32.
- **Up to 12 epochs**, **early stopping with patience 4** on **validation AUROC** — we keep the
  epoch's weights that maximise val AUROC, then evaluate **once** on test. Test is never used for any
  decision; it is touched a single time to report the final number.
- **Seed 7** everywhere.
- **Metrics out:** `results/metrics/<PROT>.<key>[.splice_matched].json` with val/test AUROC + AUPRC.
- **Weights out:** `models/lm/<PROT>.<key>[.splice_matched].pt` (for LoRA, only adapters+head).
- **`--negatives` switch:** `primary` (default) loads `dataset.tsv`; `splice_matched` loads
  `dataset.splice_matched.tsv` and suffixes its outputs — the mechanism that produces both sides of
  the ablation from one script (see the earlier explanation you asked about).

## VI.7 CNN training protocol — `src/train_cnn.py`

Same philosophy, CNN-specific: a small **grid search** over `(lr, dropout) ∈ {(1e-3,0.5), (1e-3,0.3),
(5e-4,0.5), (5e-4,0.3)}`, up to 40 epochs with patience 6, batch 64, AdamW weight decay `1e-4`, seed
7. Pick the `(lr,dropout)` with the best **val** AUROC, retrain-select by val, evaluate once on test.
Also supports `--negatives` for the ablation. Runs on CPU.

## VI.8 Why these four, and why-not others

- **Why a spread (7k → 0.5M → 20M → 100M):** it lets us plot **accuracy vs compute** and make the
  real point — bigger isn't automatically better; *fit-to-biology* and *tuning* matter more.
- **Why-not one giant model only:** you'd learn nothing about *where* the signal is (local motif vs
  long-range), and you couldn't show the specialist-beats-generalist result.
- **Why-not train one multi-protein model:** different proteins want different motifs; per-protein
  models are cleaner and let each protein's AUROC be interpreted on its own.

---

# Part VII — The validation gate (`src/validate.py`)

## VII.1 Why validate *before* training at all

The rebuild's guiding principle is "correct-from-scratch". Training 64+ models and *then* discovering
the data was subtly wrong wastes days of GPU time and — worse — can produce a plausible-looking but
invalid result. So we insert a **hard gate** between "data is built" and "training starts": the
pipeline **stops** and the data must pass an automated audit first. This directly answers the
question "will we validate our results before model building?" — yes, structurally, it cannot be
skipped.

## VII.2 The hard checks (any failure ⇒ exit 1, do not train)

Run on the **frozen cluster data**, per protein:

1. **Files exist** — `dataset.tsv` + `onehot.npz` for all 16; the splice variants for the 4 ablation
   proteins. A missing file means a prep job silently failed.
2. **Class balance is exactly 1:1** — positives == negatives. Catches a broken negative sampler.
3. **val ≥ 100 and test ≥ 100 pairs** — below this the AUROC is too noisy to trust.
4. **Split matches the chromosome rule** — the stored `split` column must equal `split_of(chrom)` for
   every row (no stale or corrupted split), **and** no chromosome may appear in more than one split.
   This is the anti-leakage assertion made executable.
5. **No cross-split identical sequences** — if the *same* `seq_rna` shows up in two different splits
   (possible via genome repeats), that's leakage; we count and fail on it.
6. **Reproducibility vs v2** — see VII.3.

## VII.3 The reproducibility hash — `positives_ref.tsv`

We claim the primary path is "byte-identical to v2". We don't just assert it — we **prove it**:

- `content_hash(dataset.tsv)` = SHA-256 of every column **except `split`**, read as strings (so
  floating-point re-formatting across pandas versions can't cause a false difference).
- `cluster/make_reference.py` was run **locally** on the known-good v2 data to write
  `cluster/positives_ref.tsv` (protein → hash). This file is committed.
- On the cluster, `validate.py` recomputes the hash from the freshly extracted data and compares. A
  **match** means the re-extract reproduced v2's content exactly (only the split label changed);
  a mismatch is surfaced as a **warning** (not a hard fail — a benign difference between the fresh
  download and v2 shouldn't block you, but you should look).

## VII.4 The soft checks (warn only)

- **GC gap per protein** (>0.03 warns).
- **Splice-matched retention** — how many pairs the splice-matched set keeps vs primary; if it drops
  below 70 %, splice-matching is starving that protein of negatives and we'd loosen the buckets.

## VII.5 We tested the gate itself

Before trusting it, we dry-ran `validate.py` locally on new-split data: **all 16 passed with
`vs_v2 = match`**, and a **negative test** (deliberately corrupting a split label and deleting an
ablation file) confirmed it correctly reports **VALIDATION FAILED** and refuses. A gate you haven't
seen fail is not a gate.

---

# Part VIII — Downstream: aggregation, ClinVar, figures

## VIII.1 Aggregation — `src/aggregate.py`

Each training run writes its own little JSON (so array tasks never clash). `aggregate.py` merges them:

- **`results/model_comparison.tsv`** — a protein × model table of test AUROC, plus a MEAN row. This is
  the headline table.
- **`results/all_metrics.tsv`** — every primary run, long-form, for the figures.
- **`results/ablation_splice.tsv`** — for the 4 ablation proteins, `primary_auroc` vs `splice_auroc`
  and their **`drop`**. A large drop = that protein/model's AUROC was substantially splice-proximity.
  This table is the whole point of the ablation, made concrete.

## VIII.2 The ClinVar variant-effect analysis — `src/clinvar.py` (the payoff)

This is where a *binding* model becomes a *disease* tool, with **zero disease training**. The logic:

**The premise.** If a model truly learned where an RBP binds, then a variant that lands in a binding
site and disrupts it should change the model's score. So define a **disruption score** for a variant:

> `delta = max over window-shifts of | p(reference sequence) − p(alternate sequence) |`

- We build the 101-nt window around the variant, once for the **reference** allele and once for the
  **alternate** allele, and score both with the model. The **difference** is how much that one base
  change moved the binding probability.
- **Why max over shifts** (`SHIFTS = [-40,-20,0,20,40]`): the motif may not sit dead-centre, so we
  slide the variant to 5 offsets within the window and take the **largest** disruption — the model
  gets its best shot at seeing the variant inside the motif. `MARGIN = 25`.
- We only consider variants that actually fall **in/near a real eCLIP peak** of some protein (a
  binding model has no business judging a variant nowhere near a site).

**The domain restriction (the honest part).** We split variants into **coding** vs **noncoding**
(from ClinVar's `MC` molecular-consequence field) and report them separately, only keeping clean
single-nucleotide **Pathogenic/Benign** variants (`CLNSIG`). A binding model has real jurisdiction
over **noncoding** variants at binding sites (that's the mechanism it models); coding variants often
cause disease by changing the *protein*, which sequence-level RNA binding doesn't capture. Reporting
them separately is scientific honesty, not cherry-picking.

**Scoring.** Each variant is scored by all four tuned models (CNN + the three LMs), taking the max
disruption per variant across the proteins whose sites it hits. Then we compute the **AUROC of the
disruption score at separating Pathogenic from Benign**, per stratum.

**The v2 result (expectation):** SpliceBERT reached **AUROC 0.825 on noncoding variants at real
sites** — from a coin-flip in v1 to a strong signal — precisely because pathogenic noncoding variants
work by disrupting the binding the model learned. Coding was ~0.60 (modest, as expected). Outputs:
`results/clinvar_scores.tsv`, `results/clinvar_summary.tsv`.

## VIII.3 Figures — `src/figures.py`

- **`model_comparison.png`** — mean test AUROC per model (bars) with per-protein dots, plus a
  "who wins each protein" panel.
- **`gpu_dashboard.png`** — mean train time, GPU utilisation, and peak memory per LM, parsed from the
  per-run `nvidia-smi` logs.
- **`acc_vs_compute.png`** — test AUROC vs training seconds (log x) — the "bigger isn't better" plot.
- **`splice_ablation.png`** — the AUROC **drop** (primary − splice-matched) per protein per model: the
  visual of "how much was splice proximity".

---

# Part IX — The cluster (NEU Explorer, Slurm)

## IX.1 What the cluster is and why we need it

Training 16 CNNs + 60 language-model fine-tunes on a laptop would take days and melt it. **NEU
Explorer** is a shared **HPC (high-performance computing) cluster**: many compute nodes (some with
GPUs) that you don't use interactively — you **submit jobs** to a scheduler, which runs them when
resources are free. The scheduler is **Slurm**. The art is expressing our work as **parallel jobs**
that fit the cluster's rules, so 60 fine-tunes run four-at-a-time across GPUs instead of one-by-one.

## IX.2 Slurm concepts you need

- **Partition** — a pool of nodes. We use `short` (CPU work, generous limits) and `gpu` (GPU nodes).
- **`sbatch script.sbatch`** — submit a batch job; the `#SBATCH` lines at the top request resources.
- **Job array (`--array=0-19`)** — submit *one* script that runs as many near-identical tasks, each
  with a different `$SLURM_ARRAY_TASK_ID`. This is how we run "one protein per task" cleanly.
- **Throttle (`%8`)** — `--array=0-19%8` means "20 tasks but at most 8 running at once" — polite
  resource use and a way to fit QOS limits.
- **QOS (quality of service) limits** — the cluster caps how many jobs one user can queue/run. On
  Explorer the **`gpu` QOS allows 8 submitted / 4 running**; `short` is effectively uncapped (50
  running). **An array's elements each count against the cap** — this is the constraint that shaped
  our GPU packing (IX.5).
- **Dependencies (`--dependency=afterok:JOBID`)** — start job B only after job A **succeeds**. This is
  how we chain prep → validate and cnn+lm → aggregate → clinvar automatically.
- **`--time=08:00:00`** — a wall-clock limit per job. We set **8 hours on everything** (your standing
  rule) — comfortably above what any task needs, so nothing is killed mid-run.

## IX.3 One-time setup — `cluster/setup_env.sh` (and the CUDA gotcha)

Run once on a compute node. It: `module load python/3.13.5`, makes a `.venv`, installs
`requirements.txt`, then **pins the RNA-LM stack** to exact versions
(`transformers==5.14.1`, `tokenizers==0.22.2`, `huggingface-hub==1.26.0`, `multimolecule==0.2.1`,
`peft==0.20.0`) so a future library update can't silently change behaviour, and uninstalls
`torchao` (an incompatibility). **The important line:**

```bash
pip install --force-reinstall torch --index-url https://download.pytorch.org/whl/cu118
```

Explorer's GPU nodes have an older driver (CUDA 12.3). The default PyTorch wheel is built against a
newer CUDA and **fails to initialise the GPU** on those nodes. The **cu118** (CUDA 11.8) build runs on
any modern driver — installing it is the fix that made GPU fine-tuning work at all. (This bit us in v2;
here it's pre-empted.)

## IX.4 Getting the data — `cluster/download_data.sh`

Reads `config/proteins.tsv`, curls each protein's peak BED from ENCODE, then pulls the GENCODE v45
genome + GTF and the ClinVar VCF, and gunzips the genome (pyfaidx needs it uncompressed). ~3 GB total.
This is why the heavy re-extract must happen on the cluster (the genome isn't kept locally).

## IX.5 The job arrays, and the QOS-aware packing

Everything is driven by **manifests** generated from config by `cluster/make_manifests.sh` — the
single source of truth, so nothing is hand-typed:

- **`manifest_prep.txt` / `manifest_cnn.txt`** — 20 lines each: `<protein>\t<negatives>` (16 primary +
  4 splice-matched).
- **`manifest_lm.txt`** — 60 lines: `<model>\t<protein>\t<negatives>` = (16+4) × 3 models.

The arrays:

- **`prep.sbatch`** — `short`, `--array=0-19%8`. Task *k* runs line *k+1* of the prep manifest → one
  `data_prep.py` run. 16 primary + 4 splice-matched datasets.
- **`train_cnn.sbatch`** — `short`, `--array=0-19%8`. One tuned CNN per line. CPU (the model is tiny).
- **`train_lm.sbatch`** — `gpu`, `--array=0-7%4`, `--gres=gpu:1`. **Here's the packing trick:** we
  have 60 LM fine-tunes but the gpu QOS allows only 8 submitted / 4 running. A naive 60-element array
  is rejected. So we use **8 array tasks**, and each task **walks `PER=8` lines** of the manifest
  (task *t* does lines `t*8+1 … t*8+8`), covering all 60 (the last few slots are empty and skipped).
  `%4` keeps ≤4 GPUs busy → fits the cap exactly. Only **primary** runs are GPU-profiled with
  `nvidia-smi` (the ablation runs skip profiling to keep the util logs clean).
- **`validate.sbatch`** — `short`. Runs `eda.py` (regenerate figures on the frozen data) then
  `validate.py` (the gate).
- **`aggregate.sbatch`** — `short`. `aggregate.py` + `figures.py`.
- **`clinvar.sbatch`** — `gpu`. `clinvar.py` + `figures.py` (needs the trained models).

## IX.6 The phase gate and the orchestration scripts

The pipeline is split into **two phases on purpose**, so the human reviews the data before any GPU
time is spent:

- **`submit_data.sh` (Phase A):** regenerate manifests → submit `prep` → submit `validate` with
  `afterok(prep)`. Then **stop**. You read `logs/validate_<id>.out`; it must say
  **VALIDATION PASSED**.
- **`submit_models.sh` (Phase B):** submit `cnn` (short) and `lm` (gpu) in parallel → `aggregate` with
  `afterok(cnn,lm)` → `clinvar` with `afterok(aggregate)`. One command, fully chained.

## IX.7 Fresh start — `cluster/clean_start.sh`

Wipes the slate for a clean run: `scancel -u $USER` (cancel all your jobs) and `rm -rf` the old
project trees (`$HOME/rbp-v2`, `$HOME/rbp-prediction`) plus this project's regenerable dirs
(`.venv/data/models/results/logs`). **Safe by design:** it is **dry-run by default** — it prints each
target with its size (or `(absent)`) and does nothing until you re-run with `--force`. So a wrong path
reveals itself as `(absent)` before any deletion.

## IX.8 Resource sizing (why these numbers)

- **CPU jobs:** 4 CPUs, 16 GB (prep needs to hold the GTF/genome index), 8 GB for CNN training.
- **GPU jobs:** 1 GPU, 8 CPUs, 24 GB. One GPU is plenty per fine-tune (batch 32, ≤100M params); more
  would just queue longer for no speedup. We measured the footprint rather than over-requesting —
  **smaller requests schedule sooner**.

---

# Part X — Every file, annotated

## X.1 Repository layout

```
rbp-binding/
├── config/proteins.tsv          the 16 proteins (single source of truth)
├── src/                         all the Python
│   ├── data_prep.py             build one protein's dataset (Part III)
│   ├── eda.py                   headless EDA → figures + summary (Part V)
│   ├── validate.py              the pre-training gate (Part VII)
│   ├── models/cnn.py            DeepBind CNN + data helpers (Part VI.1)
│   ├── train_cnn.py             tune + train the CNN (Part VI.7)
│   ├── train_lm.py              fine-tune RNA-FM / RNABERT / SpliceBERT (Part VI.6)
│   ├── aggregate.py             merge metrics + ablation table (Part VIII.1)
│   ├── clinvar.py               variant-effect analysis (Part VIII.2)
│   └── figures.py               result figures (Part VIII.3)
├── notebooks/01_eda.ipynb       the combined EDA (code + figures + narrative)
├── eda/figures/*.png, eda_summary.tsv   EDA outputs (preview; regenerated on frozen data)
├── cluster/                     the HPC kit (Part IX)
│   ├── clean_start.sh           dry-run-by-default fresh wipe
│   ├── setup_env.sh             venv + pinned stack + cu118 torch
│   ├── download_data.sh         peaks + genome + GTF + ClinVar
│   ├── make_manifests.sh        generate the array manifests from config
│   ├── make_reference.py        (run locally) build positives_ref.tsv
│   ├── positives_ref.tsv        v2 content hashes (reproducibility reference)  ← committed
│   ├── prep.sbatch / train_cnn.sbatch / train_lm.sbatch
│   ├── validate.sbatch / aggregate.sbatch / clinvar.sbatch
│   ├── submit_data.sh (Phase A) / submit_models.sh (Phase B)
│   ├── manifest_{prep,cnn,lm}.txt   generated; git-ignored
│   └── README_CLUSTER.md        the operator runbook
├── requirements.txt             core Python deps
├── .gitignore                   ignores data/models/results/logs/.venv/manifests
└── docs/RBP_BINDING_MASTERCLASS.md   this document
```

## X.2 What each artifact is, how we got it, what's inside

- **`config/proteins.tsv`** — hand-curated from ENCODE (verified accessions). Drives everything.
- **`data/raw/encode/*.bed.gz`** — downloaded from ENCODE (or symlinked from v1 locally). Reproducible
  eCLIP peaks per protein. *Inputs.*
- **`data/reference/GRCh38…fa` + `gencode.v45…gtf.gz`** — GENCODE v45, via `download_data.sh`. Genome
  sequence + gene annotation. *Inputs (cluster-only, ~3 GB).*
- **`data/raw/clinvar/clinvar.vcf.gz`** — NCBI ClinVar, via `download_data.sh`. Disease variants.
- **`data/processed/<P>/dataset.tsv` + `onehot.npz`** — *produced* by `data_prep.py` on the cluster.
  The model-ready windows. Git-ignored (regenerable, deterministic).
- **`data/processed/<P>/*.splice_matched.*`** — the ablation datasets (4 proteins only).
- **`cluster/positives_ref.tsv`** — *produced locally* by `make_reference.py` from v2 data; committed
  so the cluster can prove reproducibility. Contains `protein → sha256(content-without-split)`.
- **`cluster/manifest_*.txt`** — *produced* by `make_manifests.sh`; the array line-lists. Git-ignored.
- **`results/metrics/*.json`** — *produced* by the training jobs; one per run.
- **`results/model_comparison.tsv`, `ablation_splice.tsv`, `clinvar_summary.tsv`, `figures/*.png`** —
  *produced* by aggregate/clinvar/figures. The deliverables.
- **`eda/…` + `notebooks/01_eda.ipynb`** — *produced* by `eda.py` / the notebook builder. The EDA.
- **`.venv/`** — local Python env (pandas/numpy/matplotlib + notebook tooling) for previews; on the
  cluster, a separate `.venv` from `setup_env.sh`. Git-ignored.

**Rule of thumb:** anything under `data/`, `models/`, `results/`, `logs/`, `.venv/`, and the manifests
is *regenerable* and git-ignored; everything else (code, config, the reference hash, docs, the
notebook) is source and tracked.

---

# Part XI — What we observed, and what we expect

## XI.1 Observed now (from the EDA preview — real numbers on our data)

- **Split:** 64.6 / 15.5 / 19.9 % train/val/test; all 16 proteins clear the 100-pair floor
  (smallest val: U2AF1 133, QKI 187).
- **GC matching:** works — max mean-GC gap ≤ 0.010.
- **Redundancy/leakage:** 0 % duplicate windows; no cross-split identical sequences.
- **Motifs:** clean recovery for RBFOX2 `UGCAUG`, QKI `ACUAAC`, PTBP1 `UCUCUC`, TARDBP UG-repeat.
- **The confound:** `GGUAAG` (5′ splice donor) is the top 6-mer for PUM1/LIN28B/U2AF1, off-centre in
  position → the splice-site finding that drove the ablation.
- **Composition:** ELAVL1 38 % low-complexity (AU-rich, expected); most proteins intron-dominated.

## XI.2 Observed from training — the `rbp-binding` cluster results (primary sweep)

The full sweep ran on Explorer; all 20 CNN + all 8 LM-array tasks + aggregate completed cleanly, and
validation reported **`vs_v2 = match` for all 16** (the fresh cluster re-extract == v2 content,
byte-for-byte).

**Binding — mean test AUROC over 16 proteins (all models tuned to peak):**

| Model | Mean AUROC |
|---|---|
| **SpliceBERT** (pre-mRNA specialist) | **0.893** 🥇 |
| RNA-FM (LoRA) | 0.878 |
| CNN (DeepBind, ~7k params) | 0.860 |
| RNABERT (full FT) | 0.787 |

- **SpliceBERT wins 13/16** proteins (RNA-FM takes IGF2BP2, IGF2BP3, PTBP1). Ranking matches v2;
  absolute numbers a touch lower than v2's 0.901 because the 3-chromosome test is a harder/different
  target than v2's chr1-only — a *more* trustworthy estimate, not a worse model.
- **Specialist standouts, exactly where predicted:** PUM1 SpliceBERT **0.840** vs ~0.68–0.70 for the
  others; LIN28B **0.863** vs ~0.74 — the intron-heavy proteins where a splice-aware model shines.
- **Tiny-model limit:** RNABERT ≈ chance on Pumilio (PUM1 0.529, PUM2 0.576) — 0.5M params can't hold
  that motif; the reason we insist every model gets its best shot.

**The splice ablation — the result we built this for (`ablation_splice.tsv`):**

For the four ablation proteins we retrained on splice-distance-matched negatives and compared AUROC.
`drop = primary − splice`; a large positive drop would mean the AUROC was substantially
splice-proximity.

> **Finding: there is no meaningful splice-proximity inflation.** Every drop is tiny (|drop| ≤ 0.046,
> most ≤ 0.02) and many are *negative* (the model did as well or better on splice-matched negatives).
> The splice-driven proteins hold up — e.g. SpliceBERT on PUM1 is **0.854** (splice-matched) vs 0.840
> (primary). RBFOX2 (the clean-motif control) shows the same ~0 drop, as expected.

**What this means:** the `GGUAAG`/splice-donor enrichment EDA flagged (V.9) is a real *correlate* of
where these proteins bind, but the models — SpliceBERT included — are **not** winning by detecting
splice sites; when we remove that shortcut, performance barely moves. So SpliceBERT's edge is genuine
RBP-binding signal and the whole comparison is trustworthy. (Caveat: ±0.01–0.02 differences are within
single-run noise, so read the *magnitude* — near zero — not the sign of any one cell.) This is exactly
why we **measured** the confound instead of **removing** it (Part XII #5): the measurement is what let
us clear the doubt rather than quietly deleting real signal.

## XI.3 ClinVar variant-effect — the final result

The variant-effect analysis scored **6,055 Pathogenic/Benign SNVs at real binding sites** with all
four tuned models (the max-shift disruption `delta`). AUROC at separating Pathogenic from Benign:

| Stratum | n_path | n_benign | CNN | RNA-FM (LoRA) | RNABERT | **SpliceBERT** |
|---|---|---|---|---|---|---|
| all @ sites | 2163 | 3892 | 0.557 | 0.564 | 0.542 | **0.740** |
| **noncoding @ sites** | 430 | 1704 | 0.584 | 0.583 | 0.526 | **0.837** |
| coding @ sites | 691 | 2170 | 0.530 | 0.545 | 0.546 | 0.576 |

- **SpliceBERT reaches 0.837 on noncoding variants at real sites** — with *zero disease training* —
  even edging past v2's 0.825. The other three models sit near chance (~0.52–0.58) there.
- **Coding is modest (SpliceBERT 0.576, others ~chance)** — exactly the honest, domain-scoped result:
  a sequence-level RNA-binding model has jurisdiction over noncoding variants that work by disrupting
  binding, but little to say about coding variants that change the protein itself.
- **This closes the arc:** SpliceBERT learns genuine binding (XI.2 ablation) → that binding signal
  transfers to flagging disease-causing noncoding variants, *because* those variants act by disrupting
  the binding it learned.

*(The loader warnings in the ClinVar log — `UNEXPECTED lm_head/ss_head`, `MISSING pooler.dense` — are
benign: `multimolecule` is noting the pretrained checkpoint carries a masked-LM/structure head and a
pooler we deliberately don't use, since we take the base encoder + masked-mean-pool + our own head.)*

## XI.4 The complete picture

`rbp-binding` is a finished, self-consistent result: a splicing-aware specialist beats a 100M
generalist on binding, an ablation shows that edge is real biology rather than a splice-site shortcut,
and the learned binding transfers to disease-variant scoring exactly where it should. Every number
here is this project's own, produced by the phase-gated cluster run that passed validation with
`vs_v2 = match` on all 16 proteins.

---

# Part XII — The decision log (every fork, the choice, the road not taken)

1. **Rebuild in a new folder vs patch v2.** Chose a clean, self-contained `rbp-binding/`. Why: v2's
   split and negatives were baked in; a from-scratch rebuild with the correct split and an explicit
   validation gate is easier to trust than a patched one.
2. **Chromosome split vs random split.** Chromosome. Why: random leaks via overlapping/repeat windows
   and inflates AUROC. (Part IV.)
3. **Teammate's exact split vs a leaner one.** Teammate's (`test={1,2,20}`, `val={19,16,13,18}`). Why:
   comparability with the teammate outweighs the extra training data a leaner split would give.
4. **Full re-extract on the cluster vs reuse v2 windows.** Full re-extract. Why: a genuinely correct,
   fresh pipeline; the reproducibility hash then *proves* it matches v2.
5. **Negatives — measure the splice confound (Option A) vs remove it (C) vs ignore it (B).** Chose A:
   keep teammate-comparable negatives as the frozen primary set **and** add a splice-matched ablation
   on 4 proteins. Why: the confounded variable (splice proximity) is *real biology*; removing it (C)
   risks deleting real signal and breaks comparability, and ignoring it (B) leaves a known blind spot.
   Measuring with an ablation answers the question without either cost. (This was the key call.)
6. **Which proteins get the ablation.** PUM1, LIN28B, U2AF1 (where `GGUAAG` dominated) + RBFOX2 (clean
   `UGCAUG` control). Why: scope the extra compute to where the confound is, and include a control so
   the contrast is interpretable.
7. **All models tuned to peak (incl. full-fine-tune RNABERT).** Why: v2's frozen RNABERT made the
   comparison unfair; a fair ranking requires every model to get its best shot.
8. **LoRA for RNA-FM vs full fine-tune.** LoRA. Why: 100M params × 16 proteins × few-thousand examples
   would overfit and cost enormously; LoRA trains <1 % of params and preserves pretraining.
9. **A hard validation gate before training.** Added. Why: catch data errors before spending GPU-days;
   make "we validated" structural, not a promise.
10. **Manifest-driven arrays vs hard-coded `sed` indexing.** Manifests from config. Why: one source of
    truth, scales when the panel changes, no brittle arithmetic in the sbatch files.
11. **8-hour wall-time on every job.** Your standing rule. Why: comfortably above need, so nothing is
    killed mid-run; the scheduler reclaims early-finishers anyway.
12. **clean_start dry-run by default.** Why: a destructive `rm -rf` with an unverified path is how
    accidents happen; printing-then-`--force` makes a wrong path harmless.

---

# Appendix A — Glossary

- **RBP** — RNA-binding protein. **eCLIP** — the experiment that finds where it binds. **Peak** — a
  called binding interval. **Reproducible peak** — confirmed across replicates (IDR).
- **Positive/negative** — bound / matched-unbound window. **Matched negative** — same region + GC +
  ≥500 nt from peaks.
- **Motif** — the short sequence an RBP prefers (e.g. RBFOX2 `UGCAUG`). **Splice donor/acceptor** — the
  5′/3′ ends of an intron; `GGUAAG` is the donor consensus.
- **Region types** — 5UTR, 3UTR, CDS, exon, ncRNA_exon, intron.
- **Split / leakage** — train/val/test partition / when test info sneaks into training.
- **AUROC / AUPRC** — ranking-quality metrics (0.5 = chance).
- **One-hot** — 4×L binary encoding of a sequence. **Tokenizer / embedding** — how a language model
  turns letters into vectors. **Masked mean pooling** — averaging token vectors to one window vector.
- **Fine-tune / frozen** — update a pretrained model's weights / leave them fixed. **LoRA** — train
  tiny low-rank adapters instead of all weights.
- **Slurm / partition / QOS / job array / dependency** — the scheduler / node pool / usage cap /
  batched tasks / "run B after A".
- **Delta score (ClinVar)** — `max_shift |p(ref) − p(alt)|`, how much a variant moves binding.

# Appendix B — The commands, start to finish

```bash
# LOCAL (once): reproducibility reference from v2 data
python cluster/make_reference.py ../rbp-v2/data/processed

# CLUSTER
bash cluster/clean_start.sh              # review, then:
bash cluster/clean_start.sh --force
bash cluster/setup_env.sh                # on a compute node
bash cluster/download_data.sh
bash cluster/submit_data.sh              # Phase A: prep -> validate  (STOP)
#   read logs/validate_<id>.out  -> must say VALIDATION PASSED
bash cluster/submit_models.sh            # Phase B: cnn + lm -> aggregate -> clinvar
```

# Appendix C — The exact hyperparameters

| Where | Setting | Value |
|---|---|---|
| Windows | length / half / GC band / min-dist / seed | 101 / 50 / ±0.05 / 500 nt / 7 |
| Split | test / val | chr1,chr2,chr20 / chr19,chr16,chr13,chr18 |
| Splice buckets | edges (nt) | 50, 150, 500, 1500 |
| CNN | grid (lr,dropout) | (1e-3,0.5),(1e-3,0.3),(5e-4,0.5),(5e-4,0.3) |
| CNN | epochs / patience / batch / wd | 40 / 6 / 64 / 1e-4 |
| LMs | epochs / patience / batch / seed | 12 / 4 / 32 / 7 |
| RNA-FM | LoRA r / alpha / dropout / targets | 8 / 16 / 0.05 / query,value |
| RNA-FM | lr adapters / head | 1e-4 / 1e-3 |
| RNABERT, SpliceBERT | lr encoder / head | 3e-5 / 1e-3 |
| Head | shape | Linear(h→128)→ReLU→Dropout(0.3)→Linear(128→1) |
| ClinVar | shifts / margin | -40,-20,0,20,40 / 25 |
| Cluster | arrays (prep/cnn/lm) | 0-19%8 / 0-19%8 / 0-7%4 |
| Cluster | time limit / gpu QOS | 08:00:00 / 8 submit, 4 run |
| Env | torch build | cu118 (CUDA 11.8) |

---

# Appendix D — Compute environment (the actual run, NEU Explorer)

The hardware and scheduler limits the pipeline actually ran on, captured from the cluster after the run.

**Hardware our jobs landed on:**
- **GPU — NVIDIA Tesla V100-SXM2-32GB** (32 GB VRAM, driver 545.23.08, CUDA compute capability 7.0).
  The LM fine-tunes and ClinVar ran on nodes `d1017`/`d1019`, each a **4× V100-SXM2** node with 28
  CPU cores and 191 GB RAM. (Our `gpu` QOS scheduled us onto V100-SXM2; the partition also carries
  V100-PCIE, T4, A100, and H200 nodes.)
- **CPU — Intel Xeon Gold 6132 @ 2.60 GHz**, 28 cores/node (2 sockets × 14 cores, 1 thread/core, no
  hyperthreading). The `short`-partition CPU nodes are the same 28-core Xeon class (191–256 GB RAM).

**Scheduler limits (and how they shaped the design):**
- **`gpu` QOS: 8 submitted / 4 running / ≤ 4 GPUs per user** → the LM sweep is packed into **8 array
  tasks throttled at `%4`**, hitting the cap exactly (a naive 60-element array is rejected).
- **`short` QOS: up to 50 running, effectively uncapped submit** → the 20-task CPU arrays (prep, CNN)
  run freely.
- **Wall-time limits:** `gpu` 8 h, `short` 2 days. We set **8 h on every job** — far above need, so
  nothing is killed mid-run.

**Per-job resources actually used (from `sacct`):**

| Job | Partition | CPUs | Mem | GPU | Tasks | Elapsed / task |
|---|---|---|---|---|---|---|
| `rbp-prep` | short | 4 | 16 GB | — | 20 | 0.4–4.3 min |
| `rbp-validate` | short | 4 | 16 GB | — | 1 | 0:20 |
| `rbp-cnn` | short | 4 | 8 GB | — | 20 | 0.4–1.6 min |
| `rbp-lm` | gpu | 8 | 24 GB | 1× V100 | 8 | 8–34 min |
| `rbp-agg` | short | 2 | 8 GB | — | 1 | 0:05 |
| `rbp-clinvar` | gpu | 8 | 24 GB | 1× V100 | 1 | ~5 min |

The LM sweep is the long pole (longest task ~34 min; each task runs ~8 fine-tunes, RNA-FM slowest),
but with 4 GPUs in parallel the whole of Phase B finished well under an hour — comfortably inside the
8 h limits. Raw captures: `docs/cluster_config_summary.txt` and `docs/hw_probe.out`.

---

*End of the masterclass. If any single line here isn't clear, that's a bug in the document — ask and
we'll expand it.*
