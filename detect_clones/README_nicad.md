# NiCad Clone Detection and Post-Processing Pipeline (Java)

This README documents the **end-to-end processing workflow** used to generate
NiCad-based clone datasets for Java projects, including clone detection,
format conversion, and semantic post-processing.

The processing pipeline produces, among other intermediate artifacts, the file:

```
step2_nicad_camel-java_sim0.7_classes_fqn.jsonl
```

which is the final, enriched clone dataset used for analysis.

---

## 1. End-to-End Generation Pipeline
The processing workflow consists of the following ordered stages:

```
Java project source code
        ↓
NiCad clone detection (class level, similarity = 0.7)
        ↓
NiCad XML clone report
        ↓
Step 1: XML → JSONL conversion
        ↓
Step 2: Qualified-name enrichment
        ↓
Final clone dataset (JSONL)
```

---
## 2. Environment Setup

NiCad is executed in a dedicated Conda environment to ensure reproducibility.

### 2.1 Conda Environment Configuration

```bash
cat > environment-linux.yml <<'YAML'
name: nicad6
channels:
  - conda-forge
  - defaults
dependencies:
  - python=3.11
  - make
  - gcc_linux-64
  - gxx_linux-64
  - perl
  - wget
  - git
  - coreutils
  - grep
  - sed
  - unzip
YAML

conda env create -f environment-linux.yml
conda activate nicad6
```

---

## 3. NiCad and TXL Installation

NiCad relies on TXL for source-code parsing and transformation.

```bash
tar -xzf 18195-NiCad-6.2.tar.gz
cd NiCad-6.2

tar -xzf 20473-txl10.8b.linux64.tar.gz
./InstallTxl

txl -V
which txl
which txlc
```

TXL programs are then compiled:

```bash
make -C txl clean && make -C txl
```

> **Important note:**
> NiCad does not reliably handle symbolic links.
> Target projects must be **physically copied** into the `systems/` directory rather
> than referenced via symlinks.

---
## 4. NiCad Clone Detection

### 4.1 Configuration

NiCad is executed with the following settings:

* **Language:** Java
* **Granularity:** Class-level clones
* **Similarity threshold:** 0.7

Example command:

```bash
./nicad6 functions java systems/camel-java default-report
```

NiCad groups similar code fragments into **clone classes**, where each clone class
contains multiple **clone sources** extracted from the project.

---

## 5. Step 1: XML to JSONL Conversion

NiCad produces an XML clone report, which is converted into
**JSON Lines (JSONL)** format:

```
step1_nicad_camel-java_sim0.7_classes.jsonl
```

### 5.1 Step 1 JSONL Structure

* One JSON object per line
* Each object represents one clone class
* Each clone class contains:

  * `classid`
  * `nclones`
  * `similarity`
  * `sources[]`

Each `sources[i]` entry includes:

* `file`: Java source file path
* `range`: line range
* `nlines`: number of lines
* `code`: extracted code fragment

This file serves as the input to the post-processing stage.

---

## 5.2 Step 2: Fully Qualified Name Post-Processing

Step 2 augments the Step 1 JSONL file by attaching **fully qualified identifiers**
to each clone source, producing the final dataset:

```
step2_nicad_camel-java_sim0.7_classes_fqn.jsonl
```

This step **preserves all original clone data** and adds exactly one new field per
clone source:

```
sources[].qualified_name
```

Each qualified name is constructed using the following format:

```
<package>.<ClassName>.<methodName>(<parameters>)
```
---
## 6. Step 2 Script Usage

```bash
python step2_add_qualified_name.py \
  --in  step1_nicad_camel-java_sim0.7_classes.jsonl \
  --out step2_nicad_camel-java_sim0.7_classes_fqn.jsonl \
  --projects-root /path/to/project/root
```

The `--projects-root` argument is required to resolve relative paths such as:

```
systems/camel-java/...
```
## 7. Limitations

* Method signature extraction is heuristic and regex-based
  (no Java AST parsing)
* Complex constructs (e.g., lambdas, anonymous classes) may result in `<unknown>()`
* Class names are assumed to follow standard Java file naming conventions

---