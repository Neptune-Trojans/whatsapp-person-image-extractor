# WhatsApp Face Finder

**Find all photos of a specific person inside large WhatsApp group chat exports — automatically, using face recognition.**

WhatsApp Face Finder is a small, local Python tool that scans the images exported from a WhatsApp chat, detects every face, compares them against a few reference photos of one person, and copies the matching images into a separate folder — with a CSV report of what it found. Everything runs **offline on your own machine**; no photos are uploaded anywhere.

Useful when you need to pull one person's pictures out of a group chat with thousands of images — for an event album, a gift, a missing-person search, or just cleaning up your camera roll.

## Features

- 🔍 **Face-recognition search** over an entire chat export (powered by [InsightFace](https://github.com/deepinsight/insightface) + ONNX Runtime).
- 🖼️ **Multiple reference photos** of the target person are averaged for a more robust match.
- 🖥️ **Local web UI** — pick folders, select reference images, and watch live progress in your browser.
- ⌨️ **Command-line interface** for scripting and batch runs.
- 📄 **CSV report** (`matches.csv`) of every detected face and its similarity score.
- 🌐 **Non-English filenames supported** (e.g. Hebrew/Arabic chat names) and **100% offline / private**.

## How it works

1. Export a WhatsApp chat *with media* and unzip it (a `scripts/extract_all.py` helper is included).
2. Give the tool a few clear photos of the person you're looking for (the **reference**).
3. It detects faces in every image, computes a face embedding, and compares each to the reference using cosine similarity.
4. Images scoring above the threshold are copied into your output folder, alongside `matches.csv`.

## Installation

Requires **Python 3.10+**.

```bash
git clone https://github.com/Neptune-Trojans/whatsapp-person-image-extractor.git
cd whatsapp-person-image-extractor

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt  # installs the package (editable) + dependencies
```

## Usage

### Option A — Web UI (easiest)

```bash
python scripts/serve.py          # then open http://127.0.0.1:5000
```

In the browser: choose the **media folder** to scan, the **output folder**, click **Add images…** to select reference photos of the person, set a threshold, and hit **Run**. Progress and the match count update live.

### Option B — Command line

```bash
python scripts/find_person.py \
  --reference data/reference \
  --media "data/WhatsApp Chat - My Group" \
  --out output/john \
  --threshold 0.35
```

| Flag | Description |
| --- | --- |
| `-r`, `--reference` | Folder of reference image(s) of the target person. |
| `-m`, `--media` | Folder of images to scan (searched recursively). |
| `-o`, `--out` | Folder to copy matches into; `matches.csv` is written here. |
| `-t`, `--threshold` | Cosine-similarity match cutoff (default `0.35`). |

## Tips for good results

- Use **2–5 clear, front-facing reference photos** of the same person; more angles = better recall.
- Keep the reference set to **one person only** — mixing people dilutes the match.
- **Tune the threshold:** lower (e.g. `0.30`) finds more but risks false positives; higher (e.g. `0.45`) is stricter.

## Output

- Matching images are copied into the output folder (existing files of the same name are overwritten).
- `matches.csv` lists every detected face with its bounding box (`x_min, x_max, y_min, y_max`) and similarity score.

## Tech stack

Python · [InsightFace](https://github.com/deepinsight/insightface) (`buffalo_l` ArcFace model) · ONNX Runtime · OpenCV · NumPy · Flask.

## Privacy

This tool runs entirely on your computer. Face detection, recognition, and file copying happen locally — no images, embeddings, or data are sent over the network. Only use it on photos you have the right to process.
