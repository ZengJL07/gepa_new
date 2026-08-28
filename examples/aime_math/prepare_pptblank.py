"""One-shot preprocessing for the PPT excessive-whitespace dataset.

Reads the annotated source dataset (``dataset_v4`` produced by the ai-ppt-dataset
tooling) and emits everything the optimization run needs, so no later step ever
touches the source tree:

* ``data/pptblank.json``      — pre-split records ``{"train": [...], "val": [...], "test": [...]}``
  in the AIME key convention, loaded by ``datasets.SPECS["pptblank"]``.
* ``data/pptblank_gold.json`` — ``{image_id: gold}`` sidecar for post-hoc F1.
* ``data/pptblank/images/``            — slides re-encoded to JPEG, ASCII names.
* ``data/pptblank/images_annotated/``  — same slides with annotation boxes drawn,
  for the reflection VLM.

Three source facts drive the design:

1. **Labels come from box severity, not from a class field.** A slide with any
   ``severe`` box is a positive; a slide with no boxes at all is a negative; a
   slide whose boxes are *all* ``mild`` is deliberately ambiguous — the labelling
   spec says mild whitespace "can be considered a problem or not", so either
   answer is correct. Those become ``gold="either"`` and are dropped from
   train/val (they would score 1.0 unconditionally, contributing no optimization
   signal while eating minibatch budget) but kept in test for reporting.

2. **Negatives have no annotated image.** ``images_annotated/`` in the source only
   covers annotated slides, so a reflection prompt built from it would show the
   VLM positives only, teaching it nothing about what "clean" looks like. We copy
   the plain image into the annotated directory for negatives and say so
   explicitly in the reflection text.

3. **Splits are by pptx, not by slide.** Slides from one deck share a template and
   style, so a slide-level split leaks. Two decks per style go to train, two to
   val, the remaining four per style to test — the assignment below was picked by
   exhaustive search over all C(8,2)*C(6,2) per-style options, minimizing the
   deviation of each split's positive share from the global 0.465 while keeping
   the minimum per-split positive count high.

Usage::

    python -m examples.aime_math.prepare_pptblank
    PPTBLANK_SRC=/path/to/dataset_v4 python -m examples.aime_math.prepare_pptblank
"""

import json
import os
import shutil
import sys

from PIL import Image, ImageDraw, ImageFont

# --- Paths ---------------------------------------------------------------

DEFAULT_SRC = "/mnt/c/Users/13632/WorkBuddy/2026-08-17-21-50-57/ai-ppt-dataset/dataset_v4"
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
OUT_ROOT = os.path.join(DATA_DIR, "pptblank")

# --- Preprocessing knobs -------------------------------------------------

# Slides are re-encoded at this size. Excessive-whitespace judgement is a
# layout-level task, so full 1920x1080 detail is not needed; 960x540 JPEG q85
# cuts the per-call payload from ~119 KB to ~48 KB. Raise to (1280, 720) if
# fine-grained gaps turn out to matter.
TARGET_SIZE = (1280, 720)
JPEG_QUALITY = 85

# Style names are Chinese in the source. The reflection prompt gets the English
# form: style drives layout convention and is a generalizable axis the reflection
# LM can act on. Topic (a brand name) is deliberately NOT translated and NOT put
# in the prompt — it would only invite overfitting to specific brand names.
STYLE_EN = {
    "商务简约": "business_minimal",
    "扁平插画": "flat_illustration",
    "科技未来": "tech_futuristic",
}

# (style, ppt_id) -> split. Remaining decks fall through to test.
SPLIT_TRAIN = {
    ("商务简约", "03"), ("商务简约", "04"),
    ("扁平插画", "02"), ("扁平插画", "04"),
    ("科技未来", "01"), ("科技未来", "04"),
}
SPLIT_VAL = {
    ("商务简约", "05"), ("商务简约", "08"),
    ("扁平插画", "05"), ("扁平插画", "08"),
    ("科技未来", "02"), ("科技未来", "08"),
}

# Expected per-split counts, asserted at the end so a source-data change or a
# typo in the split tables fails loudly instead of silently reshaping the task.
EXPECTED = {
    "train": {"pos": 20, "neg": 25, "mild": 22},
    "val": {"pos": 24, "neg": 24, "mild": 14},
    "test": {"pos": 50, "neg": 59, "mild": 29},
}

BOX_COLORS = {"severe": (220, 38, 38), "mild": (234, 88, 12)}


def _font(size: int):
    """A font that can render the severity labels; falls back to PIL's default."""
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ):
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except OSError:
                pass
    return ImageFont.load_default()


def gold_label(boxes: list[dict]) -> str:
    """Map annotation boxes onto the three-state gold label.

    No boxes -> "no". Any ``severe`` box -> "yes" (this includes the 20 slides
    carrying both severe and mild boxes). All-mild -> "either", meaning both
    answers score 1.0.
    """
    severities = {b["severity"] for b in boxes}
    if not severities:
        return "no"
    if "severe" in severities:
        return "yes"
    return "either"


def split_of(style: str, ppt_id: str) -> str:
    if (style, ppt_id) in SPLIT_TRAIN:
        return "train"
    if (style, ppt_id) in SPLIT_VAL:
        return "val"
    return "test"


def render_annotated(src_path: str, boxes: list[dict], scale_x: float, scale_y: float) -> Image.Image:
    """Draw severity-coloured boxes on a resized copy of the slide.

    Box coordinates are in the source 1920x1080 space, so they are scaled by the
    same factor as the image itself.
    """
    img = Image.open(src_path).convert("RGB").resize(TARGET_SIZE, Image.LANCZOS)
    draw = ImageDraw.Draw(img)
    font = _font(18)
    for box in boxes:
        x1, y1, x2, y2 = box["bbox"]
        x1, x2 = x1 * scale_x, x2 * scale_x
        y1, y2 = y1 * scale_y, y2 * scale_y
        color = BOX_COLORS.get(box["severity"], (128, 128, 128))
        draw.rectangle([x1, y1, x2, y2], outline=color, width=4)
        label = f'{box["severity"]} {box["area_ratio"] * 100:.1f}%'
        tw = draw.textlength(label, font=font)
        # Keep the label inside the canvas even for boxes flush against an edge.
        ly = max(0, y1 - 22)
        draw.rectangle([x1, ly, x1 + tw + 10, ly + 22], fill=color)
        draw.text((x1 + 5, ly + 3), label, fill="white", font=font)
    return img


def main() -> int:
    src = os.environ.get("PPTBLANK_SRC", DEFAULT_SRC)
    index_path = os.path.join(src, "metadata", "index.json")
    if not os.path.exists(index_path):
        print(f"ERROR: source index not found at {index_path}", file=sys.stderr)
        print("Set PPTBLANK_SRC to the dataset_v4 directory.", file=sys.stderr)
        return 1

    with open(index_path, encoding="utf-8") as f:
        index = json.load(f)

    img_out = os.path.join(OUT_ROOT, "images")
    ann_out = os.path.join(OUT_ROOT, "images_annotated")
    os.makedirs(img_out, exist_ok=True)
    os.makedirs(ann_out, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)

    records: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
    gold_map: dict[str, str] = {}
    counts = {s: {"pos": 0, "neg": 0, "mild": 0} for s in records}
    skipped_either = {"train": 0, "val": 0}

    for info in index["images"]:
        style_cn = info["ppt"]["style"]
        ppt_id = info["ppt"]["id"]
        style_en = STYLE_EN.get(style_cn)
        if style_en is None:
            print(f"ERROR: unmapped style {style_cn!r}; add it to STYLE_EN.", file=sys.stderr)
            return 1

        boxes = info["annotation"]["boxes"]
        gold = gold_label(boxes)
        split = split_of(style_cn, ppt_id)

        counts[split]["pos" if gold == "yes" else "neg" if gold == "no" else "mild"] += 1

        # Ambiguous slides carry no optimization signal (they score 1.0 whatever
        # the model says), so they are excluded from train/val but kept in test.
        if gold == "either" and split in ("train", "val"):
            skipped_either[split] += 1
            continue

        image_id = f"{ppt_id}_{style_en}_{info['slide_index']:03d}"
        filename = f"{image_id}.jpg"
        src_img = os.path.join(src, info["image_path"])

        plain = Image.open(src_img).convert("RGB").resize(TARGET_SIZE, Image.LANCZOS)
        plain.save(os.path.join(img_out, filename), "JPEG", quality=JPEG_QUALITY)

        ann_path = os.path.join(ann_out, filename)
        if boxes:
            scale_x = TARGET_SIZE[0] / info["width"]
            scale_y = TARGET_SIZE[1] / info["height"]
            render_annotated(src_img, boxes, scale_x, scale_y).save(
                ann_path, "JPEG", quality=JPEG_QUALITY
            )
        else:
            # Negatives get the plain image under the annotated name, so the
            # reflection prompt can show clean slides as counter-examples.
            shutil.copy2(os.path.join(img_out, filename), ann_path)

        records[split].append(
            {
                "id": image_id,
                "input": f"images/{filename}",
                "answer": gold,
                "_annotated_path": f"images_annotated/{filename}",
                "_style_en": style_en,
                "_severity": [b["severity"] for b in boxes],
                "_area_ratios": [round(b["area_ratio"], 4) for b in boxes],
                "_topic": info["ppt"]["topic"],
                "_ppt_id": ppt_id,
                "_slide_index": info["slide_index"],
                "_source_image": info["image_filename"],
            }
        )
        gold_map[image_id] = gold

    with open(os.path.join(DATA_DIR, "pptblank.json"), "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    with open(os.path.join(DATA_DIR, "pptblank_gold.json"), "w", encoding="utf-8") as f:
        json.dump(gold_map, f, ensure_ascii=False, indent=2)

    # --- Report and verify -------------------------------------------------
    print(f"source: {src}")
    print(f"images: {TARGET_SIZE[0]}x{TARGET_SIZE[1]} JPEG q{JPEG_QUALITY} -> {OUT_ROOT}")
    print()
    print(f'{"split":6s} {"records":>8s} {"pos":>5s} {"neg":>5s} {"mild":>5s}  note')
    ok = True
    for split in ("train", "val", "test"):
        c = counts[split]
        exp = EXPECTED[split]
        note = ""
        if split in skipped_either:
            note = f"{skipped_either[split]} mild dropped (either)"
        print(
            f"{split:6s} {len(records[split]):8d} {c['pos']:5d} {c['neg']:5d} {c['mild']:5d}  {note}"
        )
        if c != exp:
            print(f"  MISMATCH expected {exp}, got {c}", file=sys.stderr)
            ok = False
    print()
    print(f"gold sidecar: {len(gold_map)} ids")

    if not ok:
        print(
            "\nERROR: split counts differ from EXPECTED. Either the source data "
            "changed or a split table is wrong — reconcile before training.",
            file=sys.stderr,
        )
        return 1

    total = sum(len(v) for v in records.values())
    print(f"OK: {total} records written (267 source slides - 36 ambiguous dropped from train/val)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
