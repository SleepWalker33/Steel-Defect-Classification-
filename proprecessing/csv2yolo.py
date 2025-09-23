#!/usr/bin/env python
"""Utilities for converting CSV annotations into YOLO label files."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from matplotlib.patches import Rectangle
from PIL import Image, ImageDraw

# ---------------------------------------------------------------------------
# Configuration (replace the placeholder paths with project-specific values)
# ---------------------------------------------------------------------------
SPLIT_DIR = Path("path/to/csv_splits")
TRAIN_SPLIT_PATH = SPLIT_DIR / "train_split.csv"
VAL_SPLIT_PATH = SPLIT_DIR / "val_split.csv"
TEST_SPLIT_PATH = SPLIT_DIR / "test_split.csv"

YOLO_OUTPUT_DIRS = {
    "train": Path("yolo_labels/train"),
    "val": Path("yolo_labels/val"),
    "test": Path("yolo_labels/test"),
}

EXAMPLE_IMAGE_DIR = Path("path/to/images/train")
EXAMPLE_LABEL_DIR = Path("path/to/labels/train")
EXAMPLE_IMAGE_NAME = "example_image.jpg"
DATA_YAML_PATH = Path("path/to/data.yaml")

IMAGE_WIDTH = 1600
IMAGE_HEIGHT = 256
NUM_CLASSES = 4


# ---------------------------------------------------------------------------
# Data preparation helpers
# ---------------------------------------------------------------------------

def encode_to_bbox(encoded_pixels: str | float) -> str | float:
    """Convert run-length encoding into bounding boxes with one-pixel padding."""
    if pd.isna(encoded_pixels):
        return np.nan

    encoded_values = list(map(int, str(encoded_pixels).split()))
    bboxes: List[str] = []

    for idx in range(0, len(encoded_values), 2):
        if idx + 1 >= len(encoded_values):
            print(f"Skipping incomplete encoding pair: {encoded_values[idx:]}")
            break

        pixel_pos = encoded_values[idx]
        length = encoded_values[idx + 1]

        col = (pixel_pos - 1) // IMAGE_HEIGHT
        row = (pixel_pos - 1) % IMAGE_HEIGHT

        row_end = (pixel_pos + length - 2) % IMAGE_HEIGHT
        col_end = (pixel_pos + length - 2) // IMAGE_HEIGHT

        xmin = max(col - 1, 0)
        ymin = max(row - 1, 0)
        xmax = min(col_end + 1, IMAGE_WIDTH - 1)
        ymax = min(row_end + 1, IMAGE_HEIGHT - 1)

        bboxes.append(f"{xmin},{ymin},{xmax},{ymax}")

    return " ".join(bboxes)


def convert_table(df: pd.DataFrame) -> pd.DataFrame:
    """Pivot per-class bounding boxes into dedicated columns."""
    converted_rows: List[dict[str, str]] = []
    grouped = df.groupby(df["ImageId_ClassId"].str.split("_").str[0])

    for image_id, group in grouped:
        image_key = image_id.split("_")[0]
        group_size = len(group)

        if group_size == NUM_CLASSES:
            bboxes = [
                group.iloc[idx]["bboxes"] if pd.notna(group.iloc[idx]["bboxes"]) else ""
                for idx in range(NUM_CLASSES)
            ]
            converted_rows.append(
                {
                    "ImageId_ClassId": image_key,
                    "1_labels": "1" if bboxes[0] else "",
                    "1_boxes": bboxes[0],
                    "2_labels": "1" if bboxes[1] else "",
                    "2_boxes": bboxes[1],
                    "3_labels": "1" if bboxes[2] else "",
                    "3_boxes": bboxes[2],
                    "4_labels": "1" if bboxes[3] else "",
                    "4_boxes": bboxes[3],
                }
            )
        else:
            for idx in range(group_size):
                bbox_value = group.iloc[idx]["bboxes"]
                bbox_value = bbox_value if pd.notna(bbox_value) else ""
                converted_rows.append(
                    {
                        "ImageId_ClassId": image_key,
                        f"{idx + 1}_labels": "1" if bbox_value else "",
                        f"{idx + 1}_boxes": bbox_value,
                    }
                )

    columns = [
        "ImageId_ClassId",
        "1_labels",
        "1_boxes",
        "2_labels",
        "2_boxes",
        "3_labels",
        "3_boxes",
        "4_labels",
        "4_boxes",
    ]

    converted_df = pd.DataFrame(converted_rows, columns=columns)
    converted_df = converted_df.fillna("")
    return converted_df


def safe_convert_to_int(value: float | str, default: int = 0) -> int:
    """Attempt to convert a value to int, falling back to *default* on failure."""
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def draw_bboxes_on_image(converted_df: pd.DataFrame, image_id: str, image_folder: Path | str) -> None:
    """Render bounding boxes stored in *_boxes columns on top of the requested image."""
    row = converted_df[converted_df["ImageId_ClassId"] == image_id]
    if row.empty:
        print(f"Image ID {image_id} not found in the DataFrame.")
        return

    image_path = Path(image_folder) / image_id
    if not image_path.exists():
        print(f"Image file not found: {image_path}")
        return

    try:
        image = Image.open(image_path).convert("RGB")
    except OSError as exc:
        print(f"Unable to open image {image_path}: {exc}")
        return

    draw = ImageDraw.Draw(image)

    for column in converted_df.columns:
        if column.endswith("_boxes"):
            bbox_str = row[column].values[0] if column in row else ""
            if not bbox_str:
                continue

            for bbox in bbox_str.split(" "):
                coords = bbox.split(",")
                if len(coords) != 4:
                    print(f"Invalid bbox format for {column}: {bbox}")
                    continue
                xmin, ymin, xmax, ymax = [safe_convert_to_int(coord, default=0) for coord in coords]
                draw.rectangle([xmin, ymin, xmax, ymax], outline="red", width=1)

    plt.figure(figsize=(8, 8))
    plt.imshow(image)
    plt.axis("off")
    plt.title(image_id)
    plt.show()


# ---------------------------------------------------------------------------
# Bounding-box processing
# ---------------------------------------------------------------------------

def merge_boxes(boxes_str: str, threshold: int = 5) -> List[List[int]]:
    """Merge adjacent bounding boxes when they are closer than *threshold* pixels."""
    if not boxes_str or not isinstance(boxes_str, str):
        return []

    boxes: List[List[int]] = []
    for bbox in boxes_str.split(" "):
        coords = list(map(int, bbox.split(",")))
        if len(coords) == 4:
            boxes.append(coords)

    if not boxes:
        return []

    boxes.sort(key=lambda box: (box[0], box[1]))
    merged_boxes: List[List[int]] = []
    current_box = boxes[0]

    for next_box in boxes[1:]:
        xmin, ymin, xmax, ymax = current_box
        nxmin, nymin, nxmax, nymax = next_box

        if abs(xmax - nxmin) <= threshold or abs(ymax - nymin) <= threshold:
            current_box = [
                min(xmin, nxmin),
                min(ymin, nymin),
                max(xmax, nxmax),
                max(ymax, nymax),
            ]
        else:
            merged_boxes.append(current_box)
            current_box = next_box

    merged_boxes.append(current_box)
    return merged_boxes


def merge_boxes_for_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Apply box merging to every *_boxes column, creating *_merged_boxes columns."""
    result = df.copy()
    for class_idx in range(1, NUM_CLASSES + 1):
        box_column = f"{class_idx}_boxes"
        merged_column = f"{class_idx}_merged_boxes"
        if box_column in result.columns:
            result[merged_column] = result[box_column].apply(merge_boxes)
    return result


def convert_boxes(boxes: Iterable[Iterable[int]]) -> List[List[float]]:
    """Convert [xmin, ymin, xmax, ymax] boxes to [x_center, y_center, width, height]."""
    converted_boxes: List[List[float]] = []
    for box in boxes or []:
        x_min, y_min, x_max, y_max = box
        x_center = (x_min + x_max) / 2
        y_center = (y_min + y_max) / 2
        width = x_max - x_min
        height = y_max - y_min
        converted_boxes.append([x_center, y_center, width, height])
    return converted_boxes


def prepare_result_dataframe(split_df: pd.DataFrame) -> pd.DataFrame:
    """Produce the final dataframe containing merged and center-based boxes."""
    merged_df = merge_boxes_for_dataframe(split_df)

    for class_idx in range(1, NUM_CLASSES + 1):
        column = f"{class_idx}_merged_boxes"
        if column in merged_df.columns:
            merged_df[column] = merged_df[column].apply(convert_boxes)

    box_columns = [f"{idx}_boxes" for idx in range(1, NUM_CLASSES + 1)]
    existing_columns = [column for column in box_columns if column in merged_df.columns]
    if existing_columns:
        merged_df = merged_df.drop(columns=existing_columns)

    return merged_df


def export_yolo_labels(df: pd.DataFrame, output_dir: Path, image_width: int, image_height: int) -> None:
    """Write YOLO label files derived from the dataframe contents."""
    output_dir.mkdir(parents=True, exist_ok=True)

    for _, row in df.iterrows():
        image_id = str(row["ImageId_ClassId"]).split(".")[0]
        file_path = output_dir / f"{image_id}.txt"

        boxes_info: List[str] = []
        for class_idx in range(1, NUM_CLASSES + 1):
            merged_column = f"{class_idx}_merged_boxes"
            if merged_column not in row or not row[merged_column]:
                continue

            for box in row[merged_column]:
                x_center, y_center, width, height = box
                normalized = (
                    x_center / image_width,
                    y_center / image_height,
                    width / image_width,
                    height / image_height,
                )
                boxes_info.append(
                    f"{class_idx - 1} " + " ".join(f"{value:.6f}" for value in normalized)
                )

        file_path.write_text("\n".join(boxes_info), encoding="utf-8")


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def visualize_example(image_dir: Path, label_dir: Path, image_name: str, yaml_path: Path) -> None:
    """Display YOLO annotations on top of an example image."""
    image_path = image_dir / image_name
    label_path = label_dir / f"{Path(image_name).stem}.txt"

    if not image_path.exists():
        print(f"Image file not found: {image_path}")
        return
    if not label_path.exists():
        print(f"Label file not found: {label_path}")
        return
    if not yaml_path.exists():
        print(f"YAML file not found: {yaml_path}")
        return

    with yaml_path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)
    class_names = data.get("names", [])

    image = plt.imread(image_path)
    if image is None:
        print(f"Unable to load image data from {image_path}")
        return

    img_h, img_w = image.shape[:2]

    with label_path.open("r", encoding="utf-8") as file:
        lines = file.readlines()

    fig, ax = plt.subplots(1, figsize=(10, 6))
    ax.imshow(image)

    for line in lines:
        values = line.strip().split()
        if len(values) != 5:
            continue

        class_id, x_center, y_center, width, height = map(float, values)
        class_id = int(class_id)

        x_center *= img_w
        y_center *= img_h
        width *= img_w
        height *= img_h

        x1 = int(x_center - width / 2)
        y1 = int(y_center - height / 2)
        x2 = int(x_center + width / 2)
        y2 = int(y_center + height / 2)

        class_name = class_names[class_id] if class_id < len(class_names) else f"class_{class_id}"

        rect = Rectangle((x1, y1), width, height, linewidth=2, edgecolor="red", facecolor="none")
        ax.add_patch(rect)
        ax.text(
            x1,
            y1 - 5,
            class_name,
            fontsize=10,
            color="white",
            bbox=dict(facecolor="red", alpha=0.6, edgecolor="none", boxstyle="round,pad=0.3"),
        )

    ax.axis("off")
    plt.show()


# ---------------------------------------------------------------------------
# Main workflow
# ---------------------------------------------------------------------------

def load_and_prepare_split(csv_path: Path) -> pd.DataFrame:
    """Load a CSV split and attach bounding boxes derived from encoded pixels."""
    split_df = pd.read_csv(csv_path)
    split_df = split_df.copy()
    split_df["bboxes"] = split_df["EncodedPixels"].apply(encode_to_bbox)
    return convert_table(split_df)


def main() -> None:
    required_paths = [TRAIN_SPLIT_PATH, VAL_SPLIT_PATH, TEST_SPLIT_PATH]
    missing = [str(path) for path in required_paths if not path.exists()]
    if missing:
        print("Update the SPLIT_DIR constants before running this script.")
        print("Missing paths:")
        for path in missing:
            print(f"  - {path}")
        return

    train_split = load_and_prepare_split(TRAIN_SPLIT_PATH)
    val_split = load_and_prepare_split(VAL_SPLIT_PATH)
    test_split = load_and_prepare_split(TEST_SPLIT_PATH)

    train_result = prepare_result_dataframe(train_split)
    val_result = prepare_result_dataframe(val_split)
    test_result = prepare_result_dataframe(test_split)

    export_yolo_labels(train_result, YOLO_OUTPUT_DIRS["train"], IMAGE_WIDTH, IMAGE_HEIGHT)
    export_yolo_labels(val_result, YOLO_OUTPUT_DIRS["val"], IMAGE_WIDTH, IMAGE_HEIGHT)
    export_yolo_labels(test_result, YOLO_OUTPUT_DIRS["test"], IMAGE_WIDTH, IMAGE_HEIGHT)

    print("YOLO label export completed.")

    if (
        EXAMPLE_IMAGE_DIR.exists()
        and EXAMPLE_LABEL_DIR.exists()
        and DATA_YAML_PATH.exists()
    ):
        visualize_example(EXAMPLE_IMAGE_DIR, EXAMPLE_LABEL_DIR, EXAMPLE_IMAGE_NAME, DATA_YAML_PATH)
    else:
        print("Update the visualization paths to preview annotations.")


if __name__ == "__main__":
    main()
