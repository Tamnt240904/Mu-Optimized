import csv
import json

from progressive_filter import build_progressive_filtered_csv


def write_selected(path):
    fieldnames = ["correct_index", "source_order", "rank", "image_name", "image_path"]
    rows = [
        {
            "correct_index": "0",
            "source_order": "10",
            "rank": "1",
            "image_name": "a.jpg",
            "image_path": "/images/a.jpg",
        },
        {
            "correct_index": "1",
            "source_order": "11",
            "rank": "2",
            "image_name": "b.jpg",
            "image_path": "/images/b.jpg",
        },
        {
            "correct_index": "2",
            "source_order": "12",
            "rank": "3",
            "image_name": "c.jpg",
            "image_path": "/images/c.jpg",
        },
    ]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_rows(path):
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def image_result(correct_index, image_name, ig, idg, mu, success=True):
    return {
        "success": success,
        "correct_index": correct_index,
        "image_name": image_name,
        "insertion_deletion": {
            "methods": {
                "IG": {"insertion_auc": 0.1, "deletion_auc": 0.2, "insdel": ig},
                "idg-pdf": {"insertion_auc": 0.1, "deletion_auc": 0.2, "insdel": idg},
                "Mu-Optimized": {"insertion_auc": 0.1, "deletion_auc": 0.2, "insdel": mu},
            }
        },
    }


def test_build_progressive_filtered_csv_preserves_order_and_correct_index(tmp_path):
    selected_csv = tmp_path / "selected.csv"
    batch_json = tmp_path / "batch.json"
    output_csv = tmp_path / "filtered.csv"
    write_selected(selected_csv)
    batch_json.write_text(
        json.dumps(
            {
                "images": [
                    image_result(2, "c.jpg", ig=0.8, idg=0.4, mu=0.5),
                    image_result(0, "a.jpg", ig=0.2, idg=0.3, mu=0.4),
                    image_result(1, "b.jpg", ig=0.1, idg=0.6, mu=0.5),
                ]
            }
        ),
        encoding="utf-8",
    )

    kept = build_progressive_filtered_csv(
        selected_csv, batch_json, output_csv, "mu_gt_idg_insdel"
    )

    filtered = read_rows(output_csv)
    assert kept == 2
    assert [row["correct_index"] for row in filtered] == ["0", "2"]
    assert [row["rank"] for row in filtered] == ["1", "3"]
    assert list(filtered[0]) == [
        "correct_index",
        "source_order",
        "rank",
        "image_name",
        "image_path",
    ]


def test_build_progressive_filtered_csv_strict_rule_and_image_name_fallback(tmp_path):
    selected_csv = tmp_path / "selected.csv"
    batch_json = tmp_path / "batch.json"
    output_csv = tmp_path / "filtered.csv"
    write_selected(selected_csv)
    batch_json.write_text(
        json.dumps(
            [
                image_result(None, "a.jpg", ig=0.5, idg=0.3, mu=0.4),
                image_result(None, "b.jpg", ig=0.2, idg=0.3, mu=0.4),
                image_result(None, "c.jpg", ig=0.1, idg=0.2, mu=0.9, success=False),
            ]
        ),
        encoding="utf-8",
    )

    kept = build_progressive_filtered_csv(
        selected_csv, batch_json, output_csv, "mu_gt_ig_idg_insdel"
    )

    filtered = read_rows(output_csv)
    assert kept == 1
    assert [row["image_name"] for row in filtered] == ["b.jpg"]
