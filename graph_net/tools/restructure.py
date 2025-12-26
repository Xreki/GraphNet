import argparse
import shutil
import os
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List


def collect_model_paths(source_dir: Path) -> Dict[str, List[str]]:
    if not source_dir.is_dir():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")

    base_model_path2extended: Dict[str, List(str)] = defaultdict(list)
    all_model_filepaths = sorted(source_dir.rglob("model.py"))
    for filepath in all_model_filepaths:
        extended_model_path = filepath.parent
        base_model_path = extended_model_path.parent
        base_model_path2extended[str(base_model_path)].append(str(extended_model_path))
    return base_model_path2extended


def main(args):
    source_dir = args.source_dir
    target_dir = args.target_dir
    base_model_path2extended = collect_model_paths(source_dir)
    max_num_extended_models = max(
        [
            len(extended_model_paths)
            for _, extended_model_paths in base_model_path2extended.items()
        ]
    )
    cluster2model_name_map = {}
    for i in range(max_num_extended_models):
        cluster_dir = target_dir / f"{args.cluster_name}_{i}"
        base_model_path2extended_name = {}
        for base_model_path, extended_model_paths in base_model_path2extended.items():
            if len(extended_model_paths) > i:
                extended_model_path = Path(extended_model_paths[i])
                rel_model_path = Path(base_model_path).relative_to(source_dir)
                base_model_path2extended_name[
                    str(rel_model_path)
                ] = extended_model_path.name
                output_path = Path(cluster_dir) / rel_model_path
                print(f"- Copy to {output_path}")
                shutil.copytree(extended_model_path, output_path)
        cluster2model_name_map[str(cluster_dir)] = base_model_path2extended_name

    filepath = os.path.join(str(target_dir), "dimension_generalized_map.json")
    print(f"Write to {filepath}")
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(cluster2model_name_map, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-dir",
        type=Path,
        required=True,
        default=None,
        help="Root directory containing sample models.",
    )
    parser.add_argument(
        "--target-dir",
        type=Path,
        required=True,
        help="Directory where duplicate models will be moved to.",
    )
    parser.add_argument(
        "--cluster-name",
        type=str,
        required=True,
        default="cluster",
        help="Directory where duplicate models will be moved to.",
    )
    args = parser.parse_args()
    main(args)
