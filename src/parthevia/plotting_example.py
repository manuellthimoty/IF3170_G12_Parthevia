from __future__ import annotations

import argparse

from .plotting import plot_saved_result


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot satu hasil run JSON tanpa menjalankan pencarian.")
    parser.add_argument("result", help="Path JSON hasil HC.to_dict() atau SearchResult.to_dict()")
    parser.add_argument("output_dir", help="Folder tujuan gambar")
    args = parser.parse_args()
    for path in plot_saved_result(args.result, args.output_dir).values():
        print(f"Saved {path}")


if __name__ == "__main__":
    main()
