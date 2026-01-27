from __future__ import annotations

from gradientcoil.debug.bundle import generate_debug_bundle, parse_args


def main() -> None:
    cfg = parse_args()
    out_dir = generate_debug_bundle(cfg)
    print(f"Bundle created at: {out_dir}")


if __name__ == "__main__":
    main()
