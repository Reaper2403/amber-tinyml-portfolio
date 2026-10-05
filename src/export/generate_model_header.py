#!/usr/bin/env python3
"""Generate a C/C++ header that embeds a binary model as a byte array."""

from __future__ import annotations

import argparse
import re
from pathlib import Path


def sanitize_symbol(value: str) -> str:
    symbol = re.sub(r"[^0-9A-Za-z_]+", "_", value).strip("_")
    if not symbol:
        raise ValueError("symbol cannot be empty after sanitization")
    if symbol[0].isdigit():
        symbol = f"model_{symbol}"
    return symbol


def format_bytes(data: bytes, values_per_line: int = 12) -> str:
    lines: list[str] = []
    for offset in range(0, len(data), values_per_line):
        chunk = data[offset : offset + values_per_line]
        values = ", ".join(f"0x{byte:02x}" for byte in chunk)
        lines.append(f"    {values}")
    return ",\n".join(lines)


def build_header(model_path: Path, symbol: str, data: bytes) -> str:
    return "\n".join(
        [
            "#pragma once",
            "",
            f"// Generated from: {model_path.name}",
            f"// Size: {len(data)} bytes",
            "",
            f"const unsigned char {symbol}[] = {{",
            format_bytes(data),
            "};",
            f"const unsigned int {symbol}_len = {len(data)};",
            "",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path, help="Path to the source model file")
    parser.add_argument("output", type=Path, help="Path to the generated header")
    parser.add_argument(
        "--symbol",
        required=True,
        help="Symbol name for the embedded byte array and length variable",
    )
    args = parser.parse_args()

    model_path = args.model.resolve()
    output_path = args.output.resolve()
    symbol = sanitize_symbol(args.symbol)

    data = model_path.read_bytes()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(build_header(model_path, symbol, data))


if __name__ == "__main__":
    main()
