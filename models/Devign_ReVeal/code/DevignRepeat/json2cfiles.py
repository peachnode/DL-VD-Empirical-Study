#!/usr/bin/env python3
import sys, json, os, re

def main():
    if len(sys.argv) != 3:
        print("Usage: python3 json2cfiles.py input.json /path/to/output_dir")
        sys.exit(1)

    input_path, out_dir = sys.argv[1], sys.argv[2]
    os.makedirs(out_dir, exist_ok=True)

    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    for i, entry in enumerate(data, start=1):
        label = entry.get('target', entry.get('label', 'unknown'))
        filename = f"{i}_{label}.c"
        out_path = os.path.join(out_dir, filename)

        func_text = entry.get('func', '')
        header = f"/* project: {entry.get('project','')}\n   label: {label}\n*/\n\n"
        with open(out_path, 'w', encoding='utf-8') as wf:
            wf.write(header + func_text)

        print("Wrote:", out_path)

if __name__ == "__main__":
    main()
