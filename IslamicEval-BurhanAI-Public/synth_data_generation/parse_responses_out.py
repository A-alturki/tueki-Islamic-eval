import json
import sys
from pathlib import Path

"""
Parse Responses API batch result JSONL and extract the json_schema output
as a proper JSON object with unescaped Arabic characters.

Additionally, write a TXT file with one segment per line (flattened),
removing internal newlines in each segment.

Usage:
  python3 parse_responses_out.py input.jsonl [output.jsonl] [output.txt]

Writes lines like:
  {"custom_id": "line-0001", "segments": ["...", "..."]}
If a line fails to parse, it is skipped with a warning to stderr.
"""

def extract_segments(obj: dict):
    try:
        # Responses API shape: response.body.output[0].content[0].text contains a JSON string
        body = obj["response"]["body"]
        output = body.get("output", [])
        if not output:
            return None
        item = output[0]
        content = item.get("content", [])
        if not content:
            return None
        text_json_str = content[0]["text"]
        data = json.loads(text_json_str)
        segments = data.get("segments")
        if isinstance(segments, list) and all(isinstance(s, str) for s in segments):
            return segments
    except Exception:
        return None
    return None


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 parse_responses_out.py input.jsonl [output.jsonl] [output.txt]", file=sys.stderr)
        sys.exit(2)

    in_path = Path(sys.argv[1])
    if len(sys.argv) > 2:
        out_path = Path(sys.argv[2])
    else:
        out_path = in_path.with_name(in_path.stem + "_parsed.jsonl")

    # Determine TXT output path
    if len(sys.argv) > 3:
        txt_path = Path(sys.argv[3])
    else:
        # Default txt next to the JSONL output
        if out_path.suffix:
            txt_path = out_path.with_suffix("")
            txt_path = txt_path.with_name(txt_path.name + "_segments.txt")
        else:
            txt_path = out_path.with_name(out_path.name + "_segments.txt")

    count_in = 0
    count_ok = 0
    segments_written = 0
    with (
        in_path.open("r", encoding="utf-8") as fin,
        out_path.open("w", encoding="utf-8", newline="\n") as fout,
        txt_path.open("w", encoding="utf-8", newline="\n") as ftxt,
    ):
        for line in fin:
            line = line.strip()
            if not line:
                continue
            count_in += 1
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                print(f"WARN: Skipping invalid JSON line {count_in}", file=sys.stderr)
                continue
            cid = obj.get("custom_id") or obj.get("customId")
            segments = extract_segments(obj)
            if segments is None:
                print(f"WARN: No segments found for line {count_in} (custom_id={cid})", file=sys.stderr)
                continue
            out_obj = {"custom_id": cid, "segments": segments}
            # ensure_ascii=False to keep Arabic chars, not \u escapes
            fout.write(json.dumps(out_obj, ensure_ascii=False) + "\n")
            count_ok += 1
            # Write each segment to TXT as a single line
            for seg in segments:
                # Flatten internal newlines and trim
                flat = " ".join(seg.splitlines()).strip()
                ftxt.write(flat + "\n")
                segments_written += 1

    print(f"Parsed {count_ok}/{count_in} lines to {out_path}")
    print(f"Wrote {segments_written} segment lines to {txt_path}")


if __name__ == "__main__":
    main()
