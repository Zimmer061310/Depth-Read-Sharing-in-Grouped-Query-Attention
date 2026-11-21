#!/usr/bin/env python3
"""Build a deterministic, source-only arXiv upload and separate form metadata."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import re
import subprocess
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha256(raw):
    return hashlib.sha256(raw).hexdigest()


def checked_file(root, name):
    rel = PurePosixPath(name)
    require(name and not rel.is_absolute() and ".." not in rel.parts
            and "\\" not in name and not name.startswith("~"),
            "unsafe source path: " + name)
    path = root / name
    require(path.is_file() and not path.is_symlink(), "missing/nonregular source: " + name)
    require(path.resolve().is_relative_to(root.resolve()), "source escapes project")
    return path


def dependencies(root):
    """Include literal compilation dependencies, not unused metadata or artwork."""
    found, pending = set(), ["main.tex"]
    patterns = [
        (r"\\(?:input|include)\s*\{([^{}]+)\}", [".tex"]),
        (r"\\includegraphics(?:\[[^\]]*\])?\s*\{([^{}]+)\}", [".pdf", ".png", ".jpg", ".jpeg"]),
        (r"\\(?:bibliography|addbibresource)(?:\[[^\]]*\])?\s*\{([^{}]+)\}", [".bib"]),
    ]
    while pending:
        name = pending.pop()
        if name in found:
            continue
        path = checked_file(root, name)
        found.add(name)
        if path.suffix != ".tex":
            continue
        tex = re.sub(r"(?<!\\)%[^\n]*", "", path.read_text())
        for pattern, suffixes in patterns:
            for group in re.findall(pattern, tex):
                for candidate in group.split(","):
                    candidate = candidate.strip()
                    options = [candidate] if PurePosixPath(candidate).suffix else [
                        candidate + suffix for suffix in suffixes]
                    for option in options:
                        rel = PurePosixPath(option)
                        require(not rel.is_absolute() and ".." not in rel.parts
                                and "\\" not in option, "unsafe dependency: " + option)
                    existing = [option for option in options if (root / option).is_file()]
                    require(existing, "unresolved dependency: " + candidate)
                    pending.append(existing[0])
    return sorted(found)


def abstract_text(tex):
    match = re.search(
        r"\\frontMatterTitleStyle Abstract.*?\\end\{center\}\s*(.*?)\\section\{Introduction\}",
        tex, re.S)
    require(match, "cannot locate the manuscript abstract")
    text = re.sub(r"(?<!\\)%[^\n]*", "", match.group(1))
    macros = dict(re.findall(
        r"\\newcommand\{\\([A-Za-z]+)\}\{([+-]?\d+(?:\.\d+)?)\}", tex))
    for name, value in sorted(macros.items(), key=lambda item: -len(item[0])):
        text = re.sub(r"\\" + name + r"(?:\{\})?(?![A-Za-z])",
                      lambda _: value, text)
    text = re.sub(r"\\route\{([^{}]+)\}", r"\1", text)
    text = text.replace(r"\%", "%")
    allowed = {"pm", "leq", "geq", "times", "to", "rho", "alpha", "mathrm", "text"}
    unknown = set(re.findall(r"\\([A-Za-z]+)", text)) - allowed
    require(not unknown, "unexpanded abstract macros: " + ", ".join(sorted(unknown)))
    text = re.sub(r"\s+", " ", text).strip()
    require(text.isascii(), "form abstract must use ASCII/TeX equivalents")
    require(len(text) <= 1920, "form abstract exceeds arXiv's 1920-character limit")
    return text


def zip_bytes(files):
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED,
                         compresslevel=9) as archive:
        for name, raw in sorted(files.items()):
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, raw)
    return stream.getvalue()


def build(paper, metadata, output):
    require(metadata.get("license_author_confirmed") is True,
            "manuscript license still needs author confirmation")
    require(metadata.get("license_url") == "https://arxiv.org/licenses/nonexclusive-distrib/1.0/",
            "this v1 preparation is bound to the author's selected arXiv license")
    require(metadata.get("funding_and_conflicts_author_confirmed") is True,
            "funding/conflict wording still needs author confirmation")
    tex = checked_file(paper, "main.tex").read_text()
    require(r"\author{" + ", ".join(metadata["authors"]) + "}" in tex,
            "form authors differ from manuscript authors")
    require(metadata["title"] in tex, "form title differs from manuscript title")
    abstract = abstract_text(tex)
    names = dependencies(paper)
    files = {name: checked_file(paper, name).read_bytes() for name in names}
    raw = zip_bytes(files)
    require(raw == zip_bytes(dict(reversed(list(files.items())))),
            "upload archive is not deterministic")
    with tempfile.TemporaryDirectory(prefix="arxiv-source-check-") as tmp:
        extracted = Path(tmp)
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            require(archive.namelist() == sorted(files), "unexpected archive inventory")
            for name in archive.namelist():
                recovered = archive.read(name)
                require(recovered == files[name], "archive byte mismatch: " + name)
                dest = extracted / name
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(recovered)
        require(dependencies(extracted) == names, "extracted dependency graph differs")
    output.mkdir(parents=True, exist_ok=True)
    (output / "arxiv-v1-source.zip").write_bytes(raw)
    form = dict(metadata, abstract=abstract)
    (output / "arxiv-form.json").write_text(json.dumps(form, indent=2) + "\n")
    fields = [("Title", metadata["title"]), ("Authors", ", ".join(metadata["authors"])),
              ("Primary category", metadata["primary_category"]),
              ("Cross-list", ", ".join(metadata.get("cross_list_categories", []))),
              ("License", metadata["license"]), ("Abstract", abstract)]
    (output / "arxiv-form.txt").write_text(
        "\n\n".join(label + ":\n" + value for label, value in fields) + "\n")
    commit = subprocess.check_output(
        ["git", "-C", str(paper), "rev-parse", "HEAD"], text=True).strip()
    report = {
        "format": "depth-read-arxiv-v1", "checkout_commit": commit,
        "upload_file": "arxiv-v1-source.zip", "upload_sha256": sha256(raw),
        "files": {name: {"bytes": len(data), "sha256": sha256(data)}
                  for name, data in sorted(files.items())},
        "abstract_characters": len(abstract), "abstract_words": len(abstract.split()),
        "exact_archive_bytes_and_dependencies_verified": True,
        "unused_files_and_operational_records_excluded": True,
        "new_training_or_scoring": False, "separate_pdf_compilation": False,
        "arxiv_submission_performed": False,
        "pending": ["Inspect arXiv-processed PDF", "Author's final submission"],
    }
    (output / "BUILD-REPORT.json").write_text(json.dumps(report, indent=2) + "\n")
    (output / "SHA256SUMS").write_text(sha256(raw) + "  arxiv-v1-source.zip\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paper", type=Path, default=ROOT / "paper")
    parser.add_argument("--metadata", type=Path, default=ROOT / "submission/arxiv-v1.json")
    parser.add_argument("--output", type=Path, default=ROOT / "output/arxiv-v1")
    args = parser.parse_args()
    report = build(args.paper.resolve(), json.loads(args.metadata.read_text()), args.output.resolve())
    print(json.dumps({key: value for key, value in report.items() if key != "files"}, indent=2))


if __name__ == "__main__":
    main()
