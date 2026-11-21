"""Negative controls for the source-only upload and form metadata."""
import io
from pathlib import Path
import sys
import tempfile
import unittest
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from prepare_arxiv import abstract_text, build, dependencies, zip_bytes


class UploadControls(unittest.TestCase):
    def test_only_actual_dependencies(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "main.tex").write_text(
                r"\input{table}" + "\n" + r"\includegraphics{figure.pdf}")
            (root / "table.tex").write_text("row")
            (root / "figure.pdf").write_bytes(b"figure")
            (root / "unused.pdf").write_bytes(b"not for upload")
            (root / "references.bib").write_text("unused companion metadata")
            self.assertEqual(dependencies(root), ["figure.pdf", "main.tex", "table.tex"])

    def test_parent_traversal_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "main.tex").write_text(r"\input{../secret}")
            with self.assertRaisesRegex(ValueError, "unsafe dependency"):
                dependencies(root)

    def test_absolute_path_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "main.tex").write_text(r"\input{/secret}")
            with self.assertRaisesRegex(ValueError, "unsafe dependency"):
                dependencies(root)

    def test_symlink_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "main.tex").write_text(r"\input{alias}")
            (root / "real.tex").write_text("not an accepted alias")
            (root / "alias.tex").symlink_to(root / "real.tex")
            with self.assertRaisesRegex(ValueError, "nonregular"):
                dependencies(root)

    def test_missing_dependency_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "main.tex").write_text(r"\includegraphics{missing.pdf}")
            with self.assertRaisesRegex(ValueError, "unresolved dependency"):
                dependencies(root)

    @staticmethod
    def abstract_fixture(text):
        return (r"\newcommand{\Gain}{-0.045051}" + "\n"
                + r"{\frontMatterTitleStyle Abstract}" + "\n"
                + r"\end{center}" + "\n" + text + "\n"
                + r"\section{Introduction}")

    def test_numeric_macros_expand(self):
        result = abstract_text(self.abstract_fixture(r"Gain \Gain{}; $Q_1$; 99.39\%."))
        self.assertEqual(result, r"Gain -0.045051; $Q_1$; 99.39%.")

    def test_unexpanded_macro_rejected(self):
        with self.assertRaisesRegex(ValueError, "unexpanded"):
            abstract_text(self.abstract_fixture(r"Gain \Unknown{}."))

    def test_long_abstract_rejected(self):
        with self.assertRaisesRegex(ValueError, "1920"):
            abstract_text(self.abstract_fixture("x" * 1921))

    def test_archive_is_sorted_deterministic_and_byte_exact(self):
        files = {"table.tex": b"row", "main.tex": b"document"}
        first = zip_bytes(files)
        self.assertEqual(first, zip_bytes(dict(reversed(list(files.items())))))
        with zipfile.ZipFile(io.BytesIO(first)) as archive:
            self.assertEqual(archive.namelist(), ["main.tex", "table.tex"])
            self.assertEqual({name: archive.read(name) for name in archive.namelist()}, files)

    def test_unconfirmed_license_blocks_build(self):
        with self.assertRaisesRegex(ValueError, "license still needs"):
            build(Path("."), {"license_author_confirmed": False}, Path("."))


if __name__ == "__main__":
    unittest.main()
