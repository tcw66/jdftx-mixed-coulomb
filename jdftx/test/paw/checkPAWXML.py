#!/usr/bin/env python3
"""Malformed PAW XML must fail before constructing a Hamiltonian."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET

exe, dataset = map(Path, sys.argv[1:3])
text = dataset.read_text()
env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
with tempfile.TemporaryDirectory(prefix="jdftx-paw-xml-") as work:
    path = Path(work) / "input.xml"
    def check(document, valid, label):
        path.write_text(document)
        proc = subprocess.run([str(exe), str(path), "--parse-only"], env=env,
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        assert (proc.returncode == 0) == valid, f"{label}: {proc.stdout}"
        print(label, "passed")
    check(text, True, "Original dataset")
    root = ET.fromstring(text)
    root.find("ae_core_density").set("grid", "missing-grid")
    check(ET.tostring(root, encoding="unicode"), False, "Unknown radial grid")
    root = ET.fromstring(text)
    root.find("projector_function").text = "0 0"
    check(ET.tostring(root, encoding="unicode"), False, "Truncated radial data")
    root = ET.fromstring(text)
    root.find("valence_states")[1].set("id", root.find("valence_states")[0].get("id"))
    check(ET.tostring(root, encoding="unicode"), False, "Duplicate state identity")
    check(text.replace("<paw_dataset", '<!DOCTYPE paw_dataset [<!ENTITY x "value">]>\n<paw_dataset', 1),
          False, "DTD rejection")
    root = ET.fromstring(text)
    root.find("zero_potential").text = "nan " + root.find("zero_potential").text.split(maxsplit=1)[1]
    check(ET.tostring(root, encoding="unicode"), False, "Nonfinite radial data")
    root = ET.fromstring(text)
    root.find("xc_functional").set("name", "LDA")
    check(ET.tostring(root, encoding="unicode"), False, "Unsupported functional")
