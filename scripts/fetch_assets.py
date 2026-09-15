"""Download pinned robot assets on the remote host, never during package import."""
import argparse
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen
import xml.etree.ElementTree as ET

REVISION = "8161bba264d7fa7c99ca301e91e7fb44737676ad"
BASE = f"https://raw.githubusercontent.com/google-deepmind/mujoco_menagerie/{REVISION}/robotstudio_so101/"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("artifacts/so101"))
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output exists; choose a new directory to preserve existing assets")
    args.output.mkdir(parents=True)
    with urlopen(BASE + "so101.xml", timeout=60) as response:
        xml = response.read()
    files = {"so101.xml": xml}
    model = ET.fromstring(xml)
    for mesh in model.findall("./asset/mesh"):
        relative = "assets/" + mesh.attrib["file"]
        with urlopen(BASE + relative, timeout=60) as response:
            files[relative] = response.read()
    with urlopen(BASE + "LICENSE", timeout=60) as response:
        files["LICENSE"] = response.read()
    manifest = {"revision": REVISION, "source": BASE, "sha256": {}}
    for name, content in files.items():
        path = args.output / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        manifest["sha256"][name] = hashlib.sha256(content).hexdigest()
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(args.output)


if __name__ == "__main__":
    main()
