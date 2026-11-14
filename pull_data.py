"""pull_data.py -- download labelled datasets from Roboflow Universe.

Run it with the isolated interpreter, NOT plain python:

    .venv-roboflow\\Scripts\\python.exe pull_data.py
    .venv-roboflow\\Scripts\\python.exe pull_data.py samples-x5zfd/keys-jcx6v
    .venv-roboflow\\Scripts\\python.exe pull_data.py some-workspace/some-project/3

No arguments pulls everything in DATASETS. An argument is the slug from a
Universe URL (universe.roboflow.com/<workspace>/<project>), optionally with
a version number; without one the newest version is used.

Why a separate interpreter: the roboflow package depends on
opencv-python-headless, which overwrites the cv2 that detect.py needs for
its windows. Living in .venv-roboflow, it cannot break the detector.

Needs ROBOFLOW_API_KEY in .env  (free account: app.roboflow.com ->
Settings -> API Keys -> Private API Key).
"""
import os
import sys

try:
    import yaml
    from roboflow import Roboflow
except ImportError:
    sys.exit("roboflow is not installed in this interpreter. Run:\n"
             "  .venv-roboflow\\Scripts\\python.exe pull_data.py")

from targets import TARGETS

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")

# (workspace, project) -- found on Roboflow Universe, chosen to cover the
# targets in targets.py. Add a line here to make a dataset part of the set.
DATASETS = [
    ("obj-detection-kd5zb", "left-off-objects-1"),   # bag, laptop, wallet, keys, phone
    ("samples-x5zfd", "keys-jcx6v"),                 # keys, ~520 images, public domain
    ("workspace-kkggm", "final-project-watch-wallet-phone-pen"),   # watch, wallet, phone
    ("it-3pwlf", "glasses-detection-cqd8m"),         # glasses, ~1200 images
    ("tahsin", "tv-remote"),                         # remote, CC BY 4.0
    ("jilson", "audio_device_detection"),            # earbuds + headset, ~220 images
]


def load_key():
    """ROBOFLOW_API_KEY from the environment, else from .env beside this file."""
    key = os.environ.get("ROBOFLOW_API_KEY", "").strip()
    if key:
        return key
    path = os.path.join(HERE, ".env")
    if os.path.exists(path):
        with open(path, encoding="utf-8-sig") as fh:      # -sig: tolerate a BOM
            for line in fh:
                line = line.strip()
                if line.startswith("#") or "=" not in line:
                    continue
                name, _, value = line.partition("=")
                if name.strip() == "ROBOFLOW_API_KEY":
                    return value.strip().strip('"').strip("'")
    return ""


def normalise(name):
    return " ".join(str(name).lower().replace("-", " ").replace("_", " ").split())


def summarise(folder):
    """What did we actually get? Images, boxes per class, and which of our
    targets each class maps onto -- so label mismatches show up now, not in
    the middle of a benchmark."""
    with open(os.path.join(folder, "data.yaml"), encoding="utf-8") as fh:
        names = yaml.safe_load(fh).get("names")
    names = list(names.values()) if isinstance(names, dict) else list(names)

    images, boxes = 0, [0] * len(names)
    for where, _, files in os.walk(folder):
        parts = where.split(os.sep)
        if "images" in parts:
            images += sum(f.lower().endswith((".jpg", ".jpeg", ".png")) for f in files)
        if "labels" in parts:
            for f in files:
                if not f.endswith(".txt"):
                    continue
                with open(os.path.join(where, f), encoding="utf-8") as fh:
                    for line in fh:
                        vals = line.split()
                        if vals and int(float(vals[0])) < len(boxes):
                            boxes[int(float(vals[0]))] += 1

    alias_to_target = {normalise(a): t for t, spec in TARGETS.items() for a in spec["aliases"]}
    print("  %d images" % images)
    for name, count in zip(names, boxes):
        target = alias_to_target.get(normalise(name))
        print("    %-24s %5d boxes   %s"
              % (name, count, "-> target '%s'" % target if target else "(not one of our targets)"))


def pull(rf, workspace, project, version=None):
    folder = os.path.join(DATA, project)
    print("\n%s/%s" % (workspace, project))
    if os.path.exists(os.path.join(folder, "data.yaml")):
        print("  already downloaded -> data\\%s" % project)
        summarise(folder)
        return

    proj = rf.workspace(workspace).project(project)
    if version is None:
        # ids look like "workspace/project/3"; take the highest number.
        numbers = [int(v["id"].rsplit("/", 1)[-1]) for v in proj.get_version_information()]
        if not numbers:
            print("  no published versions -- nothing to download")
            return
        version = max(numbers)
    print("  downloading version %d as yolov8 ..." % version)
    proj.version(version).download("yolov8", location=folder, overwrite=False)
    summarise(folder)


def main():
    key = load_key()
    if not key:
        sys.exit("no ROBOFLOW_API_KEY found. Add this line to .env:\n"
                 "  ROBOFLOW_API_KEY=<your private api key>\n"
                 "(free account: app.roboflow.com -> Settings -> API Keys)")
    os.makedirs(DATA, exist_ok=True)
    rf = Roboflow(api_key=key)

    wanted = []
    for arg in sys.argv[1:]:
        parts = arg.strip("/").split("/")
        if len(parts) < 2:
            sys.exit("expected workspace/project[/version], got '%s'" % arg)
        wanted.append((parts[0], parts[1], int(parts[2]) if len(parts) > 2 else None))
    if not wanted:
        wanted = [(w, p, None) for w, p in DATASETS]

    failed = 0
    for workspace, project, version in wanted:
        try:
            pull(rf, workspace, project, version)
        except Exception as exc:                       # one bad dataset must not stop the rest
            failed += 1
            print("  FAILED: %s: %s" % (type(exc).__name__, exc))
    print("\ndone: %d ok, %d failed" % (len(wanted) - failed, failed))


if __name__ == "__main__":
    main()
