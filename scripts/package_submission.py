"""Package only the four editable modules; never include grading logs or history."""

from datetime import datetime
from pathlib import Path
import subprocess
import zipfile

root = Path(__file__).resolve().parents[1]
branch = subprocess.run(
    ["git", "branch", "--show-current"], cwd=root, capture_output=True, text=True
)
if "teacher" in branch.stdout.lower():
    raise SystemExit("Refusing to create a student submission from a teacher branch.")
names = ["data_parallel.py", "pipeline.py", "finetune.py", "inference.py"]
for name in names:
    if not (root / "assignment" / name).is_file():
        raise SystemExit("Missing " + name)
output = (
    root
    / "dist"
    / ("hw56_submission_" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".zip")
)
output.parent.mkdir(exist_ok=True)
with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
    for name in names:
        archive.write(root / "assignment" / name, "assignment/" + name)
print(output)
