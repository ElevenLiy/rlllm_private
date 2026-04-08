from pathlib import Path
from setuptools import setup

ROOT = Path(__file__).parent.resolve()

packages = []
package_dir = {}
for init_file in ROOT.rglob("__init__.py"):
    rel_dir = init_file.parent.relative_to(ROOT)
    if str(rel_dir) == ".":
        pkg_name = "verl"
        pkg_path = "."
    else:
        pkg_name = "verl." + ".".join(rel_dir.parts)
        pkg_path = str(rel_dir)
    packages.append(pkg_name)
    package_dir[pkg_name] = pkg_path

version = (ROOT / "version" / "version").read_text(encoding="utf-8").strip()

setup(
    name="verl",
    version=version,
    description="Vendored VERL used by rllm-private",
    python_requires=">=3.10",
    packages=sorted(packages),
    package_dir=package_dir,
    include_package_data=True,
    package_data={"verl": ["version/version"]},
    install_requires=["packaging"],
)
