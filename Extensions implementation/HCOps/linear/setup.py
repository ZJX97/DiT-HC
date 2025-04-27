from pathlib import Path
import shutil
import sys
import subprocess
from setuptools import find_namespace_packages, setup, Extension
from setuptools.command.build_ext import build_ext
from setuptools.command.build_py import build_py
from wheel.bdist_wheel import bdist_wheel

# Name of your package; Must match the directory name under `CSRC_DIR`:
PKG_NAME = "kpops"
# Path to the directory of setup.py file:
SETUP_DIR = Path(__file__).parent.absolute()
# Where to create the cmake build directory:
BUILD_DIR = Path(SETUP_DIR, "build")
# Path to the c/c++ source directory:
CSRC_DIR = Path(SETUP_DIR, "csrc")
# Where to install the op library:
TORCH_OPS_DIR = Path(SETUP_DIR, "src", PKG_NAME, "_kpops")


class CMakeExtension(Extension):
    def __init__(self, name, source_dir, build_dir, install_dir):
        Extension.__init__(self, name, sources=[])
        # C/C++ source directory
        self.source_dir = Path(source_dir).absolute()
        # Build directory
        self.build_dir = Path(build_dir).absolute()
        # Lib installation directory
        self.install_dir = Path(install_dir).absolute()


class CMakeBuild(build_ext):

    def run(self):
        try:
            subprocess.check_output(["cmake", "--version"])
        except OSError:
            raise RuntimeError("CMake must be installed")

        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext: CMakeExtension):
        build_args = [
            "-S",
            ext.source_dir,
            "-B",
            ext.build_dir,
            "Release",
        ]
        subprocess.check_call(["bash", "scripts/build.sh"] + build_args)
        install_args = [
            "--install",
            ext.build_dir,
            "--prefix",
            ext.install_dir,
        ]
        subprocess.check_call(["cmake"] + install_args)


class BuildPy(build_py):
    def run(self):
        self.run_command("build_ext")
        super().run()


class BDistWheel(bdist_wheel):
    def run(self):
        self.run_command("build_py")
        super().run()

        dist_dir = Path("dist")
        dist_dir.mkdir(exist_ok=True)

        wheel_dir = Path(self.dist_dir)
        wheels = list(wheel_dir.glob("*.whl"))
        if wheels:
            wheel_file = wheels[0]
            shutil.copy2(wheel_file, dist_dir / wheel_file.name)


setup(
    ext_modules=[
        CMakeExtension(
            name=f"{PKG_NAME}._kpops",
            source_dir=CSRC_DIR,
            build_dir=BUILD_DIR,
            install_dir=TORCH_OPS_DIR,
        )
    ],
    cmdclass={
        "build_ext": CMakeBuild,
        "build_py": BuildPy,
        "bdist_wheel": BDistWheel,
    },
    packages=find_namespace_packages(where="./src"),
    package_dir={PKG_NAME: f"./src/{PKG_NAME}"},
    package_data={PKG_NAME: ["_kpops/lib/*.so"]},
)
