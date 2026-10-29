from pathlib import Path
import shutil
from setuptools import setup, find_packages
from setuptools.command.build_py import build_py

ROOT = Path(__file__).resolve().parent

class BuildWithResources(build_py):
    def run(self):
        super().run()
        target = Path(self.build_lib) / "plaque_analysis" / "_resources"
        for name in ("R", "config"):
            shutil.copytree(ROOT / "code" / name, target / "code" / name, dirs_exist_ok=True)
        for name in ("run.py", "requirement.txt"):
            shutil.copy2(ROOT / "code" / name, target / "code" / name)
        shutil.copytree(ROOT / "figures", target / "figures", dirs_exist_ok=True)

setup(
    name="plaque-analysis-public", version='0.1.1',
    description="Public interfaces, data processing and plotting for plaque analysis",
    author="Doctor Wang", license_expression="MIT", license_files=["LICENSE"],
    packages=find_packages("code"), package_dir={"": "code"},
    python_requires=">=3.12,<3.13", install_requires=[], extras_require={'analysis': ['numpy==2.1.3', 'pandas==3.0.2', 'scipy==1.17.0', 'matplotlib==3.11.0', 'scikit-learn==1.8.0', 'scikit-image==0.26.0', 'tifffile==2026.2.24', 'opencv-python==4.14.0.94', 'Pillow==12.1.1', 'joblib==1.5.2', 'lxml==6.1.3'], 'czi': ['aicspylibczi==4.0.0'], 'modeling': ['numpy==2.1.3', 'pandas==3.0.2', 'scipy==1.17.0', 'matplotlib==3.11.0', 'scikit-learn==1.8.0', 'scikit-image==0.26.0', 'tifffile==2026.2.24', 'opencv-python==4.14.0.94', 'Pillow==12.1.1', 'joblib==1.5.2', 'lxml==6.1.3', 'torch==2.12.1'], 'rna': ['numpy==2.1.3', 'pandas==3.0.2', 'scipy==1.17.0', 'matplotlib==3.11.0', 'scikit-learn==1.8.0', 'scikit-image==0.26.0', 'tifffile==2026.2.24', 'opencv-python==4.14.0.94', 'Pillow==12.1.1', 'joblib==1.5.2', 'lxml==6.1.3', 'scanpy==1.12.4', 'anndata==0.12.6', 'scikit-misc==0.5.3', 'igraph==1.0.0', 'leidenalg==0.11.0'], 'full': ['numpy==2.1.3', 'pandas==3.0.2', 'scipy==1.17.0', 'matplotlib==3.11.0', 'scikit-learn==1.8.0', 'scikit-image==0.26.0', 'tifffile==2026.2.24', 'aicspylibczi==4.0.0', 'opencv-python==4.14.0.94', 'Pillow==12.1.1', 'torch==2.12.1', 'joblib==1.5.2', 'lxml==6.1.3', 'scanpy==1.12.4', 'anndata==0.12.6', 'scikit-misc==0.5.3', 'igraph==1.0.0', 'leidenalg==0.11.0']},
    entry_points={"console_scripts": ["plaque-analysis=plaque_analysis.cli:main"]},
    cmdclass={"build_py": BuildWithResources}, include_package_data=False,
)
