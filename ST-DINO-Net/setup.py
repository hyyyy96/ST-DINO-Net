from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="stdinonet",
    version="1.0.0",
    author="Hongyang Quan, Lianglin Zou, Jifeng Song",
    author_email="songjifeng@ncepu.edu.cn",
    description="ST-DINO-Net: Spatiotemporal Network for Cloud Classification",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/hyyy96/ST-DINO-Net",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Atmospheric Science",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    include_package_data=True,
)