from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf8") as fh:
    long_description = fh.read()

with open("requirements.txt") as f:
    install_requires = f.read().splitlines()


setup(
    name="lazyqsar",
    version="0.4",
    author="Miquel Duran-Frigola",
    author_email="miquel@ersilia.io",
    url="https://github.com/ersilia-os/lazy-qsar",
    description="A library to quickly build QSAR models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    license="GPLv3",
    python_requires=">=3.7",
    install_requires=install_requires,
    packages=find_packages(exclude=("utilities")),
    entry_points={"console_scripts": []},
    classifiers=[
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords="qsar machine-learning chemistry computer-aided-drug-design",
    project_urls={"Source Code": "https://github.com/ersilia-os/lazy-qsar"},
    include_package_data=True,
)
