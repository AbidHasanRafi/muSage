"""Setup configuration for μSage (MuSage) - Adaptive Web Agent"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="musage",
    version="0.1.0",
    author="Md. Abid Hasan Rafi",
    author_email="contact@abidhasanrafi.com",
    description="μSage — Lightweight adaptive web-powered conversational assistant",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/abidhasanrafi/muSage",
    project_urls={
        "Homepage": "https://github.com/abidhasanrafi/muSage",
        "Developer": "https://abidhasanrafi.github.io/",
        "AI Extension": "https://aiextension.org/",
    },
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[
        "beautifulsoup4>=4.12.0",
        "requests>=2.31.0",
        "lxml>=4.9.0",
        "sentence-transformers>=2.2.0",
        "numpy>=1.24.0",
        "faiss-cpu>=1.7.4",
        "ddgs>=0.0.1",
        "colorama>=0.4.6",
        "tqdm>=4.66.0",
    ],
    entry_points={
        "console_scripts": [
            "musage=musage.cli:main",
        ],
    },
)
