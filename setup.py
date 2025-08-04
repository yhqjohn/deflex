from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="deflex",
    version="0.1.0",
    author="Deflex Development Team",
    author_email="deflex@example.com",
    description="Deep Formula Discovery for Complex Systems",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/username/deflex",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Mathematics",
    ],
    install_requires=[
        "torch>=2.0.0",           # PyTorch for neural networks
        "julia>=0.6.0",           # Python-Julia interface
        "scikit-learn>=1.0.0",    # Base estimator classes and utilities
        "numpy>=1.20.0",          # Numerical computing
        "scipy>=1.7.0",           # Scientific computing
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov",
            "black",
            "isort",
            "flake8",
            "mypy",
        ],
        "docs": [
            "sphinx>=4.0",
            "sphinx-rtd-theme",
            "sphinx-autodoc-typehints",
        ],
    },
    entry_points={
        "console_scripts": [
            "deflex-train=deflex.cli:main",  # Command-line interface
        ],
    },
    python_requires=">=3.8",
    include_package_data=True,
    zip_safe=False,
)