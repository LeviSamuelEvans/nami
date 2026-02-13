# nami

[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org) [![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c)](https://pytorch.org)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE) [![codecov](https://codecov.io/gh/LeviSamuelEvans/nami/branch/main/graph/badge.svg)](https://codecov.io/gh/LeviSamuelEvans/nami)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Documentation](https://github.com/LeviSamuelEvans/nami/actions/workflows/docs.yaml/badge.svg)](https://levisamuelevans.github.io/nami/)

A minimal, modular flow matching library for research studies, providing the building blocks for flow-based generative models with explicit shape semantics and pluggable components. Primarily, this is a personal tool for use in other projects (and for fun). Much of the design is heavily inspired by the fantastic [Zuko](https://github.com/probabilists/zuko/tree/master) library, who coined the term `LazyDistribution`.

See the [documentation](https://levisamuelevans.github.io/nami/) for examples, tutorials and the full reference API.

## Installation

### Prerequisites

- Python >= 3.10
- [pixi](https://pixi.sh) package manager

### Setup

Clone the repository and run the setup task, which installs nami in editable mode:

```bash
git clone https://github.com/LeviSamuelEvans/nami
cd nami
```

```bash
pixi run setup
```

This creates a conda environment via pixi and installs the package with `pip install -e .`.

### Without pixi

If you prefer not to use pixi, you can install directly with pip (requires PyTorch >= 2.0):

```bash
pip install -e .
```

### Development tasks

pixi provides several convenience tasks:

| Command | Description |
|---------|-------------|
| `pixi run test` | Run tests with pytest |
| `pixi run cov` | Run tests with coverage report |
| `pixi run lint` | Lint with ruff |
| `pixi run fmt` | Format with ruff |
| `pixi run typecheck` | Type-check with mypy |
| `pixi run docs` | Build Sphinx documentation |
| `pixi run kernel` | Install a Jupyter kernel named `nami` |
