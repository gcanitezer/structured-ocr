# Structured OCR

LaTeX OCR System for Full Document Reconstruction

## Features

- OCR pipeline for LaTeX documents
- Compilability verification
- Training and evaluation framework
- REST API and CLI interface

## Installation

```bash
pip install -e .
```

## Usage

```bash
latexocr infer image.png -o output.tex
latexocr verify output.tex
latexocr train --data-dir ./data
latexocr evaluate
```

## Development

```bash
make dev-install
make test
make lint
make format
```

## Docker

```bash
docker build -t structured-ocr .
docker run -it structured-ocr
```