"""
# Unified Tile Extractor

A highly modular, efficient, and annotation-aware patch extraction pipeline for WSIs.

## Features
- CuCIM/OpenSlide backend
- Otsu, Pen, Annotation-based masking
- HDF5 / PNG saving
- Tissue filtering with deep model (.pt)
- Full CLI YAML configuration

## Usage
```bash
pip install -e .
tile-extract --config config/unified_patch_config.yaml
```
"""
