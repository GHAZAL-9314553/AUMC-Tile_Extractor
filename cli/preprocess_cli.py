import argparse
import yaml
import os

from masking_utils import UnifiedMasker
from tile_reader import get_wsi_reader
from patch_extractor import EfficientPatchExtractor
from batch_extractor import BatchExtractor
from annotation_loader import load_annotation_mask

def load_config(path: str) -> dict:
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def parse_args():
    parser = argparse.ArgumentParser(description="Unified Tile Extractor CLI")
    parser.add_argument('--config', type=str, required=True, help="Path to config.yaml")
    parser.add_argument('--override_save_dir', type=str, help="Optional override save directory")
    return parser.parse_args()

def main():
    args = parse_args()
    config = load_config(args.config)

    wsi_paths = config['data']['wsi_paths']
    save_dirs = config['data']['save_dirs']
    annotation_paths = config['data'].get('annotation_paths', [None] * len(wsi_paths))

    reader_backend = config['reader']['backend']
    patch_size = config['extraction']['patch_size']
    stride = config['extraction'].get('stride', patch_size)
    level = config['extraction'].get('level', 0)

    use_otsu = config['masking'].get('use_otsu', True)
    use_pen_filter = config['masking'].get('use_pen_filter', True)
    use_annotation = config['masking'].get('use_annotation', False)

    extractors = []
    for wsi_path, save_dir, annotation_path in zip(wsi_paths, save_dirs, annotation_paths):
        reader = get_wsi_reader(reader_backend, wsi_path)

        annotation_mask = None
        if use_annotation and annotation_path is not None:
            shape = (reader.get_dimensions()[1], reader.get_dimensions()[0])  # H, W
            annotation_mask = load_annotation_mask(annotation_path, shape)

        masker = UnifiedMasker(
            use_otsu=use_otsu,
            use_pen_filter=use_pen_filter,
            use_annotation=use_annotation,
            annotation_mask=annotation_mask
        )

        extractor = EfficientPatchExtractor(
            reader=reader,
            masker=masker,
            patch_size=patch_size,
            stride=stride,
            level=level
        )

        if args.override_save_dir:
            save_dir = os.path.join(args.override_save_dir, os.path.basename(save_dir))

        extractors.append((extractor, save_dir))

    batch = BatchExtractor(
        extractors=[e[0] for e in extractors],
        save_dirs=[e[1] for e in extractors]
    )
    batch.run()

if __name__ == "__main__":
    main()
