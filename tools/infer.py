"""Predict a single image and save a class-index PNG (no annotation needed)."""
import argparse
from pathlib import Path

from _bootstrap import bootstrap


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config')
    parser.add_argument('checkpoint')
    parser.add_argument('image')
    parser.add_argument('--out', required=True, help='Output PNG path; IDs are 0..C-1.')
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    if Path(args.out).suffix.lower() != '.png':
        parser.error('--out must end in .png to preserve integer class IDs.')
    if Path(args.out).resolve() == Path(args.image).resolve():
        parser.error('--out must be different from the input image.')
    bootstrap()

    import numpy as np
    from mmengine.config import Config
    from mmengine.utils import import_modules_from_strings
    from mmseg.apis import inference_model, init_model
    from PIL import Image

    cfg = Config.fromfile(args.config)
    if cfg.get('custom_imports'):
        import_modules_from_strings(**cfg.custom_imports)
    model = init_model(cfg, args.checkpoint, device=args.device)
    prediction = inference_model(model, args.image)
    mask = prediction.pred_sem_seg.data.squeeze(0).cpu().numpy()
    if mask.max() > 255:
        raise ValueError('This exporter supports up to 256 class IDs.')
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask.astype(np.uint8)).save(output)
    print(f'Saved class-index mask to {output}')


if __name__ == '__main__':
    main()
