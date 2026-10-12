"""Locate this project and the containing stock MMSegmentation checkout."""
import sys
from pathlib import Path


def bootstrap():
    project_root = Path(__file__).resolve().parents[1]
    mmseg_root = project_root.parent.parent
    if not (mmseg_root / 'mmseg' / '__init__.py').is_file():
        raise RuntimeError(
            'Copy the complete EACG directory into mmsegmentation/project/EACG '
            '(or mmsegmentation/projects/EACG) before using these launchers.')
    # Project registration is independent of the parent folder name.
    sys.path.insert(0, str(mmseg_root))
    sys.path.insert(0, str(project_root))
    return mmseg_root


def run_upstream(name):
    import runpy

    mmseg_root = bootstrap()
    entry = mmseg_root / 'tools' / name
    if not entry.is_file():
        raise FileNotFoundError(f'A complete MMSegmentation checkout is required: {entry}')
    # Forward all CLI arguments to the official entrypoint unchanged.
    runpy.run_path(str(entry), run_name='__main__')
