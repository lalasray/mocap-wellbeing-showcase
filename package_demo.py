"""Build a portable source bundle; dependencies are installed by the recipient."""
import argparse
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parent
FILES = ['demo.py','motion.py','emotion_model.py','protective_model.py',
         'setup_model.py','setup_protective_model.py','setup_demo.bat','launch_demo.bat',
         'package_demo.py','requirements.txt','requirements-model.txt','README.md',
         'models/model_info.json',
         'models/stgcn_500_5.pt','models/HOT_bi_cnn_banet_coordinate.hdf5',
         'vendor/taew_stgcn/st_gcn.py','vendor/taew_stgcn/stgcn_graph.py',
         'vendor/taew_stgcn/tgcn.py','vendor/taew_stgcn/LICENSE','vendor/emopain_fusion/LICENSE']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--include-sample',action='store_true')
    args = parser.parse_args()
    files = [ROOT/name for name in FILES]
    if args.include_sample:
        files += list(ROOT.glob('*.c3d'))
    for path in files:
        if not path.is_file():
            raise SystemExit(f'Missing {path}. Run setup_demo.bat before packaging.')
    target = ROOT/'dist/Pose_stress.zip'
    target.parent.mkdir(exist_ok=True)
    with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path,Path('Pose_stress')/path.relative_to(ROOT))
    print(f'Created {target} ({target.stat().st_size/1024**2:.1f} MiB)')


if __name__ == '__main__':
    main()
