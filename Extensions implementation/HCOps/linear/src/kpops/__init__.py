from pathlib import Path
import torch
import os
PACKAGE_DIR = Path(__file__).parent.absolute()


def load_torch_ops(lib_dir: Path):
    print("load kpops:", lib_dir)
    # @see: "https://docs.google.com/document/d/1_W62p8WJOQQUzPsJYa7s701JXt0qf2OfLub2sbkHOaU/edit?tab=t.0#heading=h.i5q3a2pv0qzc"
    for root, dirs, files in  os.walk(lib_dir):
        for file in files:
            if file.endswith(".so") :
                torch.ops.load_library(str(Path(root).joinpath(file)))


load_torch_ops(PACKAGE_DIR / "_kpops")
