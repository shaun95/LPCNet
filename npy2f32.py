import numpy as np
import argparse
import sys
import os

dir_name = os.path.dirname(os.path.realpath(__file__))
npy_data = np.load(os.path.join(dir_name, sys.argv[1]))
npy_data = npy_data.astype(np.float32)
npy_data = npy_data.reshape((-1,))
npy_data.tofile(os.path.join(dir_name, sys.argv[1].split(".")[0] + ".f32"))
