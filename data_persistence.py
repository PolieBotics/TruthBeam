import os
import h5py
from config import Config
from utils import create_dir_if_not_exists
from datetime import datetime

def create_hdf5_file():
    create_dir_if_not_exists(Config.OUTPUT_DIR)
    run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(Config.OUTPUT_DIR, run_name)
    create_dir_if_not_exists(run_dir)
    file_path = os.path.join(run_dir, "data.h5")
    hf = h5py.File(file_path, "w")
    hf.create_dataset("emissions", (Config.LOOP_COUNT, Config.EMISSION_HEIGHT, Config.EMISSION_WIDTH, 3), dtype='uint8')
    hf.create_dataset("recordings", (Config.LOOP_COUNT, Config.RECORDING_HEIGHT, Config.RECORDING_WIDTH, 3), dtype='uint8')
    hf.create_dataset("hashes", (Config.LOOP_COUNT,32), dtype='uint8')
    return hf, file_path

def write_metadata(hf, init_blockhash, final_hash):
    hf.attrs["initial_blockhash"] = init_blockhash.hex()
    hf.attrs["final_hash"] = final_hash.hex()
    hf.attrs["dummy_run"] = Config.DUMMY_RUN
