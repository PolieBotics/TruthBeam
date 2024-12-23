import sys
import os
import time
import numpy as np
import pygame
import cv2
import logging
from PIL import Image
from config import Config
from utils import blake3_hash
from data_persistence import create_hdf5_file, write_metadata
from perlin import generate_image_from_hash
from blockchain import submit_hash_to_rsk, get_fresh_blockhash

sys.path.append("../python-common")
import TIS

logger = logging.getLogger("TruthBeam.Record")
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

class CustomData:
    def __init__(self):
        self.busy = False
        self.latest_image = None
        self.new_image_available = False

def on_new_image(tis, userdata):
    if userdata.busy:
        return
    userdata.busy = True
    image = tis.get_image()
    userdata.latest_image = image.copy()
    userdata.new_image_available = True
    userdata.busy = False

def capture_image(CD, timeout=2.0):
    start = time.perf_counter()
    while (time.perf_counter() - start) < timeout:
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN and event.key in [pygame.K_q, pygame.K_ESCAPE]:
                return None
        if CD.new_image_available and CD.latest_image is not None:
            img = CD.latest_image
            CD.new_image_available = False
            return img
        time.sleep(0.01)
    return None

def main():
    os.environ['DISPLAY'] = ':0'
    Tis = TIS.TIS()
    Tis.open_device(Config.CAMERA_SERIAL,
                    Config.RECORDING_WIDTH,
                    Config.RECORDING_HEIGHT,
                    "15/1", TIS.SinkFormats.BGRA, False)
    CD = CustomData()
    Tis.set_image_callback(on_new_image, CD)
    Tis.set_property("TriggerMode", "On")
    Tis.set_property("GainAuto", "Off")
    Tis.set_property("Gain", Config.GAIN_VALUE)
    Tis.set_property("ExposureAuto", "Off")
    Tis.set_property("ExposureTime", Config.EXPOSURE_TIME)
    try:
        Tis.set_property("BalanceWhiteAuto", "Off")
        Tis.set_property("BalanceWhiteRed", 1.2)
        Tis.set_property("BalanceWhiteGreen", 1.0)
        Tis.set_property("BalanceWhiteBlue", 1.4)
    except:
        pass
    Tis.start_pipeline()

    pygame.init()
    pygame.font.init()
    width, height = Config.EMISSION_WIDTH, Config.EMISSION_HEIGHT
    flags = pygame.FULLSCREEN | pygame.HWSURFACE | pygame.DOUBLEBUF
    screen = pygame.display.set_mode((width, height), flags)
    pygame.mouse.set_visible(False)

    logger.info("Warming up camera...")
    warmup_start = time.perf_counter()
    while (time.perf_counter() - warmup_start < Config.WARMUP_TIME):
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN and event.key in [pygame.K_q, pygame.K_ESCAPE]:
                Tis.stop_pipeline()
                pygame.quit()
                logger.info("Quit during warmup.")
                return

    logger.info("Getting fresh blockhash from RSK...")
    fresh_blockhash = get_fresh_blockhash()

    INIT_VEC = blake3_hash(fresh_blockhash)
    CURRENT_HASH = INIT_VEC

    hf, hdf5_path = create_hdf5_file()
    logger.info(f"Created HDF5 file: {hdf5_path}")

    running = True
    for i in range(Config.LOOP_COUNT):
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN and event.key in [pygame.K_q, pygame.K_ESCAPE]:
                running=False
        if not running:
            break

        emission_img = generate_image_from_hash(CURRENT_HASH)
        emission_np = np.array(emission_img)
        surf = pygame.surfarray.make_surface(emission_np.swapaxes(0,1))
        screen.blit(surf,(0,0))
        pygame.display.flip()

        end_wait = time.perf_counter()+Config.LATENCY_PERIOD
        while time.perf_counter()<end_wait:
            for event in pygame.event.get():
                if event.type==pygame.KEYDOWN and event.key in [pygame.K_q, pygame.K_ESCAPE]:
                    running=False
                    break
            if not running:
                break
        if not running:
            break

        captured = capture_image(CD, timeout=2.0)
        if captured is None:
            logger.warning(f"No image captured at iteration {i}, skipping iteration.")
            break
        bgr = cv2.cvtColor(captured, cv2.COLOR_BGRA2BGR)

        CURRENT_HASH = blake3_hash(CURRENT_HASH, bgr.tobytes())
        hf["emissions"][i]=emission_np
        hf["recordings"][i]=bgr
        hf["hashes"][i]=np.frombuffer(CURRENT_HASH,dtype='uint8')

        logger.info(f"Iteration {i}: Stored data and updated hash.")

    final_hash=CURRENT_HASH
    write_metadata(hf, fresh_blockhash, final_hash)

    if Config.DUMMY_RUN:
        logger.info("DUMMY_RUN=True, not submitting final hash to blockchain.")
    else:
        logger.info("Submitting final hash to RSK...")
        try:
            tx_hash=submit_hash_to_rsk(final_hash)
            logger.info(f"Final hash submitted. tx_hash={tx_hash.hex()}")
            hf.attrs["final_tx_hash"]=tx_hash.hex()
        except Exception as e:
            logger.error(f"Failed to submit final hash: {e}")

    hf.close()
    Tis.stop_pipeline()
    pygame.quit()
    logger.info("Recording process end.")

if __name__=="__main__":
    main()
