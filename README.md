# **The Truth Beam – README**

## **Introduction**

**The Truth Beam** is a proof-of-concept pipeline for securely recording scenes using a projector–camera loop, anchored by cryptographic hash chaining and (optionally) validated with a simple neural forensics module. Random seeds derived from a public blockchain (e.g., RSK) generate Perlin-based structured noise, ensuring each projected frame is cryptographically bound to its preceding recording and blockchain timestamp. A simple autoencoder–GAN can then flag inconsistent emission–capture pairs. The RSK key and blockchain submission are *optional*—if you prefer a purely local chain, you can skip those steps.

## **Workflow Overview**

1. **Set Up Configs**  
   - In **`config.py`**, specify crucial parameters:
     - **Camera serial number** (`CAMERA_SERIAL`) to match your hardware.
     - **Optional RSK private key** (`RSK_PRIVATE_KEY`) if you want final hashes posted on RSK. If you leave it as `0x...`, the script will skip that step or you can set `DUMMY_RUN=True`.
   - Adjust resolution, loop counts, and other parameters (e.g. for training, half-resolution sizes) as needed.

2. **Record Representative Datasets**  
   - Use **`secure_record.py`** to gather data:
     1. **Project** structured noise (seeded by blockchain or local hash).
     2. **Capture** frames via camera.
     3. **Chain** each frame’s hash so any tampering breaks continuity.
     4. (Optionally) **Submit** final hash to RSK if `DUMMY_RUN=False`.
   - This yields one or more HDF5 files in the `saved_datasets/` directory.  

3. **Train Neural Networks**  
   - Collect a **representative** set of these HDF5 datasets and place them in `saved_datasets` or in a custom `training_data` folder.
   - Run **`train_networks.py`** to:
     - Train an **Adversarial Autoencoder** (for reconstructing emission–recording pairs).
     - Train a **LatentGAN** on the autoencoder’s latent space, if desired (for advanced authenticity checks).
   - This produces trained model files (`aae_model_latest.pth`, etc.) in your current directory or whichever path you designate.

4. **Verify New / Novel Datasets**  
   - **Record** or obtain some new test data and place them in `novel_datasets/` (or specify them directly).
   - Run:
     1. **`verify_1_perlin.py`** to check the cryptographic chain and re-generate Perlin noise for MSE comparison (detect large mismatches).
     2. **`verify_2_recordings.py`** to see difference maps, authenticity heatmaps, etc. (if you have the AE + latent models trained).
   - Inspect the logs or CSV outputs for mismatch warnings, hash failures, or high error signals.

5. **Interpret Results**  
   - If everything is consistent:
     - The hash chain should be unbroken (`"OK"`).
     - Perlin MSE should remain small, indicating minimal distortion.
     - The AE or latent checks should not flag large anomalies.
   - Suspicious or tampered data typically fails these checks.

## **Requirements & Installation**

- **Python 3.8+**  
- Key packages: PyTorch, PyCUDA, OpenCV, Pillow, h5py, requests, web3, blake3, etc.  
- A GPU-compatible environment if you plan to use CUDA acceleration.
- **Camera & TIS.py**: If capturing live data, you need a camera from **The Imaging Source** or compatible drivers, plus [tiscamera’s `TIS.py`](https://github.com/TheImagingSource/tiscamera).
- **Optional RSK**: For anchoring final hashes publicly, set up an RSK node or use a public endpoint.

Example minimal install (not exhaustive):
```
pip install torch torchvision pycuda opencv-python pillow h5py requests web3 blake3
```
Place `TIS.py` from tiscamera’s GitHub into your Python path or environment as needed.

## **How It Works**

1. **Project + Capture**  
   - Perlin or structured noise is derived from a **blockhash** (or local seed).  
   - The projector displays this noise on the scene.  
   - A co-located camera captures each frame, producing `(emission, recording)` pairs.

2. **Chaining & Optional Blockchain**  
   - Each new capture is hashed with the previous hash, forming a chain that any tampering invalidates.  
   - The final hash can be submitted on RSK (or left local).

3. **Neural Forensics**  
   - The autoencoder tries to reconstruct valid `(emission, recording)` pairs.  
   - A latent-space GAN might help detect subtle forgeries or mismatches.  
   - Heatmaps or difference maps highlight anomalies in suspicious frames.

## **File-by-File**

1. **`config.py`**  
   Main parameters: **camera serial**, RSK node, private key, image sizes, loop counts, and training hyperparameters.

2. **`blockchain.py`**  
   Fetches a fresh blockhash from RSK, optionally posts final hash on-chain if you provide `RSK_PRIVATE_KEY`.

3. **`data_persistence.py`**  
   Creates HDF5 files to hold `(emissions, recordings, hashes)`. Writes metadata (initial blockhash, final hash, etc.).

4. **`models.py`**  
   PyTorch definitions of the **Adversarial Autoencoder**, a **Discriminator**, and optional **Latent Discriminator + Generator** for advanced checks.

5. **`perlin.py`**  
   GPU-accelerated Perlin noise using PyCUDA, seeded by the blockchain blockhash or cryptographic chain.

6. **`secure_record.py`**  
   Runs the actual recording loop, retrieving a blockhash, seeding noise, projecting, capturing frames, chaining the hashes, and optionally posting the final hash to RSK.

7. **`train_networks.py`**  
   Simplified script to train the autoencoder on `(emission, recording)` data, then train a latent GAN on the AE’s internal representations.

8. **`utils.py`**  
   Contains the Blake3 hashing function, image/tensor utilities, and private-key derivation logic.

9. **`verify_1_perlin.py`**  
   Verifies local hash chain continuity and compares stored emission with re-generated Perlin noise (MSE) to detect major tampering or mismatch.

10. **`verify_2_recordings.py`**  
   Uses the trained AE/latent models to produce difference maps and authenticity scores. If data is consistent, errors remain small; anomalies appear in the heatmaps.

---

**Truth Beam™** is trademarked, and the technology has patents pending. Patent licenses will be granted for research and personal use.
