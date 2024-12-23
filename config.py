class Config:
    #---------------------------------------------------------
    # Critical Parameters for Secure Recording and Blockchain
    #---------------------------------------------------------
    CAMERA_SERIAL = "..."  # Camera serial number
    RSK_ENDPOINT = "https://public-node.rsk.co"
    RSK_PRIVATE_KEY = "0x..."

    #---------------------------------------------------------
    # Recording and Camera Settings
    #---------------------------------------------------------
    GAIN_VALUE = 24
    EXPOSURE_TIME = 16667
    EMISSION_WIDTH = 1920
    EMISSION_HEIGHT = 1080
    RECORDING_WIDTH = 5320
    RECORDING_HEIGHT = 4600
    LOOP_COUNT = 5
    LATENCY_PERIOD = 2.4
    WARMUP_TIME = 3.0
    DUMMY_RUN = True

    # Directories for Data and Outputs
    OUTPUT_DIR = "saved_datasets"
    NOVEL_DATASETS_DIR = "novel_datasets"
    TRAINING_OUTPUTS_DIR = "training_outputs"
    MODEL_DIR = "models"
    SAMPLES_DIR = "samples"

    # Verification Parameters
    VERIFICATION_PARANOIA = 1.0
    SIMILARITY_THRESHOLD = 0.95
    DISCRIMINATOR_THRESHOLD = 0.5

    #---------------------------------------------------------
    # Perlin Noise Parameters (For Emission Generation)
    #---------------------------------------------------------
    PERLIN_BLOCK_SIZE = (32,32,1)
    PERLIN_OFFSET_RANGE = 50.0
    PERLIN_SCALE_MODULUS = 2000
    PERLIN_SCALE_DIVISOR = 100000.0
    PERLIN_SCALE_MIN = 0.005

    #---------------------------------------------------------
    # Half-Resolution Parameters (For ML Training and Verification)
    #---------------------------------------------------------
    HALF_EMISSION_WIDTH = EMISSION_WIDTH // 2
    HALF_EMISSION_HEIGHT = EMISSION_HEIGHT // 2
    HALF_RECORDING_WIDTH = RECORDING_WIDTH // 2
    HALF_RECORDING_HEIGHT = RECORDING_HEIGHT // 2

    #---------------------------------------------------------
    # Adversarial Autoencoder Training Parameters
    #---------------------------------------------------------
    EPOCHS = 200
    BATCH_SIZE = 6
    LEARNING_RATE = 1e-4
    MODEL_SAVE_EVERY = 5
    SAMPLE_OUTPUT_EVERY = 5
    ADV_WEIGHT = 1.0
    RECON_WEIGHT = 1.0

    #---------------------------------------------------------
    # Latent GAN (Bonus) Parameters
    #---------------------------------------------------------
    LATENT_GAN_EPOCHS = 200
    LATENT_GAN_LR = 2e-4
    LATENT_GAN_BATCH_SIZE = 4
    NOISE_DIM = 128
    LATENT_GAN_MODEL_SAVE_EVERY = 5
    LATENT_GAN_SAMPLE_OUTPUT_EVERY = 5

    #---------------------------------------------------------
    # Stabilization and Loss Weights
    #---------------------------------------------------------
    USE_GRADIENT_PENALTY = True
    GP_LAMBDA = 10.0
    REAL_LABEL = 0.9
    FAKE_LABEL = 0.1

