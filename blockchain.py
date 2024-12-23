import logging
import sys
from web3 import Web3
from config import Config
from utils import derive_address_from_private_key
import requests

logger = logging.getLogger("TruthBeam.Blockchain")

def check_rsk_connectivity():
    payload = {"jsonrpc":"2.0","method":"eth_blockNumber","params":[],"id":1}
    try:
        r = requests.post(Config.RSK_ENDPOINT, json=payload, timeout=5)
        if r.status_code == 200 and "result" in r.json():
            return True
        else:
            logger.error(f"Unexpected response from RSK endpoint: {r.status_code} {r.text}")
            return False
    except requests.exceptions.RequestException as e:
        logger.error(f"RSK connectivity check failed: {e}")
        return False

def submit_hash_to_rsk(final_hash):
    w3 = Web3(Web3.HTTPProvider(Config.RSK_ENDPOINT))
    from_address = derive_address_from_private_key(Config.RSK_PRIVATE_KEY)

    try:
        nonce = w3.eth.get_transaction_count(from_address)
    except Exception as e:
        logger.error(f"Failed to connect to RSK endpoint: {e}")
        logger.error("Check your internet connection or RSK endpoint URL in config.py.")
        sys.exit(1)

    data_hex = final_hash.hex()
    gas_price = Web3.to_wei('0.06', 'gwei')
    tx_for_estimate = {
        'nonce': nonce,
        'to': from_address,
        'value': 0,
        'gasPrice': gas_price,
        'data': '0x' + data_hex
    }

    try:
        estimated_gas = w3.eth.estimate_gas(tx_for_estimate)
        gas_limit = int(estimated_gas * 1.2)
    except Exception as e:
        logger.error(f"Gas estimation failed: {e}")
        sys.exit(1)

    tx = {
        'nonce': nonce,
        'to': from_address,
        'value': 0,
        'gas': gas_limit,
        'gasPrice': gas_price,
        'data': '0x' + data_hex
    }
    signed_tx = w3.eth.account.sign_transaction(tx, private_key=bytes.fromhex(Config.RSK_PRIVATE_KEY[2:]))
    try:
        tx_hash = w3.eth.send_raw_transaction(signed_tx.raw_transaction)
        logger.info(f"Submitted final hash to RSK. tx_hash={tx_hash.hex()}")
        return tx_hash
    except Exception as e:
        logger.error(f"Transaction submission failed: {e}")
        raise

def get_fresh_blockhash():
    if not check_rsk_connectivity():
        logger.error("Cannot reach RSK endpoint. Check your network or endpoint URL in config.py.")
        sys.exit(1)

    w3 = Web3(Web3.HTTPProvider(Config.RSK_ENDPOINT))
    logger.info("Getting initial blockhash from RSK 'latest' block...")
    try:
        latest_block = w3.eth.get_block('latest')
    except Exception as e:
        logger.error(f"Failed to fetch latest block from RSK: {e}")
        sys.exit(1)

    initial_blockhash = latest_block.hash
    logger.info(f"Initial blockhash: {initial_blockhash.hex()}")

    logger.info("Waiting for next RSK block to ensure freshness...")
    start = latest_block.number
    import time
    while True:
        try:
            block = w3.eth.get_block('latest')
        except Exception as e:
            logger.error(f"Failed to fetch block from RSK while waiting: {e}")
            sys.exit(1)

        if block.number > start:
            logger.info(f"New block found: {block.number}, hash={block.hash.hex()}")
            return block.hash
        time.sleep(1)
