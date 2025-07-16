import logging
from prompt import build_prompt_prediction
from utils import initialize_client, run_inference_concurrent

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


if __name__ == '__main__':
    # Define execution parameters
    MAX_CONCURRENT_WORKERS = 16
    INPUT_JSON_PATH = './outputs/total_caption.json'
    OUTPUT_JSON_PATH = './outputs/prediction.json'
    IMAGE_ROOT_DIR = 'images'
    MODEL_NAME = 'o3'

    # Initialize the client
    initialize_client(
        base_url="",
        api_key="",
    )

    # Run the main function
    run_inference_concurrent(
        json_path=INPUT_JSON_PATH,
        output_path=OUTPUT_JSON_PATH,
        img_root=IMAGE_ROOT_DIR,
        model=MODEL_NAME,
        max_workers=MAX_CONCURRENT_WORKERS,
        prompt_builder=build_prompt_prediction,
        output_field="prediction"
    )
