import os
import time
import json
import base64
import logging
from tqdm import tqdm
from openai import OpenAI
from os.path import exists
from concurrent.futures import ThreadPoolExecutor, as_completed

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Global client instance
client = None

def safe_inference(prompt, model='gpt-4o', max_retries=5, retry_delay=2):
    """Execute inference with retry mechanism"""
    attempt = 0
    while attempt < max_retries:
        try:
            response = inference_one_step(prompt, [], model)
            return response
        except Exception as e:
            attempt += 1
            logger.error(f"Attempt {attempt} failed: {e}")
            if attempt < max_retries:
                logger.info(f"Waiting {retry_delay} seconds before retrying...")
                import time
                time.sleep(retry_delay)
                retry_delay += 2
            else:
                logger.error("Max retries reached. Skipping this item.")
                return "ERROR: Max retries reached."

def initialize_client(base_url="", api_key=""):
    """Initialize the OpenAI client globally"""
    global client
    client = OpenAI(base_url=base_url, api_key=api_key)

def encode_image(image_path):
    """Encode image to base64 string"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def inference_one_step(prompt, base64_images, model):
    """Perform inference with the given prompt and images"""
    if client is None:
        raise ValueError("Client not initialized. Call initialize_client() first.")
    
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [{
                    "type": "text",
                    "text": prompt
                }] + [{
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{base64_image}"
                    },
                } for base64_image in base64_images]
            },
        ],
    )
    return response.choices[0].message.content

def process_item_generic(item, img_root, model, prompt_builder, output_field, max_retries=5, retry_delay=2):
    """
    Generic process_item function that can be used for both caption and prediction tasks.
    
    Args:
        item: Dictionary containing item data
        img_root: Root directory for images
        model: Model name to use
        prompt_builder: Function to build the prompt (build_prompt_caption or build_prompt_prediction)
        output_field: Field name for the output (e.g., "description" or "prediction")
        max_retries: Maximum number of retries
        retry_delay: Delay between retries
    """
    index = item['index']

    try:
        # Handle different input structures for caption vs prediction
        if output_field == "description":
            # For caption: item has 'question' and 'image_path'
            question, image_paths = item['question'], item['image_path']
            prompt = prompt_builder(question)
        else:
            # For prediction: item has 'image_path' and prompt is built from full item
            image_paths = item['image_path']
            prompt = prompt_builder(item)
        
        base64_images = [encode_image(os.path.join(img_root, img_path)) for img_path in image_paths]

        attempt = 0
        response_content = None

        while attempt < max_retries:
            try:
                response_content = inference_one_step(prompt, base64_images, model)
                break  # Success
            except Exception as e:
                attempt += 1
                logger.error(f"Index {index} | Attempt {attempt}/{max_retries} failed: {e}")
                if attempt < max_retries:
                    time.sleep(retry_delay * attempt)  # Exponential backoff
                else:
                    logger.error(f"Index {index} | Max retries reached. Marking as error.")
                    response_content = f"ERROR: Max retries reached - {e}"
        
        # Return a new dictionary with the original item's data plus the new output
        return {**item, output_field: response_content}

    except Exception as e:
        logger.error(f"FATAL error processing item with index {index}: {e}", exc_info=True)
        # Return a structured error record, preserving the original item data
        return {**item, output_field: f"ERROR: Unrecoverable failure in processing pipeline: {e}"}

def run_inference_concurrent(
    json_path,
    output_path,
    img_root,
    model='gpt-4o',
    max_workers=4,
    prompt_builder=None,
    output_field="description"
):
    """
    Generic inference function that can be used for both caption and prediction tasks.
    
    Args:
        json_path: Path to input JSON file
        output_path: Path to output JSON file
        img_root: Root directory for images
        model: Model name to use
        max_workers: Number of concurrent workers
        prompt_builder: Function to build prompts
        output_field: Field name for output (e.g., "description" or "prediction")
    """
    # 1. Load the full dataset.
    # Assumes the input JSON is a list of objects, each with a unique 'index' key.
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            full_dataset = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logger.error(f"Failed to load or parse input file {json_path}: {e}")
        return

    # 2. Implement breakpoint resume capability.
    existing_results = []
    if exists(output_path):
        try:
            with open(output_path, 'r', encoding='utf-8') as f:
                existing_results = json.load(f)
        except json.JSONDecodeError:
            logger.warning(f"Output file {output_path} is corrupted. Starting fresh.")

    # Use the 'index' key from the data to identify completed items.
    completed_indices = {item.get('index') for item in existing_results if isinstance(item.get('index'), int)}
    items_to_process = [item for item in full_dataset if item.get('index') not in completed_indices]

    if not items_to_process:
        logger.info("All items have been processed according to the output file. Exiting.")
        return

    logger.info(f"Total items in dataset: {len(full_dataset)}")
    logger.info(f"Items already processed: {len(completed_indices)}")
    logger.info(f"Items remaining to process: {len(items_to_process)}")

    # 3. Process remaining items concurrently.
    new_results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_item = {
            executor.submit(process_item_generic, item, img_root, model, prompt_builder, output_field): item 
            for item in items_to_process
        }

        progress = tqdm(as_completed(future_to_item), total=len(items_to_process), desc="Processing Items")
        for future in progress:
            try:
                result = future.result()
                if result:
                    new_results.append(result)
            except Exception as e:
                item = future_to_item[future]
                logger.error(f"A task for index {item.get('index')} raised an unhandled exception: {e}")

    # 4. Merge, sort, and save results.
    if new_results:
        combined_results = existing_results + new_results

        # Sort by the pre-existing 'index' key to ensure original order.
        combined_results.sort(key=lambda x: x.get('index', float('inf')))

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(combined_results, f, ensure_ascii=False, indent=4)
        logger.info(f"Processing complete. Saved {len(combined_results)} total items to {output_path}.")
    else:
        logger.info("No new items were processed in this run.")