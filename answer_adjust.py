import json
import logging
import os
from prompt import build_answer_adjustment_prompt
from utils import initialize_client, safe_inference

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def extract_adjusted_answer(response):
    """Extract adjusted answer from response using XML-like tags."""
    import re
    pattern = r"<adjusted_answer>(.*?)</adjusted_answer>"
    match = re.search(pattern, response, re.DOTALL)
    if match:
        return match.group(1).strip()
    return response  # Return full response if tags not found

def process_item_adjustment(item, template_content, model):
    """Process a single item for answer format adjustment."""
    logger.info(f"Processing item {item.get('index', 'unknown')} for format adjustment")
    
    prompt = build_answer_adjustment_prompt(item, template_content)
    response = safe_inference(prompt, model)
    adjusted_answer = extract_adjusted_answer(response)
    
    # Update item with adjusted answer
    item['adjusted_answer'] = adjusted_answer
    item['original_answer'] = item.get('prediction', item.get('answer', ''))
    
    return item

def run_answer_adjustment(json_path, template_path, output_path, model):
    """Run answer format adjustment process."""
    # Load prediction data
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Load answer template
    with open(template_path, 'r', encoding='utf-8') as f:
        template_content = f.read()
    
    logger.info(f"Starting answer format adjustment for {len(data)} items")
    logger.info(f"Using template from: {template_path}")
    
    # Process each item
    adjusted_data = []
    for i, item in enumerate(data):
        logger.info(f"Processing item {i+1}/{len(data)}")
        try:
            adjusted_item = process_item_adjustment(item, template_content, model)
            adjusted_data.append(adjusted_item)
            
            # Save intermediate results after each item
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(adjusted_data, f, ensure_ascii=False, indent=4)
                
        except Exception as e:
            logger.error(f"Error processing item {item.get('index', i)}: {e}")
            # Add error item to maintain order
            item['error'] = str(e)
            adjusted_data.append(item)
    
    logger.info(f"Answer format adjustment completed. Results saved to {output_path}")
    return adjusted_data

if __name__ == '__main__':
    # Initialize the client
    initialize_client(
        base_url="",
        api_key="",
    )
    
    # Configuration
    model = 'o4-mini'
    input_path = './outputs/prediction_refined.json'
    template_path = './outputs/answer_template.txt'
    output_path = './outputs/prediction_refined_adjusted.json'
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    run_answer_adjustment(input_path, template_path, output_path, model)
