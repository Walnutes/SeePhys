import json
import logging
import os
from prompt import (
    build_refinement_prompt, 
    build_mathematical_accuracy_prompt,
    build_logical_flow_prompt,
    build_completeness_prompt
)
from utils import initialize_client, safe_inference

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def extract_solution_from_response(response, tag_name):
    """Extract solution from response using XML-like tags."""
    import re
    pattern = f"<{tag_name}>(.*?)</{tag_name}>"
    match = re.search(pattern, response, re.DOTALL)
    if match:
        return match.group(1).strip()
    return response  # Return full response if tags not found

def process_item_multi_step(item, model):
    """Process a single item through multiple refinement steps."""
    logger.info(f"Processing item {item.get('index', 'unknown')} through multi-step refinement")
    
    # Step 1: General refinement
    logger.info("Step 1: General refinement")
    prompt1 = build_refinement_prompt(item)
    response1 = safe_inference(prompt1, model=model)
    refined_reasoning = extract_solution_from_response(response1, "refined_reasoning")
    
    # Update item with refined reasoning
    item['refined_reasoning'] = refined_reasoning
    
    # Step 2: Mathematical accuracy check
    logger.info("Step 2: Mathematical accuracy check")
    prompt2 = build_mathematical_accuracy_prompt(item)
    response2 = safe_inference(prompt2, model=model)
    corrected_solution = extract_solution_from_response(response2, "corrected_solution")
    
    # Update item with corrected solution
    item['mathematically_corrected_reasoning'] = corrected_solution
    
    # Step 3: Logical flow improvement
    logger.info("Step 3: Logical flow improvement")
    prompt3 = build_logical_flow_prompt(item)
    response3 = safe_inference(prompt3, model=model)
    improved_solution = extract_solution_from_response(response3, "improved_solution")
    
    # Update item with improved solution
    item['logically_improved_reasoning'] = improved_solution
    
    # Step 4: Completeness check
    logger.info("Step 4: Completeness check")
    prompt4 = build_completeness_prompt(item)
    response4 = safe_inference(prompt4, model=model)
    complete_solution = extract_solution_from_response(response4, "complete_solution")
    
    # Update item with complete solution
    item['final_refined_reasoning'] = complete_solution
    
    # Store all intermediate results
    item['refinement_steps'] = {
        'step1_general_refinement': response1,
        'step2_mathematical_accuracy': response2,
        'step3_logical_flow': response3,
        'step4_completeness': response4
    }
    
    return item

def run_multi_step_refinement(json_path, output_path, model):
    """Run multi-step refinement process."""
    # Load data
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    logger.info(f"Starting multi-step refinement for {len(data)} items")
    
    # Process each item through all refinement steps
    refined_data = []
    for i, item in enumerate(data):
        logger.info(f"Processing item {i+1}/{len(data)}")
        try:
            refined_item = process_item_multi_step(item, model)
            refined_data.append(refined_item)
            
            # Save intermediate results after each item
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(refined_data, f, ensure_ascii=False, indent=4)
                
        except Exception as e:
            logger.error(f"Error processing item {item.get('index', i)}: {e}")
            # Add error item to maintain order
            item['error'] = str(e)
            refined_data.append(item)
    
    logger.info(f"Multi-step refinement completed. Results saved to {output_path}")
    return refined_data

if __name__ == '__main__':
    # Initialize the client
    initialize_client(
        base_url="",
        api_key="",
    )
    
    # Configuration
    model = 'o4-mini'
    input_path = './outputs/prediction.json'
    output_path = './outputs/prediction_refined.json'
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    run_multi_step_refinement(input_path, output_path, model)
