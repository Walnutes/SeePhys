import json
import logging
from prompt import build_template_analysis_prompt, build_final_analysis_prompt
from utils import initialize_client, safe_inference

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def process_batch(batch_data, batch_num, total_batches, model):
    """Process a single batch of data."""
    prompt = build_template_analysis_prompt(batch_data, batch_num, total_batches)
    return safe_inference(prompt, model)  

if __name__ == '__main__':
    # Initialize the client
    initialize_client(
        base_url="",
        api_key=" ",
    )
    
    with open("./dev.json", 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Configuration
    batch_size = 25  # Process 25 items at a time
    model = 'o4-mini'
    total_batches = (len(data) + batch_size - 1) // batch_size
    
    # Process data in batches
    batch_results = []
    for i in range(0, len(data), batch_size):
        batch_num = (i // batch_size) + 1
        batch_data = data[i:i + batch_size]
        logger.info(f"Processing batch {batch_num}/{total_batches}")
        
        result = process_batch(batch_data, batch_num, total_batches, model)
        batch_results.append(result)
        
        # Save intermediate results
        intermediate_path = f"./outputs/answer_template_split_{batch_num}.txt"
        with open(intermediate_path, 'w', encoding='utf-8') as f:
            f.write(result)
    
    # Combine all batch results
    logger.info("Combining batch results...")
    final_prompt = build_final_analysis_prompt(batch_results)
    final_analysis = safe_inference(final_prompt, model)
    
    # Save final template
    output_path = "./outputs/answer_template.txt"
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(final_analysis)
    
    print(f"Template has been generated and saved to {output_path}")
