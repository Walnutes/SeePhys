def build_prompt_prediction(item):
    question = item['question']
    description = item['description']

    prompt = '''
You are an expert Physics Problem Solver and Educator. Your task is to solve a physics problem based on a structured description of its visual and textual components. You must not only find the correct answer but also present your solution in a clear, logical, and pedagogically sound manner that demonstrates a deep understanding of the underlying principles.
You will be provided with the structured **Image(s)**, **Image Description** and the **Question**.

**Format your output as follows:**
- Step-by-Step Solution
- Final Answer

---
### **Final Answer Formatting Guide**

The format of the content inside `\\boxed{}` must match the type of answer demanded by the question.
*   **For a Single Value:** Place the single number, symbol, or formula in the box.
    *   Example: `\\boxed{4.9 \\, \\text{m/s}^2}` or `\\boxed{\\frac{m h}{M+m}\\cot\\theta}`

*   **For Multi-Part Answers:** If the question asks for several quantities (e.g., "the spins and parities," or "the force and torque"), list them separated by commas within a *single* `\\boxed{}`.
    *   Example: `\\boxed{-1.91 \\mu_{N}, 5.79 \\mu_{N}, -1.14 \\times 10^{-25} \\mathrm{~cm}^{2}}`

*   **For Conditional Answers:** If the answer depends on different conditions or cases, write out the cases explicitly inside a *single* `\\boxed{}`.
    *   Example: `\\boxed{\\text{Case (i): For } ka < \\frac{Mg}{2} \\text{ the equilibrium is at } \\theta=0 \\text{ (unstable) and } \\theta=\\pi \\text{ (stable). Case (ii): ...}}`

*   **For Piecewise Functions:** Use the LaTeX `cases` environment inside the `\\boxed{}`.
    *   Example: `\\boxed{V(r)=\\begin{cases} \\frac{Q}{4\\pi a(\\alpha a+1)}, & r \\le a \\\\ \\frac{Qe^{\\alpha a}e^{-\\alpha r}}{4\\pi(\\alpha a+1)r}, & r > a \\end{cases}}`

*   **For Qualitative Answers:** Place the concise, definitive text inside the box.
    *   Example: `\\boxed{\\text{Bulbs 1 and 3 only}}`

*   **For Vector Answers:** Use standard vector notation.
    *   Example: `\\boxed{\\mathbf{J}_{EM}=\\frac{qB_{0}(b^{2}-a^{2})}{2}\\,\\mathbf{e}_{z}}`

---
**Overall Formatting Rules:**
*   All mathematical formulas must be wrapped in SINGLE dollar signs (`$...$`).
*   All LaTeX special characters inside the dollar signs MUST be escaped with TWO backslashes (e.g., `\\theta`, `\\frac`).

---
    '''
    prompt += '\n Image Description: ' + description
    prompt += '\n Question: ' + question

    if item['sig_figs']:
        sf = str(int(item['sig_figs']))
        prompt += f"\n The final answer MUST retain {sf} significant figures."

    return prompt

def build_prompt_caption(init_question):
    caption_prompt = '''
You are a meticulous Physics Data Annotation Specialist. Your primary mission is to deconstruct multimodal physics problems (consisting of images and text) and translate them into a highly structured and comprehensive natural language description. The goal is to create a "golden" reference text that is as unambiguous and detailed as a data file, which will be used to evaluate the accuracy of other AI models. Your adherence to the format described below is critical.
You will be provided with a physics problem that consists of up to two parts: One or more **images**, and its corresponding **question text**.

### **Guiding Principles for Analysis:**
1.  **Category-First, Structure-Always:** Your entire analysis begins with correctly identifying the image's category. This category dictates the focus of your description. You must then follow the specified markdown structure precisely for your output.
2.  **Separate What is Seen from What is Inferred:** Your description must maintain a strict separation between elements explicitly visible in the diagram and properties inferred from the accompanying text (e.g., "frictionless"). The output format has dedicated sections for this.
3.  **Comprehensive and Atomic Breakdown:** Every single element in the diagram—objects, surfaces, vectors, labels, points on a graph, etc.—must be identified and described individually within the "Component Breakdown" section.
4.  **Holistic Synthesis:** The image and question text are a single unit. Use the text to define labels, understand the scenario, and extract all inferred properties.

---
### **Instructions for Structuring Your Output**
You must generate a single text block. The response must be structured using markdown with headings, bolded keywords, and bullet points exactly as specified below. For each image provided, create a complete descriptive block starting with `### Image [N]: [Category]`.

#### **Required Output Structure:**

**### Image [N]: [Primary Category Name]**
*(Replace [N] with the image number, and [Primary Category Name] with the category you identify from the list below.)*

**Scene Summary:** A single, concise sentence that describes the overall purpose and content of the diagram.

**Explicit Component Breakdown:**
*(This section is for **visible elements only**.)*
*   **[Component Name] (`[label]`):** A description of the component. The `[label]` should be the exact text or symbol labeling the component in the diagram. If there is no label, use `None`.
*   *(Repeat for every single visible component: objects, vectors, surfaces, axes, points, etc.)*

**Interactions and Relationships:**
*(This section describes how the explicit components are connected and arranged.)*
*   Describe the spatial and physical connections between components (e.g., "The block `m_1` is connected to the block `m_2` via the string.").
*   Describe the topological layout for circuits (e.g., "Resistor `R_1` is in series with the parallel branch containing `R_2` and `R_3`.").
*   Trace the path of rays for optics or describe the shape of curves for graphs.

**Implicit and Inferred Properties:**
*(This section is **only** for information derived from the question text or standard physics conventions, not explicitly drawn.)*
*   **[Component or System Name]:** [Inferred Property]. (e.g., **Inclined Plane:** The surface is frictionless.)
*   **[Component or System Name]:** [Inferred Property]. (e.g., **Connecting String:** Assumed to be massless and inextensible.)
*   *(List every piece of non-visual information.)*

**Identified Ambiguities:**
*(If any part of the image is illegible or its meaning is unclear even with context, list it here. If none, state "None.")*
*   [Description of ambiguous element].

---
### **Reference Guide: Image Categories**

*   **Mechanics Diagram:** Problems involving forces, motion, energy, and momentum (e.g., blocks, planes, pulleys, springs, pendulums).
*   **Free-Body Diagram:** An isolated diagram showing all force vectors acting on a single object.
*   **Circuit Diagram:** A diagram of an electrical circuit, including components like resistors, capacitors, inductors, and power sources.
*   **Data Plot / Graph:** A graphical representation of data, such as a velocity-time graph or a stress-strain curve.
*   **Ray Optics Diagram:** A diagram showing light rays interacting with lenses, mirrors, or other optical elements.
*   **Field Diagram:** A diagram illustrating a vector field, such as an electric or magnetic field.
*   **Thermodynamic Diagram:** A plot representing thermodynamic states and processes, such as a P-V diagram.

---
**Final Formatting Rules:**
*   Your entire output must be a single response.
*   All mathematical formulas or symbols must be wrapped in SINGLE dollar signs (e.g., `$m_1$`).
*   All LaTeX special characters inside the dollar signs MUST be escaped with TWO backslashes (e.g., `$\\theta$`).
*   Do not add any introductory or concluding text outside of the prescribed format.

Now, analyze the provided image(s) and question text, and generate the structured natural language description following this category-adaptive format.
-----
\n Original question: 
'''
    final_question = caption_prompt + init_question
    return final_question   

def build_template_analysis_prompt(items):
    """Build prompt for analyzing answer templates."""
    prompt = f"""I will provide you with {len(items)} question-answer pairs from a physics dataset. Please analyze these pairs and identify the most representative examples that demonstrate common answer patterns and formats.

For each pair, I will show:
Question: [the question]
Answer: [the answer]

Please analyze these pairs and:
1. Identify the most representative examples that show common answer patterns
2. Explain why each selected example is representative
3. Describe the key format patterns in the answers
4. Provide a template that captures the common structure of these answers

Here are the question-answer pairs:

"""
    
    for i, item in enumerate(items, 1):
        prompt += f"\nPair {i}:\n"
        prompt += f"Question: {item['question']}\n"
        prompt += f"Answer: {item['answer']}\n"
    
    prompt += "\nPlease analyze these pairs and provide your findings in the following format:\n"
    prompt += "1. Representative Examples:\n"
    prompt += "   - Example 1: [question number] - [brief explanation of why it's representative]\n"
    prompt += "2. Common Patterns:\n"
    prompt += "   - Pattern 1: [description]\n"
    prompt += "3. Answer Template:\n"
    prompt += "   [template structure]\n"
    
    return prompt

def build_final_analysis_prompt(batch_results):
    """Build prompt for final analysis combining all batch results."""
    prompt = """I will provide you with the analysis results from multiple batches of question-answer pairs. Please combine these analyses to create a comprehensive template that captures the most important patterns across all cases.

Here are the batch analyses:

"""
    
    for i, result in enumerate(batch_results, 1):
        prompt += f"\nBatch {i} Analysis:\n{result}\n"
    
    prompt += "\nPlease provide a final comprehensive analysis that:\n"
    prompt += "1. Identifies the most representative examples across all batches\n"
    prompt += "2. Combines and prioritizes the common patterns found\n"
    prompt += "3. Creates a unified template that captures the most important aspects of all answer formats\n"
    prompt += "4. Highlights any patterns that are particularly important for accuracy\n\n"
    prompt += "Please format your response in the same structure as the individual analyses."
    
    return prompt

def build_refinement_prompt(item):
    """Build prompt for refining reasoning."""
    question = item['question']
    image_description = item['image_description'][0] if isinstance(item['image_description'], list) else item['image_description']
    caption = item['caption']
    reasoning = item['reasoning']
    
    prompt = f"""Please analyze and refine the following reasoning for a physics problem:

Question: {question}

Image Description: {image_description}

Caption: {caption}

Current Reasoning: {reasoning}

Please check for any obvious errors, inconsistencies, or areas that could be improved. Consider:
1. Mathematical accuracy
2. Logical flow
3. Clarity of explanation
4. Completeness of solution
5. Proper use of physics principles

Provide your refined reasoning in the same format as the original, maintaining all mathematical expressions and step numbers. If you find no issues, explain why the reasoning is sound.

Please format your response with:
1. A brief analysis of the current reasoning
2. The refined reasoning (if changes are needed)
3. Any additional insights or clarifications

Output your response in <analysis> </analysis> tags followed by <refined_reasoning> </refined_reasoning> tags."""

    return prompt

def build_mathematical_accuracy_prompt(item):
    """Build prompt for checking mathematical accuracy specifically."""
    question = item['question']
    image_description = item['image_description'][0] if isinstance(item['image_description'], list) else item['image_description']
    reasoning = item.get('refined_reasoning', item.get('reasoning', ''))
    
    prompt = f"""You are a mathematical physics expert. Please carefully check the mathematical accuracy of the following physics problem solution:

Question: {question}

Image Description: {image_description}

Current Solution: {reasoning}

Focus specifically on:
1. **Mathematical correctness**: Check all calculations, formulas, and numerical operations
2. **Unit consistency**: Verify that all units are consistent throughout the solution
3. **Significant figures**: Ensure proper handling of significant figures
4. **Algebraic manipulations**: Check for any algebraic errors
5. **Physical constants**: Verify correct use of physical constants
6. **Dimensional analysis**: Ensure dimensional consistency

If you find any mathematical errors, provide the corrected solution. If the mathematics is correct, explain why.

Output your response in <mathematical_analysis> </mathematical_analysis> tags followed by <corrected_solution> </corrected_solution> tags."""

    return prompt

def build_logical_flow_prompt(item):
    """Build prompt for improving logical flow and clarity."""
    question = item['question']
    image_description = item['image_description'][0] if isinstance(item['image_description'], list) else item['image_description']
    reasoning = item.get('refined_reasoning', item.get('reasoning', ''))
    
    prompt = f"""You are a physics education expert. Please analyze and improve the logical flow and clarity of the following physics problem solution:

Question: {question}

Image Description: {image_description}

Current Solution: {reasoning}

Focus specifically on:
1. **Logical progression**: Ensure each step follows logically from the previous one
2. **Clarity of explanation**: Make explanations clear and accessible
3. **Step-by-step structure**: Ensure the solution is well-organized
4. **Physical intuition**: Include physical reasoning where appropriate
5. **Assumptions**: Clearly state any assumptions made
6. **Conclusion**: Ensure the final answer is clearly presented

Provide an improved version that maintains all mathematical accuracy while enhancing clarity and logical flow.

Output your response in <flow_analysis> </flow_analysis> tags followed by <improved_solution> </improved_solution> tags."""

    return prompt

def build_completeness_prompt(item):
    """Build prompt for checking solution completeness."""
    question = item['question']
    image_description = item['image_description'][0] if isinstance(item['image_description'], list) else item['image_description']
    reasoning = item.get('refined_reasoning', item.get('reasoning', ''))
    
    prompt = f"""You are a comprehensive physics problem solver. Please check if the following solution addresses all aspects of the problem:

Question: {question}

Image Description: {image_description}

Current Solution: {reasoning}

Focus specifically on:
1. **Problem requirements**: Ensure all parts of the question are addressed
2. **Physical principles**: Verify that all relevant physics principles are applied
3. **Boundary conditions**: Check if boundary conditions are properly considered
4. **Special cases**: Consider if any special cases need to be addressed
5. **Units and dimensions**: Ensure final answer has correct units
6. **Physical interpretation**: Provide physical meaning of the result

If the solution is incomplete, provide a complete version. If it's complete, explain why it addresses all aspects.

Output your response in <completeness_analysis> </completeness_analysis> tags followed by <complete_solution> </complete_solution> tags."""

    return prompt

def build_answer_adjustment_prompt(item, template_content):
    """Build prompt for adjusting answer format based on template."""
    question = item['question']
    original_answer = item.get('prediction', item.get('answer', ''))
    
    prompt = f"""You are a physics answer formatting specialist. Your task is to adjust the format of a physics answer to match a specific template while preserving the core content and mathematical accuracy.

Question: {question}

Original Answer: {original_answer}

Answer Template (Reference Format): {template_content}

Please adjust the format of the original answer to match the template's style and structure. Focus on:

1. **Formatting consistency**: Match the template's formatting style
2. **Mathematical notation**: Use consistent mathematical notation as shown in the template
3. **Step structure**: Follow the template's step-by-step structure
4. **Final answer presentation**: Format the final answer according to the template
5. **Language style**: Match the template's language and explanation style

**Important**: Do NOT change the core mathematical content, calculations, or final numerical result. Only adjust the formatting, presentation, and style to match the template.

Provide the adjusted answer in <adjusted_answer> </adjusted_answer> tags."""

    return prompt

