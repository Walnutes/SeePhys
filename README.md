# Solution for ICML 2025 AI for Math Workshop & Challenge 2-SeePhys

This repository contains our solution for the [ICML 2025 AI for Math Workshop&Challenge 2](https://www.codabench.org/competitions/7925/).

## 🚀 Quick Start

### Environment Setup

```bash
pip install openai tqdm
```

### API Configuration

Set your API credentials in each script:

```python
initialize_client(
    base_url="YOUR_API_BASE_URL",
    api_key="YOUR_API_KEY",
)
```

## 🔄 Core Workflow


```
[total.json] → caption.py → [total_caption.json] → prediction.py → [prediction.json]
                                    ↓
(Optional) refine.py → [prediction_refined.json]
                                    ↓
(Optional) [dev.json] → answer_template.py → [answer_template.txt] → answer_adjust.py → [prediction_refined_adjusted.json]
```

## 🏗️ Main Steps

1. **Generate Descriptions** (`caption.py`): Analyze problems and creates structured descriptions
2. **Generate Solutions** (`prediction.py`): Use descriptions to generate final answers
3. **(Optional) Multi-Step Refinement** (`refine.py`): Four-step process to improve solution quality
4. **(Optional) Template Generation** (`answer_template.py`): Analyze patterns for formatting templates
5. **(Optional) Format Adjustment** (`answer_adjust.py`): Standardize answer formats

---

*For detailed implementation and configuration options, please refer to the individual script files.*
