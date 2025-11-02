import pymupdf4llm
from litellm import completion, batch_completion


PROMPT = """You are a materials science expert. Your task is to extract ONLY the explicitly stated synthesis information from the provided research paper. Do not generate, assume, or infer any information not directly presented in the paper.

## Key Contribution
Summarize the key contributions of the paper:
- Novel materials or compounds: <summary>
- Unique synthesis methods: <summary>
- Specific applications or domains: <summary>

## Target Materials
Extract only the chemical formulas of the target materials (the main materials that were synthesized, not intermediate products or precursors). If multiple target materials exist, separate them with commas.
Examples: "SiC", "Al2O3, TiO2", "Li2FeSiO4"
If no clear target materials can be identified, respond with "Not specified".

## Precursors
Extract only the chemical formulas of all precursor materials (starting materials, reactants) used in the synthesis. Do not include solvents, catalysts, or final products.
Examples: "Al2O3, TiO2", "Li2CO3, FeSO4, SiO2", "ZnO"
If no clear precursors can be identified, respond with "Not specified".

## Recipe
Convert the synthesis process into numbered steps using as many steps as needed, but do not exceed 12 steps maximum. Each step should be exactly one sentence that describes a specific action or process. Include all important parameters (temperatures, times, amounts, ratios) exactly as mentioned in the original text, but convert all temperatures to Celsius (°C) if they are given in Kelvin.
IMPORTANT: Focus only on the synthesis steps. Do not include characterization, analysis, or evaluation steps such as XRD, TEM, SEM, EDX, FTIR, Raman spectroscopy, mechanical testing, electrical measurements, or any other analytical procedures. The recipe should end when the material synthesis is complete.
For complex processes that would require more than 12 steps, combine similar or sequential actions into single steps.
Temperature conversion examples:
- "Heat at 973K" → "Heat at 700°C"
- "Heat at 800°C" → "Heat at 800°C" (already in Celsius)
Format:
Step 1: [synthesis action with specific conditions]
Step 2: [synthesis action with specific conditions]
Step 3: [synthesis action with specific conditions]
...
(Use as many steps as needed for synthesis only, maximum 12 steps)
Please provide only the numbered synthesis steps. Do not include any characterization or analysis steps.


IMPORTANT RULES:
1. DO NOT generate or assume any missing information
2. If specific details are not mentioned in the paper, indicate "N/A"
3. Use exact numbers and units as presented in the paper
4. Maintain original measurement units
5. Quote unusual or specific procedures directly when necessary
6. Format all information using proper markdown with headers (##) and bullet points

Remember: Accuracy and authenticity are crucial. Only include information explicitly stated in the paper."""


import fitz  # PyMuPDF

def pdf_bytes_to_markdown(pdf_bytes):
    # Create a PyMuPDF document from bytes
    pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")
    
    # Convert to markdown
    md_text = pymupdf4llm.to_markdown(pdf_document)
    
    # Close the document to free resources
    pdf_document.close()
    
    return md_text

def extract_recipe_from_text(texts, model="gpt-4o-2024-11-20"):
    def filter_text(text):
        if len(text) < 100:
            return None
        if len(text) > 50000:
            text = text[:50000]
        return text
    
    texts = [filter_text(text) for text in texts]
    messages = [[
        {"role": "system", "content": PROMPT},
        {"role": "user", "content": "Scientific Paper:\n" + text},
    ] for text in texts if text is not None]

    messages = batch_completion(
        model=model,
        messages=messages,
        max_tokens=4096,
        temperature=0.6,
    )
    return [message.choices[0].message.content for message in messages]

def read_pdf(pdf_file):
    with open(pdf_file, "rb") as f:
        pdf_bytes = f.read()
        text = pdf_bytes_to_markdown(pdf_bytes)
        return text
    
def pdf_bytelist_to_recipes(pdf_bytelist, model="gpt-4o-2024-11-20"):
    texts = [pdf_bytes_to_markdown(pdf_bytes) for pdf_bytes in pdf_bytelist]
    return extract_recipe_from_text(texts, model=model)

if __name__ == "__main__":
    pdf_files = ["test.pdf", "test.pdf"]
    texts = [read_pdf(pdf_file) for pdf_file in pdf_files]
    texts = extract_recipe_from_text(texts, model="gpt-4o-mini")
    print("\n\nExtracted Recipes:\n")
    for i, text in enumerate(texts):
        print(f"Recipe {i + 1}:\n{text}\n\n")