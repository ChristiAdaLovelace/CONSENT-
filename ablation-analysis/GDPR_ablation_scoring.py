# ---------------------------------------------------------------------------------------
#pip install -q torch transformers==4.36.0 accelerate bitsandbytes pandas openpyxl
import pandas as pd
import torch
import re
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from tqdm import tqdm
from google.colab import files
# ---------------------------------------------------------------------------------------
MODEL_ID = "microsoft/Phi-3-mini-4k-instruct"
OUTPUT_FILE = "gdpr_analyzed_output.xlsx"
COLUMN_NAME = "final_form_output"
SAMPLE_SIZE = 250 
print("\n Upload Excel File Including 250 automated test cases.")
uploaded = files.upload()
INPUT_FILE = list(uploaded.keys())[0]
print(f" File uploaded: {INPUT_FILE}\n")
# ---------------------------------------------------------------------------------------
def setup_llm():
    print(f" Loading {MODEL_ID}...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        trust_remote_code=True,
        device_map="auto",
        attn_implementation="eager"
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer
def analyze_gdpr_direct(text, model, tokenizer):
    """Direct generation without pipeline to avoid cache issues"""
    text = text[:1500] if len(text) > 1500 else text
    prompt = f"""<|user|>
You are a GDPR compliance expert. Rate this consent form's GDPR compliance from 0-100%.
CONSENT FORM:
{text}
Evaluate: legal basis, data subject rights, purpose limitation, security measures.
Respond EXACTLY in this format:
SCORE: [number]%
EXPLANATION: [brief explanation]<|end|>
<|assistant|>"""

    try:
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=200,
                temperature=0.3,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                use_cache=False
            )
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        if "<|assistant|>" in response:
            response = response.split("<|assistant|>")[-1].strip()
        score = None
        score_patterns = [
            r'SCORE:\s*(\d+)%?',
            r'Score:\s*(\d+)%?',
            r'(\d+)%',
        ]
        for pattern in score_patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                potential_score = int(match.group(1))
                if 0 <= potential_score <= 100:
                    score = potential_score
                    break
        explanation_match = re.search(r'EXPLANATION:\s*(.+)', response, re.IGNORECASE | re.DOTALL)
        if explanation_match:
            explanation = explanation_match.group(1).strip()
        else:
            explanation = response
        explanation = re.sub(r'SCORE:\s*\d+%?\s*', '', explanation, flags=re.IGNORECASE)
        explanation = ' '.join(explanation.split())[:500]
        if score is None:
            numbers = re.findall(r'\b(\d+)\b', response)
            for num in numbers:
                if 0 <= int(num) <= 100:
                    score = int(num)
                    break
        if score is None:
            return None, f"Could not extract score from: {response[:200]}"
        return score, explanation
    except Exception as e:
        return None, f"Error: {str(e)}"
# ---------------------------------------------------------------------------------------
print("Loading Data")
try:
    df = pd.read_excel(INPUT_FILE)
    print(f"✓ Loaded {len(df)} rows\n")
    df_sample = df.head(SAMPLE_SIZE).copy()
    if COLUMN_NAME not in df_sample.columns:
        print(f"Available columns: {df_sample.columns.tolist()}")
        exit()
    print(f"Using column: '{COLUMN_NAME}'")
except Exception as e:
    print(f"Error: {e}")
    exit()
try:
    model, tokenizer = setup_llm()
    print("Model loaded!\n")
except Exception as e:
    print(f"✗ Failed: {e}")
    exit()
# ---------------------------------------------------------------------------------------
print("\nAnalyzing Forms")
scores = []
explanations = []
success_count = 0
error_count = 0
skip_count = 0
for index, row in tqdm(df_sample.iterrows(), total=len(df_sample), desc="Processing"):
    try:
        text_content = str(row[COLUMN_NAME])
        if pd.isna(text_content) or len(text_content.strip()) < 100:
            scores.append(None)
            explanations.append("Skipped: Text too short")
            skip_count += 1
            continue
        score, explanation = analyze_gdpr_direct(text_content, model, tokenizer)
        if score is None:
            scores.append(None)
            explanations.append(explanation)
            error_count += 1
        else:
            scores.append(score)
            explanations.append(explanation)
            success_count += 1
        if (len(scores)) % 5 == 0:
            valid_scores = [s for s in scores if s is not None]
            if valid_scores:
                print(f"\n  ✓ {len(scores)}/{len(df_sample)} | Current avg: {sum(valid_scores)/len(valid_scores):.1f}%")
    except Exception as e:
        scores.append(None)
        explanations.append(f"Error: {str(e)}")
        error_count += 1
# ---------------------------------------------------------------------------------------
df_sample['llm_compliance_score'] = scores
df_sample['llm_explanation'] = explanations
valid_scores = [s for s in scores if s is not None]
print("/nRESULTS")
print(f"Success: {success_count}")
print(f"Errors: {error_count}")
print(f"Skipped: {skip_count}")
if valid_scores:
    print(f"\nCOMPLIANCE METRICS:")
    print(f"   Average Score: {sum(valid_scores)/len(valid_scores):.1f}%")
    print(f"   Min: {min(valid_scores)}% | Max: {max(valid_scores)}%")
    print(f"   Median: {sorted(valid_scores)[len(valid_scores)//2]}%")
    print(f"   Std Dev: {pd.Series(valid_scores).std():.1f}%")
    print(f"\nSAMPLE RESULTS:")
    for i in range(min(3, len(df_sample))):
        if scores[i] is not None:
            print(f"\nForm #{i+1}:")
            print(f"  Score: {scores[i]}%")
            print(f"  Explanation: {explanations[i][:150]}...")
else:
    print("\nNo valid scores generated!")
# ---------------------------------------------------------------------------------------
try:
    df_sample.to_excel(OUTPUT_FILE, index=False)
    print(f"\n{'='*80}")
    print(f"Saved: {OUTPUT_FILE}")
    files.download(OUTPUT_FILE)
    print("Downloaded!")
except Exception as e:
    print(f"Save error: {e}")
  # ---------------------------------------------------------------------------------------
