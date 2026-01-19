# --------------------------------------------------------------------
# pip install gradio_client pandas huggingface_hub matplotlib faker
# pip install -q transformers accelerate bitsandbytes
# --------------------------------------------------------------------
"""
FormFlow Stress Test - Real LLM Data Generation
Uses actual LLM to generate realistic field values
"""
import os
import importlib.util
TEMPLATE_DIR = "templates"
if not os.path.exists(TEMPLATE_DIR):
    os.makedirs(TEMPLATE_DIR)
try:
    from google.colab import files
    print("Please upload your template .py files.")
    uploaded = files.upload()
    for filename in uploaded.keys():
        os.rename(filename, os.path.join(TEMPLATE_DIR, filename))
    print("Files uploaded successfully.")
except ImportError:
    print("ERROR.")
TEMPLATES = {}
print("\nLoading templates...")
for filename in os.listdir(TEMPLATE_DIR):
    if filename.endswith(".py") and not filename.startswith("__"):
        category_name = filename.replace(".py", "").replace("_", " ").title()
        file_path = os.path.join(TEMPLATE_DIR, filename)

        # Dynamic module import
        spec = importlib.util.spec_from_file_location(category_name, file_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        if hasattr(module, 'templates'):
            group_templates = module.templates
            for key, value in group_templates.items():
                full_key = f"{category_name} → {key}"
                TEMPLATES[full_key] = value
            print(f"Loaded {len(group_templates)} from {filename}")

print(f"Templates Loaded: {len(TEMPLATES)}")
# --------------------------------------------------------------------
import torch
from transformers import pipeline
print("Downloading and loading model to GPU... (This takes ~2 mins)")
local_pipe = pipeline(
    "text-generation",
    model="mistralai/Mistral-7B-Instruct-v0.1",
    torch_dtype=torch.float16,
    device_map="auto",
    model_kwargs={"load_in_4bit": True}
)
# --------------------------------------------------------------------
def call_llm(prompt):
    formatted_prompt = f"<|system|>\nYou are a helpful assistant that outputs only JSON.<|end|>\n<|user|>\n{prompt}<|end|>\n<|assistant|>"
    try:
        outputs = local_pipe(
            formatted_prompt,
            max_new_tokens=400,
            do_sample=True,
            temperature=0.7,
            return_full_text=False
        )
        return outputs[0]["generated_text"]
    except Exception as e:
        print(f"Local Gen Error: {e}")
        return None
print("Testing local generation...")
print(call_llm("Generate JSON: {\"status\": \"ready\"}"))
# --------------------------------------------------------------------
import time
import pandas as pd
import concurrent.futures
import re
import json
import random
import matplotlib.pyplot as plt
import seaborn as sns
from gradio_client import Client
import requests
from templates.education import templates as education_templates
from templates.medical import templates as medical_templates
from templates.research import templates as research_templates
from templates.human_resources import templates as hr_templates
from templates.industrial import templates as industrial_templates

template_categories = {
    "Education": education_templates,
    "Medical": medical_templates,
    "Research": research_templates,
    "Human Resources": hr_templates,
    "Industrial": industrial_templates,
}
TEMPLATES = {
    f"{category} → {key}": value
    for category, group in template_categories.items()
    for key, value in group.items()
}
print(f"Loaded {len(TEMPLATES)} templates from your modules")
TARGET_SPACE = "Koromama/UOWM"
CONCURRENCY_LEVEL = 5
TOTAL_REQUESTS = 50
MAX_DYNAMIC_FIELDS = 30
try:
    from google.colab import userdata
    HF_TOKEN = userdata.get('HF_TOKEN')
    if not HF_TOKEN:
        raise ValueError("Token is empty")
    print("Retrieved HF_TOKEN from Colab Secrets")
except Exception as e:
    print(f"ERROR: {e}")
    raise

LLM_MODEL = "mistralai/Mistral-7B-Instruct-v0.1"

def call_llm(prompt):
    """
    Generates text using the locally loaded GPU model.
    Zero network latency, no timeouts, no API limits.
    """
    formatted_prompt = f"<|system|>\nYou are a helpful data generator that outputs ONLY valid JSON.<|end|>\n<|user|>\n{prompt}<|end|>\n<|assistant|>"
    try:
        outputs = local_pipe(
            formatted_prompt,
            max_new_tokens=400,
            do_sample=True,     
            temperature=0.7,
            return_full_text=False
        )
        return outputs[0]["generated_text"]
    except Exception as e:
        print(f"Local Gen Error: {e}")
        return None
test_response = call_llm("Say 'hello' in JSON format: {\"greeting\": \"hello\"}")
if test_response:
    print(f"LLM working! Response: {test_response[:100]}")
else:
    print("LLM test failed - will use simple fallback")
def extract_required_fields(template_text):
    """Extract [MISSING FIELD: X] and [NEEDED FIELD: X]"""
    missing = re.findall(r'\[MISSING FIELD:\s*(.*?)\]', template_text)
    needed = re.findall(r'\[NEEDED FIELD:\s*(.*?)\]', template_text)
    all_fields = []
    seen = set()
    for f in (needed + missing):
        if f not in seen:
            seen.add(f)
            all_fields.append(f)
    return all_fields

def generate_field_values(template_name, required_fields):
    """
    Use LLM to generate realistic values for required fields
    """
    if not required_fields:
        return []

    category = template_name.split(" → ")[0] if " → " in template_name else "General"
    prompt = f"""Generate realistic data for a {category} consent form.

Fields needed:
{chr(10).join([f'- {field}' for field in required_fields])}

Respond with ONLY a JSON object with these exact keys and realistic values. Be creative and suggestions as if you were speaking verbally in detail.
If the object is not a name or a number (e.g. date) use at least 3 sentences to describe it.
JSON:"""

    print(f"Calling LLM to generate {len(required_fields)} fields...")

    response = call_llm(prompt)

    if response:
        try:
            json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response)
            if json_match:
                data = json.loads(json_match.group())
                values = []
                for field in required_fields:
                    if field in data:
                        values.append(str(data[field]))
                    else:
                        matched = False
                        for key in data:
                            if field.lower() in key.lower() or key.lower() in field.lower():
                                values.append(str(data[key]))
                                matched = True
                                break
                        if not matched:
                            values.append(f"Generated_{field.replace(' ', '_')}")
                print(f"LLM generated values successfully")
                return values
        except Exception as e:
            print(f"JSON parsing failed: {e}")
    print(f"Using fallback generation")
    fallback_values = []
    for field in required_fields:
        field_lower = field.lower()
        if "name" in field_lower:
            fallback_values.append(f"{random.choice(['John', 'Jane', 'Alex', 'Maria'])} {random.choice(['Smith', 'Johnson', 'Williams', 'Brown'])}")
        elif "email" in field_lower:
            fallback_values.append(f"user{random.randint(1,999)}@test.com")
        elif "date" in field_lower:
            fallback_values.append("2024-06-15")
        elif "institution" in field_lower or "facility" in field_lower:
            fallback_values.append(f"{random.choice(['Springfield', 'Central', 'Regional'])} {random.choice(['University', 'Hospital', 'Institute'])}")
        elif "purpose" in field_lower or "description" in field_lower:
            fallback_values.append(f"To conduct necessary {category.lower()} activities")
        elif "duration" in field_lower:
            fallback_values.append(f"{random.randint(1,12)} months")
        else:
            fallback_values.append(f"Test_{field.replace(' ', '_')}")

    return fallback_values
# --------------------------------------------------------------------
def run_session(req_id):
    """Execute one complete form generation test"""
    print(f"\n{'='*70}")
    print(f"Request {req_id+1}/{TOTAL_REQUESTS}")
    print(f"{'='*70}")
    template_key = random.choice(list(TEMPLATES.keys()))
    template_text = TEMPLATES[template_key]
    print(f"Template: {template_key}")
    required_fields = extract_required_fields(template_text)
    print(f"Fields needed: {required_fields}")
    t_gen_start = time.time()
    generated_values = generate_field_values(template_key, required_fields)
    t_gen_end = time.time()
    gen_time = round(t_gen_end - t_gen_start, 2)
    print(f" Generated in {gen_time}s:")
    for field, value in zip(required_fields, generated_values):
        print(f"      • {field}: {value}")
    api_field_values = generated_values + [""] * (MAX_DYNAMIC_FIELDS - len(generated_values))
    api_field_values = api_field_values[:MAX_DYNAMIC_FIELDS]

    payload = [
        template_key,
        "Test Institution",
        "requester@test.edu",
        "participant@test.com",
        *api_field_values
    ]
    print(f"Calling FormFlow at {TARGET_SPACE}...")
    try:
        client = Client(TARGET_SPACE, hf_token=HF_TOKEN)

        t_api_start = time.time()
        result = client.predict(*payload, api_name="/generate_consent_form")
        t_api_end = time.time()

        api_latency = round(t_api_end - t_api_start, 2)

        # Extract output
        if isinstance(result, (list, tuple)):
            output_text = str(result[0]) if result else ""
        else:
            output_text = str(result)

        output_tokens = len(output_text.split())
        tps = round(output_tokens / api_latency, 2) if api_latency > 0 else 0

        print(f"  SUCCESS!")
        print(f"  API Latency: {api_latency}s")
        print(f"  Output: {output_tokens} tokens")
        print(f"  Speed: {tps} tokens/sec")
        print(f"  Preview: {output_text[:150]}...")

        return {
            "id": req_id,
            "status": "SUCCESS",
            "category": template_key.split(" → ")[0],
            "template": template_key,
            "num_fields": len(required_fields),
            "gen_time": gen_time,
            "api_latency": api_latency,
            "total_time": round(gen_time + api_latency, 2),
            "output_tokens": output_tokens,
            "tps": tps,
            "generated_input_data": str(dict(zip(required_fields, generated_values))),
            "final_form_output": output_text
        }

    except Exception as e:
        print(f"FAILED: {e}")
        return {
            "id": req_id,
            "status": "ERROR",
            "template": template_key,
            "error": str(e)
        }
test_start = time.time()
results = []
with concurrent.futures.ThreadPoolExecutor(max_workers=CONCURRENCY_LEVEL) as executor:
    futures = [executor.submit(run_session, i) for i in range(TOTAL_REQUESTS)]

    for future in concurrent.futures.as_completed(futures):
        result = future.result()
        results.append(result)

test_duration = time.time() - test_start
print("\n RESULTS")
df = pd.DataFrame(results)
df.to_csv("formflow_test_results.csv", index=False)
print(f" Saved: formflow_test_results.csv")

success_df = df[df['status'] == 'SUCCESS']
failed = len(df) - len(success_df)
print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print(f" Successful: {len(success_df)}")
print(f" Failed: {failed}")
print(f" Duration: {round(test_duration, 2)}s")

if not success_df.empty:
    print(f"\n Performance:")
    print(f"   Generation Time: {success_df['gen_time'].mean():.2f}s avg")
    print(f"   API Latency: {success_df['api_latency'].mean():.2f}s avg")
    print(f"   Total Time: {success_df['total_time'].mean():.2f}s avg")
    print(f"   Throughput: {success_df['tps'].mean():.1f} tokens/sec")
    print(f"   Output Size: {success_df['output_tokens'].mean():.0f} tokens")

    print(f"\n By Category:")
    print(success_df.groupby('category')[['api_latency', 'tps']].mean().round(2))

    try:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        sns.boxplot(data=success_df, x='category', y='api_latency', ax=axes[0,0])
        axes[0,0].set_title('API Latency by Category')
        axes[0,0].tick_params(axis='x', rotation=15)

        sns.histplot(success_df['tps'], bins=10, kde=True, ax=axes[0,1])
        axes[0,1].set_title('Throughput Distribution')

        sample = success_df.head(10)
        axes[1,0].bar(range(len(sample)), sample['gen_time'], label='Generation')
        axes[1,0].bar(range(len(sample)), sample['api_latency'],
                     bottom=sample['gen_time'], label='API')
        axes[1,0].set_title('Time Breakdown')
        axes[1,0].legend()

        sns.scatterplot(data=success_df, x='num_fields', y='api_latency',
                       hue='category', ax=axes[1,1])
        axes[1,1].set_title('Fields vs Latency')

        plt.tight_layout()
        plt.savefig('formflow_test_report.png', dpi=150)
        print("\n Saved: formflow_test_report.png")
        plt.show()
    except:
        pass
print("\n TEST COMPLETE")
# --------------------------------------------------------------------
