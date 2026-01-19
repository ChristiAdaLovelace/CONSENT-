#---------------------------------------------------------------------------------------------
#wget https://github.com/grafana/k6/releases/download/v0.48.0/k6-v0.48.0-linux-amd64.tar.gz
#tar -xzf k6-v0.48.0-linux-amd64.tar.gz
#gradio_client
#mv k6-v0.48.0-linux-amd64/k6 /usr/local/bin/
#k6 version
#---------------------------------------------------------------------------------------------
#from google.colab import userdata
#import json
#HF_TOKEN = userdata.get('HF_TOKEN')
#SPACE_URL = "https://huggingface.co/spaces/Koromama/UOWM"  
#print(f" Token loaded: {HF_TOKEN[:10]}...")
#print(f" Target: {SPACE_URL}")
#---------------------------------------------------------------------------------------------
import os
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import random
from google.colab import userdata
BASE_URL = "https://koromama-uowm.hf.space"
HF_TOKEN = userdata.get('HF_TOKEN')
CONCURRENT_USERS = 200
TEST_DURATION_SECONDS = 180 
RAMP_UP_SECONDS = 30
scenarios = [
    {
        "name": "Skincare",
        "template": "Medical → skincare",
        "fields": [
            "Botox Treatment", "2026-02-15", "Dr. Smith", "MD",
            "Cosmetic injection procedure", "No allergies",
            "Temporary bruising", "Ice packs if needed",
            "Laser discussed", "Test Clinic", "info@clinic.com",
            "N/A", "Treatment records", "Name, history, photos",
            "Consent GDPR 6(1)(a)", "EU database", "10 years",
            "Not shared", "Right to access", "privacy@clinic.com"
        ]
    },
    {
        "name": "Vaccination",
        "template": "Medical → vaccination",
        "fields": [
            "Flu Vaccine", "2026-01-20", "Health Center",
            "Nurse Jane", "RN", "LOT123",
            "Soreness, fever", "Allergic reaction rare",
            "Epinephrine available", "Public health database",
            "10 years retention"
        ]
    },
]
stats = {
    "total_requests": 0,
    "successful": 0,
    "failed": 0,
    "response_times": [],
    "errors": []
}
stats_lock = threading.Lock()
def log_result(success, duration, error=None):
    """Thread-safe logging of test results"""
    with stats_lock:
        stats["total_requests"] += 1
        if success:
            stats["successful"] += 1
            stats["response_times"].append(duration)
        else:
            stats["failed"] += 1
            if error:
                stats["errors"].append(str(error))

def run_single_test(user_id, iteration):
    """Run a single test request"""
    scenario = random.choice(scenarios)
    start_time = time.time()
    try:
        client = Client(BASE_URL, hf_token=HF_TOKEN if HF_TOKEN else None)
        inputs = [
            scenario["template"],      
            "Test Medical Center",     
            "test@clinic.com",         
            "patient@test.com",        
            False,                    
        ]
        all_fields = scenario["fields"] + [""] * (30 - len(scenario["fields"]))
        inputs.extend(all_fields)
        result = client.predict(
            *inputs,
            api_name="/generate"  
        )
        duration = time.time() - start_time
        if result and len(str(result)) > 100:
            print(f"User {user_id} | Iter {iteration} | {scenario['name']} | {duration:.2f}s")
            log_result(True, duration)
            return True
        else:
            print(f"User {user_id} | Iter {iteration} | Empty response | {duration:.2f}s")
            log_result(False, duration, "Empty response")
            return False
    except Exception as e:
        duration = time.time() - start_time
        error_msg = str(e)[:100]
        print(f"User {user_id} | Iter {iteration} | Error: {error_msg} | {duration:.2f}s")
        log_result(False, duration, error_msg)
        return False
def simulate_user(user_id, test_end_time):
    """Simulate a single user making requests"""
    iteration = 0
    while time.time() < test_end_time:
        iteration += 1
        run_single_test(user_id, iteration)
        wait_time = random.uniform(2, 5)
        time.sleep(wait_time)
    print(f" User {user_id} completed {iteration} iterations")
#---------------------------------------------------------------------------------------------
def print_stats():
    """Print final statistics"""
    print("\nLOAD TEST RESULTS")
    print(f"\nTotal Requests:   {stats['total_requests']}")
    print(f"Successful:         {stats['successful']} ({stats['successful']/max(stats['total_requests'],1)*100:.1f}%)")
    print(f"Failed:             {stats['failed']} ({stats['failed']/max(stats['total_requests'],1)*100:.1f}%)")
    if stats['response_times']:
        times = sorted(stats['response_times'])
        print(f"\nResponse Times:")
        print(f"  Min:              {min(times):.2f}s")
        print(f"  Max:              {max(times):.2f}s")
        print(f"  Average:          {sum(times)/len(times):.2f}s")
        print(f"  Median:           {times[len(times)//2]:.2f}s")
        print(f"  95th percentile:  {times[int(len(times)*0.95)]:.2f}s")
    if stats['errors']:
        print(f"\nTop Errors:")
        error_counts = {}
        for err in stats['errors'][:20]:  # First 20 errors
            error_counts[err] = error_counts.get(err, 0) + 1
        for err, count in sorted(error_counts.items(), key=lambda x: -x[1])[:5]:
            print(f"  [{count}x] {err[:60]}")
#---------------------------------------------------------------------------------------------
def main():
    print("GRADIO LOAD TEST - Python Client Method")
    print(f"Target:             {BASE_URL}")
    print(f"Concurrent Users:   {CONCURRENT_USERS}")
    print(f"Test Duration:      {TEST_DURATION_SECONDS}s")
    print(f"Ramp-up:            {RAMP_UP_SECONDS}s")
    print(f"Authentication:     {'Yes (HF_TOKEN set)' if HF_TOKEN else 'No (public)'}")
    print("=" * 70 + "\n")
    if not HF_TOKEN:
        print(" WARNING: No HF_TOKEN set.")
        print(" Set with: export HF_TOKEN='your_token'\n")
    test_end_time = time.time() + TEST_DURATION_SECONDS
    with ThreadPoolExecutor(max_workers=CONCURRENT_USERS) as executor:
        futures = []
        for i in range(CONCURRENT_USERS):
            if i > 0:
                delay = RAMP_UP_SECONDS / CONCURRENT_USERS
                time.sleep(delay)
            print(f" Starting User {i+1}...")
            future = executor.submit(simulate_user, i+1, test_end_time)
            futures.append(future)
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"User thread error: {e}")
    print_stats()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n Test interrupted by user")
        print_stats()
      #---------------------------------------------------------------------------------------------
