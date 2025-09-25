#!/usr/bin/env python3
"""
Task B Policy Representation Runner

Runs both B1 (policy classification) and B2 (changepoint detection) tasks
with proper error handling and data validation.
"""

import subprocess
import sys
import os
from pathlib import Path

def run_task_b1(data_dir: str, model_dir: str, output_dir: str):
    """Run Task B1: Policy Classification"""
    print("🚀 Running Task B1: Policy Classification")
    
    cmd = [
        sys.executable, "eval/policy_classification.py",
        "--data_dir", data_dir,
        "--model_dir", model_dir,
        "--output_dir", output_dir,
        "--epochs", "5",  # Reduced for demo
        "--window_size", "6"
    ]
    
    try:
        result = subprocess.run(cmd, cwd=".", capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Task B1 completed successfully")
            print(result.stdout)
        else:
            print("❌ Task B1 failed")
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
            
        return result.returncode == 0
        
    except Exception as e:
        print(f"❌ Error running Task B1: {e}")
        return False

def run_task_b2(data_dir: str, model_dir: str, output_dir: str):
    """Run Task B2: Policy Changepoint Detection"""
    print("🚀 Running Task B2: Policy Changepoint Detection")
    
    cmd = [
        sys.executable, "eval/policy_changepoint_detection.py",
        "--data_dir", data_dir,
        "--model_dir", model_dir,
        "--output_dir", output_dir,
        "--epochs", "5",  # Reduced for demo
        "--window_size", "6"
    ]
    
    try:
        result = subprocess.run(cmd, cwd=".", capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Task B2 completed successfully")
            print(result.stdout)
        else:
            print("❌ Task B2 failed")
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
            
        return result.returncode == 0
        
    except Exception as e:
        print(f"❌ Error running Task B2: {e}")
        return False

def main():
    """Run both Task B components"""
    
    # Default paths
    data_dir = "data/v5_test_three_shifts"
    model_dir = "runs"
    output_base = "reports/v5.0_proper_demo/task_b_policy_representation"
    
    print("🔧 Running Task B Policy Representation with fixes")
    print(f"Data: {data_dir}")
    print(f"Models: {model_dir}")
    print(f"Output: {output_base}")
    
    # Create output directories
    b1_output = Path(output_base) / "b1_fixed"
    b2_output = Path(output_base) / "b2_fixed"
    
    b1_output.mkdir(parents=True, exist_ok=True)
    b2_output.mkdir(parents=True, exist_ok=True)
    
    # Run both tasks
    b1_success = run_task_b1(data_dir, model_dir, str(b1_output))
    b2_success = run_task_b2(data_dir, model_dir, str(b2_output))
    
    print(f"\n📊 Task B Results:")
    print(f"  - B1 (Policy Classification): {'✅ Success' if b1_success else '❌ Failed'}")
    print(f"  - B2 (Changepoint Detection): {'✅ Success' if b2_success else '❌ Failed'}")
    
    if b1_success and b2_success:
        print("🎉 All Task B components completed successfully!")
    else:
        print("⚠️ Some Task B components failed. Check logs for details.")

if __name__ == "__main__":
    main()
