import subprocess
import sys

def main():
    experiments = [
        "experiments/E2_homogeneous_datasets.py",
        "experiments/E3_operating_point.py",
        "experiments/E5_whitening_stability.py",
        "experiments/E6_adaptive_attacks.py",
        "experiments/E7_external_baselines.py",
        "experiments/E10_statistical_tests.py"
    ]
    
    print("="*80)
    print("Running Honest Evaluation Final Pipeline")
    print("="*80)
    
    for script in experiments:
        print(f"\n>>> Running {script}...")
        try:
            subprocess.run([sys.executable, script], check=True)
            print(f">>> Successfully completed {script}")
        except subprocess.CalledProcessError as e:
            print(f">>> Error running {script}: {e}")
            sys.exit(1)
            
    print("\n" + "="*80)
    print("All honest evaluation experiments completed successfully.")
    print("="*80)

if __name__ == "__main__":
    main()
