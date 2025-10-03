"""
Script to install required visualization dependencies
"""
import subprocess
import sys

def main():
    """Install matplotlib, seaborn, and other viz libraries."""
    packages = [
        "matplotlib>=3.8.0",
        "seaborn>=0.13.0",
        "pandas>=2.1.0",
    ]
    
    print("Installing visualization dependencies...")
    print("=" * 60)
    
    for package in packages:
        print(f"Installing {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    
    print("=" * 60)
    print("✓ All visualization dependencies installed!")
    print("\nYou can now run: python scripts/generate_visualizations.py")

if __name__ == "__main__":
    main()
