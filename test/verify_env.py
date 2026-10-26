"""
Verification script for Data Science & AI environment.
Validates GPU (CUDA), libraries, and dataset availability.
"""

import sys


def main():
    print(f"Python executable: {sys.executable}")
    print(f"Python version: {sys.version}\n")

    # Check CUDA / GPU
    try:
        import torch

        print(f"PyTorch version: {torch.__version__}")
        cuda_avail = torch.cuda.is_available()
        print(f"CUDA Available: {cuda_avail}")
        if cuda_avail:
            print(f"Device Name: {torch.cuda.get_device_name(0)}")
            print(
                f"VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB"
            )
    except ImportError as e:
        print(f"PyTorch not loaded: {e}")

    # Check core data libraries
    libs = [
        "numpy",
        "pandas",
        "polars",
        "scipy",
        "matplotlib",
        "seaborn",
        "plotly",
        "sklearn",
        "xgboost",
        "lightgbm",
        "catboost",
        "imblearn",
        "shap",
    ]
    print("\n--- Package Verification ---")
    for lib in libs:
        try:
            mod = __import__(lib)
            ver = getattr(mod, "__version__", "installed")
            print(f"  [OK] {lib}: {ver}")
        except ImportError as e:
            print(f"  [MISSING] {lib}: {e}")

    # Check dataset
    import os

    dataset = "data/DATA IBM/card_transaction.v1.csv"
    if os.path.exists(dataset):
        size_mb = os.path.getsize(dataset) / (1024 * 1024)
        print(f"\n[OK] Dataset '{dataset}' found ({size_mb:.2f} MB).")
        try:
            import polars as pl

            df = pl.read_csv(dataset, n_rows=5)
            print(f"Polars read test success: {df.shape[1]} columns detected.")
        except Exception as e:
            print(f"Polars read test note: {e}")
    else:
        print(f"\n[WARNING] Dataset '{dataset}' not found.")


if __name__ == "__main__":
    main()
