import sys
import os
import importlib.metadata

print("--- CrediSense AI Environment Verification ---")
print(f"Python Executable Path: {sys.executable}")
print(f"Python Version: {sys.version}")
print("")

# Map package import name to pip metadata name
package_metadata_map = {
    "flask": "flask",
    "pandas": "pandas",
    "numpy": "numpy",
    "sklearn": "scikit-learn",
    "xgboost": "xgboost",
    "shap": "shap",
    "joblib": "joblib",
    "dotenv": "python-dotenv"
}

verifications = {}

for import_name, metadata_name in package_metadata_map.items():
    try:
        # Test import
        if import_name == "dotenv":
            import dotenv
        else:
            __import__(import_name)
        
        # Resolve version via importlib.metadata
        try:
            version = importlib.metadata.version(metadata_name)
        except importlib.metadata.PackageNotFoundError:
            version = "Version N/A"
        
        verifications[import_name] = ("Success", version)
    except ImportError as e:
        verifications[import_name] = ("Failed", str(e))

print("Package Import & Version Check:")
for pkg, (status, version) in verifications.items():
    print(f"  - {pkg:<12}: Status: {status:<8} Version: {version}")

print("")
# Additional specific checks for XGBoost and SHAP
print("Detailed Check:")
try:
    import xgboost as xgb
    print("  - XGBoost successfully imported as 'xgb'.")
except Exception as e:
    print(f"  - XGBoost import failed: {e}")

try:
    import shap
    print("  - SHAP successfully imported as 'shap'.")
except Exception as e:
    print(f"  - SHAP import failed: {e}")

# Check if we are running inside the virtual environment
venv_path = os.path.join("C:\\Users\\anilk\\Documents\\CrediSenseAI", ".venv")
is_in_venv = venv_path.lower() in sys.executable.lower()
print(f"\nVirtual Environment Match Check: {is_in_venv} (Looking for: '{venv_path}')")
