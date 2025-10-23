from pathlib import Path

"""
Configuration module for DemandPredictor API.
Stores paths and common settings for API use.
"""

class Config:

    def __init__(self):
        self._root = Path(__file__).parent.parent
        self._models_dir = self._root / "saved_models" / "xgb_model"

    """
    Returns path to saved model folder.
    """
    def model(self) -> Path:
        if not self._models_dir.exists():
            raise FileNotFoundError(f"Model folder not found at {self._models_dir}")
        return self._models_dir