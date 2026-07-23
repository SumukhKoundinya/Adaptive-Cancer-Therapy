"""
scripts/convert_to_openvino.py
==============================
Convert trained PyTorch models to Intel OpenVINO format.

Run this after training your ResponseModel and SurvivalModel:
    python scripts/convert_to_openvino.py

This creates optimized IR models in models/openvino_ir/
"""

import sys
from pathlib import Path
import torch

# Add project to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from ml.response_model import ResponseModel
from ml.survival_model import SurvivalModel
from ml.openvino_inference import ModelConverter


def convert_response_model(model_path: str = "checkpoints/response_model.pth"):
    """Convert ResponseModel to OpenVINO."""
    print("\n" + "="*60)
    print("Converting ResponseModel to OpenVINO...")
    print("="*60)
    
    # Load trained model
    model = ResponseModel(input_dim=6)
    if Path(model_path).exists():
        model.load_state_dict(torch.load(model_path))
        print(f"✓ Loaded checkpoint: {model_path}")
    else:
        print(f"⚠ Checkpoint not found: {model_path}")
        print("  Using random weights for demo")
    
    model.eval()
    
    # Convert
    output_dir = "models/openvino_ir"
    ModelConverter.torch_to_openvino(
        model,
        input_shape=(6,),  # 6 clinical features
        output_dir=output_dir,
        model_name="response_model",
    )


def convert_survival_model(model_path: str = "checkpoints/survival_model.pth"):
    """Convert SurvivalModel to OpenVINO."""
    print("\n" + "="*60)
    print("Converting SurvivalModel to OpenVINO...")
    print("="*60)
    
    # Load trained model
    model = SurvivalModel(input_dim=6)
    if Path(model_path).exists():
        model.load_state_dict(torch.load(model_path))
        print(f"✓ Loaded checkpoint: {model_path}")
    else:
        print(f"⚠ Checkpoint not found: {model_path}")
        print("  Using random weights for demo")
    
    model.eval()
    
    # Convert
    output_dir = "models/openvino_ir"
    ModelConverter.torch_to_openvino(
        model,
        input_shape=(6,),  # 6 clinical features
        output_dir=output_dir,
        model_name="survival_model",
    )


if __name__ == "__main__":
    print("\n🔧 Intel OpenVINO Model Conversion")
    print("   Converting PyTorch models to OpenVINO IR format")
    
    convert_response_model()
    convert_survival_model()
    
    print("\n" + "="*60)
    print("✓ Conversion complete!")
    print("="*60)
    print("\nModels saved to: models/openvino_ir/")
    print("\nTo use in your code:")
    print("""
    from ml.openvino_inference import ResponseModelOpenVINO
    
    model = ResponseModelOpenVINO(
        use_openvino=True,
        model_path="models/openvino_ir/response_model.xml"
    )
    predictions = model.predict(X_test)
    """)
    print("\nBenefits:")
    print("  • 2-4x faster inference on CPU")
    print("  • 4-8x smaller model size")
    print("  • Deploy on Intel hardware without PyTorch")
