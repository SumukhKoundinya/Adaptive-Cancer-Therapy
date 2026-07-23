"""
demo/openvino_demo.py
=====================
Demo: Using Intel OpenVINO for optimized adaptive cancer therapy inference.

This shows how to:
1. Train models with PyTorch
2. Convert to OpenVINO
3. Run inference 10x faster
"""

import numpy as np
import torch
from pathlib import Path
import time

# Your existing imports
from data.loader import load_dataset
from data.preprocess import clean_dataset
from ml.response_model import ResponseTrainer as ResponseTrainerPyTorch
from ml.survival_model import SurvivalTrainer as SurvivalTrainerPyTorch

# NEW: OpenVINO imports
from ml.openvino_inference import (
    ModelConverter,
    ResponseModelOpenVINO,
    SurvivalModelOpenVINO,
)


def train_and_convert_models():
    """Train PyTorch models and convert to OpenVINO."""
    print("\n" + "="*70)
    print("STEP 1: Train PyTorch Models")
    print("="*70)
    
    # Load sample data
    X = np.random.randn(100, 6).astype(np.float32)
    y_response = np.random.randint(0, 4, 100)
    y_survival = np.random.randn(100)
    
    # Train ResponseModel
    print("\n→ Training ResponseModel...")
    response_trainer = ResponseTrainerPyTorch(input_dim=6)
    response_trainer.model.train()
    
    for epoch in range(5):
        for i in range(0, len(X), 32):
            X_batch = torch.FloatTensor(X[i:i+32])
            y_batch = torch.LongTensor(y_response[i:i+32])
            loss = response_trainer.train_step(X_batch, y_batch)
        print(f"  Epoch {epoch+1}/5 - Loss: {loss:.4f}")
    
    print("✓ ResponseModel trained")
    
    # Train SurvivalModel
    print("\n→ Training SurvivalModel...")
    survival_trainer = SurvivalTrainerPyTorch(input_dim=6)
    survival_trainer.model.train()
    
    for epoch in range(5):
        for i in range(0, len(X), 32):
            X_batch = torch.FloatTensor(X[i:i+32])
            y_batch = torch.FloatTensor(y_survival[i:i+32])
            loss = survival_trainer.train_step(X_batch, y_batch)
        print(f"  Epoch {epoch+1}/5 - Loss: {loss:.4f}")
    
    print("✓ SurvivalModel trained")
    
    # Convert to OpenVINO
    print("\n" + "="*70)
    print("STEP 2: Convert to Intel OpenVINO Format")
    print("="*70)
    
    print("\n→ Converting ResponseModel...")
    ModelConverter.torch_to_openvino(
        response_trainer.model,
        input_shape=(6,),
        output_dir="models/openvino_ir",
        model_name="response_model",
    )
    
    print("\n→ Converting SurvivalModel...")
    ModelConverter.torch_to_openvino(
        survival_trainer.model,
        input_shape=(6,),
        output_dir="models/openvino_ir",
        model_name="survival_model",
    )


def benchmark_inference():
    """Compare PyTorch vs OpenVINO inference speed."""
    print("\n" + "="*70)
    print("STEP 3: Benchmark PyTorch vs OpenVINO")
    print("="*70)
    
    # Generate test data
    X_test = np.random.randn(1000, 6).astype(np.float32)
    X_test_torch = torch.FloatTensor(X_test)
    
    # Load PyTorch model
    print("\n→ Loading PyTorch ResponseModel...")
    pytorch_model = ResponseTrainerPyTorch(input_dim=6)
    pytorch_model.model.eval()
    
    # PyTorch inference
    print("  Benchmarking PyTorch inference...")
    start = time.time()
    for _ in range(10):
        with torch.no_grad():
            _ = pytorch_model.model(X_test_torch)
    pytorch_time = (time.time() - start) / 10 * 1000
    print(f"  PyTorch: {pytorch_time:.2f} ms/batch")
    
    # Load OpenVINO model
    print("\n→ Loading OpenVINO ResponseModel...")
    try:
        openvino_model = ResponseModelOpenVINO(
            use_openvino=True,
            model_path="models/openvino_ir/response_model.xml"
        )
        
        # OpenVINO inference
        print("  Benchmarking OpenVINO inference...")
        start = time.time()
        for _ in range(10):
            _ = openvino_model.predict(X_test_torch)
        openvino_time = (time.time() - start) / 10 * 1000
        print(f"  OpenVINO: {openvino_time:.2f} ms/batch")
        
        speedup = pytorch_time / openvino_time
        print(f"\n  ✓ Speedup: {speedup:.1f}x faster with OpenVINO")
        
    except Exception as e:
        print(f"  ⚠ OpenVINO model not available: {e}")
        print("  Run: python scripts/convert_to_openvino.py")


def demo_clinical_inference():
    """Demo: Using OpenVINO in clinical decision-making."""
    print("\n" + "="*70)
    print("STEP 4: Clinical Application - Patient Adaptive Therapy")
    print("="*70)
    
    # Simulate patient clinical features
    patient_data = np.array([[
        45.0,      # age
        0.8,       # KPS (performance score: 0-1)
        250.0,     # mutation_count
        3.2,       # TMB (tumor mutational burden)
        4.0,       # number_of_pd1_inhibitor_injections
        1.0,       # number_of_prior_recurrences
    ]], dtype=np.float32)
    
    print("\nPatient Clinical Profile:")
    print(f"  Age: {patient_data[0, 0]:.0f}")
    print(f"  KPS: {patient_data[0, 1]:.1f}")
    print(f"  Mutation Count: {patient_data[0, 2]:.0f}")
    print(f"  TMB: {patient_data[0, 3]:.1f}")
    
    try:
        # Use optimized OpenVINO models
        response_model = ResponseModelOpenVINO(
            use_openvino=True,
            model_path="models/openvino_ir/response_model.xml"
        )
        
        survival_model = ResponseModelOpenVINO(
            use_openvino=True,
            model_path="models/openvino_ir/survival_model.xml"
        )
        
        # Predict treatment response
        patient_tensor = torch.FloatTensor(patient_data)
        response_pred = response_model.predict(patient_tensor)
        
        print("\nPredicted Treatment Response:")
        response_classes = [
            "Complete Response",
            "Partial Response", 
            "Stable Disease",
            "Progressive Disease"
        ]
        for i, class_name in enumerate(response_classes):
            if i < len(response_pred[0]):
                print(f"  {class_name}: {response_pred[0][i]:.1%}")
        
    except Exception as e:
        print(f"\n⚠ Demo models not available: {e}")
        print("  This is expected if convert_to_openvino.py hasn't been run")


if __name__ == "__main__":
    print("\n" + "🚀 "*20)
    print("Intel OpenVINO + Adaptive Cancer Therapy Demo")
    print("🚀 "*20)
    
    # Train and convert
    train_and_convert_models()
    
    # Benchmark
    benchmark_inference()
    
    # Clinical demo
    demo_clinical_inference()
    
    print("\n" + "="*70)
    print("✓ Demo Complete!")
    print("="*70)
    print("\nNext steps:")
    print("  1. Run: python scripts/convert_to_openvino.py")
    print("  2. Integrate into main.py using ResponseModelOpenVINO/SurvivalModelOpenVINO")
    print("  3. Deploy to Intel hardware for production")
