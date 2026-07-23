"""
ml/openvino_inference.py
========================
Intel OpenVINO inference wrapper for optimized model deployment.

Provides:
- Model conversion from PyTorch → ONNX → OpenVINO IR
- Fast CPU/GPU inference with OpenVINO Runtime
- Quantization support for edge deployment
- Compatible with ResponseModel and SurvivalModel

Usage:
    converter = ModelConverter()
    converter.torch_to_openvino("response_model.pth", "models/ir/response")
    
    inference = OpenVINOInference("models/ir/response.xml")
    predictions = inference.predict(X_test)
"""

from __future__ import annotations
import torch
import numpy as np
from pathlib import Path
from typing import Optional


class ModelConverter:
    """Convert PyTorch models to OpenVINO IR format."""
    
    @staticmethod
    def torch_to_onnx(
        model: torch.nn.Module,
        input_shape: tuple,
        output_path: str,
        model_name: str = "model",
    ) -> str:
        """Export PyTorch model to ONNX format."""
        import torch.onnx
        
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Create dummy input matching your data shape
        dummy_input = torch.randn(1, *input_shape)
        
        onnx_path = str(Path(output_path) / f"{model_name}.onnx")
        
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            input_names=["input"],
            output_names=["output"],
            opset_version=14,
            do_constant_folding=True,
        )
        
        print(f"✓ Exported to ONNX: {onnx_path}")
        return onnx_path
    
    @staticmethod
    def onnx_to_openvino(
        onnx_path: str,
        output_dir: str,
        model_name: str = "model",
    ) -> str:
        """Convert ONNX to OpenVINO IR format."""
        from openvino.tools import mo
        
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        ir_model = mo.convert_model(
            onnx_path,
            input_shape=[1, -1],  # batch_size=1, features=variable
            output_dir=output_dir,
            model_name=model_name,
        )
        
        xml_path = str(Path(output_dir) / f"{model_name}.xml")
        print(f"✓ Converted to OpenVINO IR: {xml_path}")
        return xml_path
    
    @staticmethod
    def torch_to_openvino(
        torch_model: torch.nn.Module,
        input_shape: tuple,
        output_dir: str,
        model_name: str = "model",
    ) -> str:
        """Full pipeline: PyTorch → ONNX → OpenVINO."""
        onnx_path = ModelConverter.torch_to_onnx(
            torch_model, input_shape, output_dir, model_name
        )
        ir_path = ModelConverter.onnx_to_openvino(
            onnx_path, output_dir, model_name
        )
        return ir_path


class OpenVINOInference:
    """Fast inference with OpenVINO Runtime."""
    
    def __init__(self, model_xml_path: str, device: str = "CPU"):
        """
        Initialize OpenVINO inference engine.
        
        Args:
            model_xml_path: Path to .xml file (e.g., "models/ir/response.xml")
            device: "CPU", "GPU", "MYRIAD" (Intel Neural Stick), "HDDL"
        """
        from openvino.runtime import Core
        
        self.device = device
        self.core = Core()
        
        # Load compiled model
        self.model = self.core.read_model(model=model_xml_path)
        self.compiled_model = self.core.compile_model(self.model, device)
        
        # Get input/output layer names
        self.input_name = next(iter(self.compiled_model.inputs))
        self.output_name = next(iter(self.compiled_model.outputs))
        
        print(f"✓ OpenVINO model loaded on {device}")
        print(f"  Input shape: {self.input_name.shape}")
        print(f"  Output shape: {self.output_name.shape}")
    
    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Run inference.
        
        Args:
            x: Input array (batch_size, features)
            
        Returns:
            Predictions from model output
        """
        # Ensure float32 for OpenVINO
        x = np.asarray(x, dtype=np.float32)
        
        input_data = {self.input_name: x}
        results = self.compiled_model(input_data)
        
        return results[self.output_name]
    
    def predict_from_torch(self, x: torch.Tensor) -> np.ndarray:
        """Predict from PyTorch tensor."""
        x_np = x.cpu().detach().numpy().astype(np.float32)
        return self.predict(x_np)


class ResponseModelOpenVINO:
    """Wrapper: ResponseModel with OpenVINO optimization."""
    
    def __init__(self, use_openvino: bool = False, model_path: Optional[str] = None):
        """
        Args:
            use_openvino: If True, use OpenVINO optimized inference
            model_path: Path to .xml file if use_openvino=True
        """
        self.use_openvino = use_openvino
        
        if use_openvino:
            assert model_path, "model_path required for OpenVINO"
            self.inference = OpenVINOInference(model_path, device="CPU")
        else:
            from ml.response_model import ResponseModel
            self.model = ResponseModel(input_dim=6)
    
    def predict(self, x):
        """Predict (auto-selects PyTorch or OpenVINO)."""
        if self.use_openvino:
            return self.inference.predict_from_torch(x)
        else:
            with torch.no_grad():
                logits = self.model(x)
                return torch.argmax(logits, dim=1)


class SurvivalModelOpenVINO:
    """Wrapper: SurvivalModel with OpenVINO optimization."""
    
    def __init__(self, use_openvino: bool = False, model_path: Optional[str] = None):
        """
        Args:
            use_openvino: If True, use OpenVINO optimized inference
            model_path: Path to .xml file if use_openvino=True
        """
        self.use_openvino = use_openvino
        
        if use_openvino:
            assert model_path, "model_path required for OpenVINO"
            self.inference = OpenVINOInference(model_path, device="CPU")
        else:
            from ml.survival_model import SurvivalModel
            self.model = SurvivalModel(input_dim=6)
    
    def predict(self, x):
        """Predict (auto-selects PyTorch or OpenVINO)."""
        if self.use_openvino:
            return self.inference.predict_from_torch(x)
        else:
            with torch.no_grad():
                return self.model(x)
