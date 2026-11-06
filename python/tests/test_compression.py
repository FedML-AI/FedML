import importlib.util
import os
import sys

import pytest
import torch

# Load fedml.utils.compression directly
_compression_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "fedml", "utils", "compression.py")
)
_spec = importlib.util.spec_from_file_location("fedml.utils.compression", _compression_path)
_compression = importlib.util.module_from_spec(_spec)
sys.modules["fedml.utils.compression"] = _compression
_spec.loader.exec_module(_compression)

Int4BlockCompressor = _compression.Int4BlockCompressor
EFInt4BlockCompressor = _compression.EFInt4BlockCompressor
compressors = _compression.compressors


class TestInt4Compression:
    def test_invalid_parameters(self):
        with pytest.raises(ValueError):
            Int4BlockCompressor(block_size=0)
        with pytest.raises(ValueError):
            Int4BlockCompressor(block_size=-16)
        with pytest.raises(ValueError):
            Int4BlockCompressor(block_size=32.0)
        with pytest.raises(ValueError):
            Int4BlockCompressor(block_size=True)

    def test_compressors_registry(self):
        assert "int4" in compressors
        assert "efint4" in compressors
        c1 = compressors["int4"]()
        c2 = compressors["efint4"]()
        assert isinstance(c1, Int4BlockCompressor)
        assert isinstance(c2, EFInt4BlockCompressor)

    def test_int4_compress_decompress_shape_and_payload(self):
        torch.manual_seed(42)
        tensor = torch.randn(10, 20)  # 200 float32 elements = 800 bytes
        compressor = Int4BlockCompressor(block_size=32)

        packed, scales, shape = compressor.compress(tensor, name="layer1")

        assert packed.dtype == torch.uint8
        # 200 padded to 224 (7 blocks of 32), packed 2 per byte -> 112 bytes
        assert packed.numel() == 112
        assert packed.element_size() == 1
        assert (packed.numel() * packed.element_size()) < (tensor.numel() * tensor.element_size()) // 6

        decompressed = compressor.decompress(packed, shape, scales, name="layer1")
        assert decompressed.shape == tensor.shape
        assert decompressed.dtype == torch.float32

        mse = torch.mean((tensor - decompressed) ** 2).item()
        assert mse < 0.05

    def test_whole_tensor_scaling(self):
        torch.manual_seed(100)
        tensor = torch.randn(16, 16)
        compressor = Int4BlockCompressor(block_size=None)

        packed, scales, shape = compressor.compress(tensor, name="full")
        assert scales.numel() == 1

        decompressed = compressor.decompress(packed, shape, scales, name="full")
        assert decompressed.shape == tensor.shape
        mse = torch.mean((tensor - decompressed) ** 2).item()
        assert mse < 0.05

    def test_multidimensional_tensors(self):
        torch.manual_seed(77)
        compressor = Int4BlockCompressor(block_size=32)

        # 1D
        t1 = torch.randn(128)
        p1, s1, sh1 = compressor.compress(t1)
        d1 = compressor.decompress(p1, sh1, s1)
        assert d1.shape == t1.shape
        assert torch.mean((t1 - d1) ** 2).item() < 0.05

        # 3D
        t3 = torch.randn(4, 8, 16)
        p3, s3, sh3 = compressor.compress(t3)
        d3 = compressor.decompress(p3, sh3, s3)
        assert d3.shape == t3.shape
        assert torch.mean((t3 - d3) ** 2).item() < 0.05

        # 4D Conv weight
        t4 = torch.randn(16, 3, 5, 5)
        p4, s4, sh4 = compressor.compress(t4)
        d4 = compressor.decompress(p4, sh4, s4)
        assert d4.shape == t4.shape
        assert torch.mean((t4 - d4) ** 2).item() < 0.05

    def test_unaligned_and_odd_sizes(self):
        torch.manual_seed(99)
        tensor = torch.randn(37)
        compressor = Int4BlockCompressor(block_size=16)

        packed, scales, shape = compressor.compress(tensor)
        decompressed = compressor.decompress(packed, shape, scales)

        assert decompressed.shape == tensor.shape
        torch.testing.assert_close(decompressed, tensor, atol=0.5, rtol=0.2)

    def test_zero_tensor_handling(self):
        zeros = torch.zeros(16, 16)
        compressor = Int4BlockCompressor(block_size=16)

        packed, scales, shape = compressor.compress(zeros)
        decompressed = compressor.decompress(packed, shape, scales)

        torch.testing.assert_close(decompressed, zeros)

    def test_non_finite_rejection(self):
        compressor = Int4BlockCompressor()
        with_nan = torch.tensor([1.0, float("nan"), 2.0])
        with pytest.raises(ValueError, match="non-finite values"):
            compressor.compress(with_nan)

        with_inf = torch.tensor([1.0, float("inf"), 2.0])
        with pytest.raises(ValueError, match="non-finite values"):
            compressor.compress(with_inf)

    def test_empty_tensor_handling(self):
        compressor = Int4BlockCompressor()
        empty = torch.empty(0)

        packed, scales, shape = compressor.compress(empty)
        decompressed = compressor.decompress(packed, shape, scales)
        assert decompressed.numel() == 0

    def test_error_feedback_compressor(self):
        torch.manual_seed(123)
        ef_compressor = EFInt4BlockCompressor(block_size=32)

        # Simulate 5 training rounds on a weight gradient tensor
        name = "fc.weight"
        residuals_history = []
        for _ in range(5):
            grad = torch.randn(8, 16) * 0.1
            packed, scales, shape = ef_compressor.compress(grad, name=name)
            decompressed = ef_compressor.decompress(packed, shape, scales, name=name)

            assert decompressed.shape == grad.shape
            assert name in ef_compressor.residuals
            res_norm = torch.norm(ef_compressor.residuals[name]).item()
            residuals_history.append(res_norm)

        # Residuals remain stably bounded across rounds
        assert all(r < 1.0 for r in residuals_history)

        ef_compressor.clear()
        assert len(ef_compressor.residuals) == 0

    def test_alternative_calling_conventions(self):
        torch.manual_seed(5)
        tensor = torch.randn(8, 8)
        compressor = Int4BlockCompressor(block_size=16)

        # Test decompress using cached metadata
        packed, scales, _ = compressor.compress(tensor, name="param")
        d1 = compressor.decompress(packed, name="param")
        assert d1.shape == tensor.shape

        # Test decompress_new
        d2 = compressor.decompress_new(packed, scales=scales, name="param")
        assert d2.shape == tensor.shape

        # Test passing shape as 2nd positional argument
        d3 = compressor.decompress(packed, tensor.shape, scales, name="param")
        assert d3.shape == tensor.shape
