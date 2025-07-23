"""
Lunar Lake NPU INT8 Support Testing
This script tests various approaches to INT8 support on Lunar Lake NPU
including FakeQuantize, native INT8, and weight-only quantization.
It aims to identify the best method for INT8 operations on NPU.
The goal is to ensure compatibility and performance for INT8 workloads.
Created by: [Morteza Heidari]
Date: [07/23/2025]
"""
import numpy as np
import openvino as ov

def create_matmul_int8_fakequantize(shape, w_fp16, x_reference):
    """
     SOLUTION 1: FakeQuantize approach for INT8 simulation
    This is the RECOMMENDED approach for Lunar Lake NPU INT8 support
    """
    dtype = ov.Type.f16  # FakeQuantize requires FP16/FP32 input!
    
    act = ov.op.Parameter(dtype, ov.Shape(shape))
    act.set_friendly_name("act")

    w_const = ov.op.Constant(dtype, ov.Shape(w_fp16.shape), w_fp16.flatten())
    w_const.set_friendly_name("weights")

    def create_fake_quantize_int8(x, reference_data):
        """Create proper INT8 FakeQuantize node"""
        # Calculate quantization range for INT8 
        min_val = float(np.min(reference_data))
        max_val = float(np.max(reference_data))
        
        # Ensure valid range
        if min_val == max_val:
            min_val = -10.0  # Use realistic range for testing
            max_val = 10.0
        
        # KEY FIX: Proper INT8 quantization parameters
        return ov.opset15.fake_quantize(
            x,
            ov.op.Constant(ov.Type.f16, ov.Shape([]), [min_val]),    # input_low
            ov.op.Constant(ov.Type.f16, ov.Shape([]), [max_val]),    # input_high  
            ov.op.Constant(ov.Type.f16, ov.Shape([]), [-128.0]),     # output_low (INT8 min)
            ov.op.Constant(ov.Type.f16, ov.Shape([]), [127.0]),      # output_high (INT8 max)
            levels=256  # 2^8 = 256 levels for INT8
        )

    # Apply fake quantization to simulate INT8
    act_fq = create_fake_quantize_int8(act, x_reference)
    w_fq = create_fake_quantize_int8(w_const, w_fp16)
    
    # MatMul operation
    mat = ov.opset15.matmul(act_fq, w_fq, False, True)
    mat.set_friendly_name("matmul_int8_fq")
    return ov.Model(mat, [act])

def create_matmul_native_int8(shape, w_int8):
    """
     SOLUTION 2: Native INT8 with type conversion
    Alternative approach if FakeQuantize doesn't work
    """
    try:
        # Use INT8 directly
        act = ov.op.Parameter(ov.Type.i8, ov.Shape(shape))
        act.set_friendly_name("act_int8")
        
        w_const = ov.op.Constant(ov.Type.i8, ov.Shape(w_int8.shape), w_int8.flatten())
        w_const.set_friendly_name("weights_int8")
        
        # KEY: Convert to FP32 for computation (NPU requirement)
        act_fp = ov.opset15.convert(act, ov.Type.f32)
        w_fp = ov.opset15.convert(w_const, ov.Type.f32)
        
        mat = ov.opset15.matmul(act_fp, w_fp, False, True)
        mat.set_friendly_name("matmul_native_int8")
        return ov.Model(mat, [act])
    except Exception as e:
        print(f" Native INT8 approach failed: {e}")
        return None

def create_matmul_quantized_weights_only(shape, w_int8):
    """
     SOLUTION 3: Quantize weights only, keep activations FP16
    This often works better on NPU
    """
    try:
        # FP16 activations
        act = ov.op.Parameter(ov.Type.f16, ov.Shape(shape))
        act.set_friendly_name("act_fp16")
        
        # INT8 weights converted to FP16 with quantization simulation
        w_fp16 = w_int8.astype(np.float16)
        w_const = ov.op.Constant(ov.Type.f16, ov.Shape(w_fp16.shape), w_fp16.flatten())
        
        # Apply FakeQuantize only to weights
        min_val = float(np.min(w_int8))
        max_val = float(np.max(w_int8))
        
        w_fq = ov.opset15.fake_quantize(
            w_const,
            ov.op.Constant(ov.Type.f16, ov.Shape([]), [min_val]),
            ov.op.Constant(ov.Type.f16, ov.Shape([]), [max_val]),
            ov.op.Constant(ov.Type.f16, ov.Shape([]), [-128.0]),
            ov.op.Constant(ov.Type.f16, ov.Shape([]), [127.0]),
            levels=256
        )
        
        mat = ov.opset15.matmul(act, w_fq, False, True)
        mat.set_friendly_name("matmul_weights_quantized")
        return ov.Model(mat, [act])
    except Exception as e:
        print(f" Weight-only quantization failed: {e}")
        return None

# --------------------------------------------------------------------
# 🔥 COMPREHENSIVE TEST FOR LUNAR LAKE NPU INT8 ISSUES
# --------------------------------------------------------------------
print("Testing INT8 support on Lunar Lake NPU...")
print("=" * 60)

b, d, t = 100, 1024, 1024

# Generate test data
x_i8 = np.random.randint(-10, 10, size=(b, d), dtype=np.int8)
w_i8 = np.random.randint(-10, 10, size=(t, d), dtype=np.int8)

print(f"Test Data:")
print(f"  Input shape: {x_i8.shape}")
print(f"  Weight shape: {w_i8.shape}")
print(f"  Input range: [{np.min(x_i8)}, {np.max(x_i8)}]")
print(f"  Weight range: [{np.min(w_i8)}, {np.max(w_i8)}]")

try:
    core = ov.Core()
    available_devices = core.available_devices
    print(f"\n Available devices: {available_devices}")
    
    # Choose device
    device = "NPU" if "NPU" in available_devices else "CPU"
    print(f" Testing on: {device}")
    
    print("\n" + "="*60)
    
    # Test all three approaches
    approaches = [
        ("FakeQuantize (Recommended)", lambda: create_matmul_int8_fakequantize(x_i8.shape, w_i8.astype(np.float16), x_i8.astype(np.float16))),
        ("Native INT8", lambda: create_matmul_native_int8(x_i8.shape, w_i8)),
        ("Weight-only Quantization", lambda: create_matmul_quantized_weights_only(x_i8.shape, w_i8))
    ]
    
    successful_approaches = []
    
    for approach_name, model_creator in approaches:
        print(f"\n Testing: {approach_name}")
        print("-" * 40)
        
        try:
            # Create model
            model = model_creator()
            if model is None:
                print(f" Model creation failed")
                continue
                
            print(f"✅ Model created successfully")
            
            # Compile model
            compiled = core.compile_model(model, device)
            print(f" Model compiled on {device}")
            
            # Test inference
            if approach_name == "Native INT8":
                input_data = x_i8  # Use INT8 input
            else:
                input_data = x_i8.astype(np.float16)  # Use FP16 input
                
            result = compiled(input_data)
            output_shape = result[list(result.keys())[0]].shape
            print(f" Inference successful, output shape: {output_shape}")
            
            successful_approaches.append(approach_name)
            
        except Exception as e:
            print(f" {approach_name} failed: {e}")
            if "NPU" in device:
                print("Trying CPU fallback...")
                try:
                    compiled_cpu = core.compile_model(model, "CPU")
                    result_cpu = compiled_cpu(input_data)
                    print(f" CPU fallback successful")
                except Exception as cpu_e:
                    print(f" CPU fallback also failed: {cpu_e}")

    # Summary
    print("\n" + "="*60)
    print("SUMMARY FOR LUNAR LAKE NPU INT8 ISSUE:")
    print("="*60)
    
    if successful_approaches:
        print(f" Successful approaches: {', '.join(successful_approaches)}")
        print(f"\n RECOMMENDATION:")
        if "FakeQuantize (Recommended)" in successful_approaches:
            print("   Use FakeQuantize approach - it properly simulates INT8 with FP16 parameters")
        elif "Weight-only Quantization" in successful_approaches:
            print("   Use weight-only quantization - keeps activations FP16, quantizes weights")
        else:
            print("   Use native INT8 with type conversion")
    else:
        print(" No approaches succeeded on NPU")
        print("💡 TROUBLESHOOTING:")
        print("   1. Check NPU driver version")
        print("   2. Verify OpenVINO version compatibility")
        print("   3. Try updating to latest NPU drivers")
        print("   4. Consider using FP16 instead of INT8 for now")
    
    print(f"\n TECHNICAL NOTES:")
    print(f"   - FakeQuantize requires FP16/FP32 input tensors, not INT8")
    print(f"   - NPU may require specific quantization patterns")
    print(f"   - Weight-only quantization often has better NPU support")
    print(f"   - Some INT8 operations may need driver updates")

except Exception as e:
    print(f" Critical error: {e}")
    import traceback
    traceback.print_exc()
    
    print(f"\n If you see this error, try:")
    print(f"   1. pip install --upgrade openvino")
    print(f"   2. Check NPU driver installation")
    print(f"   3. Restart system after driver update")