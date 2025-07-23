# AI-PC Enablement Repository

This repository contains comprehensive testing scripts, solutions, and enablement resources for AI acceleration across different compute platforms (CPU, GPU, NPU). It focuses on identifying and solving common AI-PC integration challenges with practical, reusable solutions.

## Purpose

This repository serves as a technical resource hub for:
- **Multi-Platform AI Testing**: Comprehensive test algorithms for CPU, GPU, and NPU acceleration
- **Problem-Solution Mapping**: Systematic identification and resolution of AI acceleration issues
- **Performance Optimization**: Testing scripts and solutions for optimal AI workload performance
- **Integration Troubleshooting**: Root cause analysis and fixes for common AI-PC integration problems

## Repository Structure

```
AI-PC-Enablement/
├── npu-solutions/              # NPU-specific testing and solutions
│   └── npu-int8-support-testing.py
├── gpu-solutions/              # GPU acceleration testing and fixes
├── cpu-solutions/              # CPU optimization testing and solutions
├── multi-platform/             # Cross-platform compatibility tests
├── common-issues/              # Frequently encountered problems and solutions
└── benchmarking/               # Performance testing and validation tools
```

## Current Solutions

### NPU Acceleration
- **File**: `npu-solutions/npu-int8-support-testing.py`
- **Problem**: INT8 quantization compatibility issues on Lunar Lake NPU
- **Solution**: Comprehensive testing of FakeQuantize, native INT8, and weight-only quantization approaches
- **Features**:
  - Multiple quantization strategy testing
  - Automatic device compatibility detection
  - Performance comparison and recommendations
  - Fallback mechanism validation

### GPU Acceleration (Planned)
- Model inference optimization
- Memory management solutions
- Multi-GPU scaling tests

### CPU Acceleration (Planned)
- Threading optimization validation
- SIMD instruction utilization tests
- Memory bandwidth optimization

##  Getting Started

### Prerequisites
- Python 3.9+
- OpenVINO Runtime
- Intel NPU drivers (for NPU testing)
- NumPy

### Installation
```bash
pip install openvino numpy
```

### Running Tests
```bash
# NPU INT8 quantization testing
cd npu-solutions
python npu-int8-support-testing.py

# Future: GPU and CPU test scripts will follow similar patterns
```

##  Supported Platforms

- **Primary**: Intel Lunar Lake NPU
- **Fallback**: CPU execution
- **Future**: Additional Intel AI accelerators

##  Contribution Methodology

This repository follows a problem-solution approach:

### Problem Identification
1. **Symptom Analysis**: Document the specific AI acceleration issue
2. **Root Cause Investigation**: Identify underlying technical causes
3. **Platform Analysis**: Determine which compute platforms are affected

### Solution Development
1. **Test Script Creation**: Develop comprehensive testing algorithms
2. **Multiple Approach Testing**: Implement and validate different solution strategies
3. **Performance Validation**: Benchmark solutions against baseline performance
4. **Documentation**: Create clear problem-solution mapping

### Repository Organization
Solutions are organized by:
1. **Compute Platform** (NPU, GPU, CPU, Multi-platform)
2. **Problem Category** (Quantization, Memory, Performance, Compatibility)
3. **Solution Complexity** (Quick fixes, Comprehensive solutions, Advanced optimizations)

## Solution Categories

### Current Solutions
-  **NPU INT8 Quantization**: Multiple strategy testing and validation

### In Development  
-  **GPU Memory Optimization**: VRAM usage and allocation testing
-  **CPU Threading Efficiency**: Multi-core utilization validation
-  **Cross-Platform Compatibility**: Unified inference testing
-  **Model Format Conversion**: ONNX, OpenVINO, PyTorch compatibility
-  **Performance Profiling**: Latency and throughput analysis tools

### Common Issue Categories
- **Quantization Problems**: INT8, FP16, mixed precision issues
- **Memory Management**: OOM errors, memory leak detection
- **Driver Compatibility**: Version conflicts and requirements
- **Performance Bottlenecks**: Suboptimal acceleration utilization
- **Model Loading Issues**: Format compatibility and conversion problems

## Troubleshooting

Common issues and solutions will be documented here as they are encountered and resolved.

##  Support

For AI acceleration issues or to contribute solutions:
1. Check existing solutions in platform-specific directories
2. Review common issues documentation
3. Run relevant test scripts to validate the problem
4. Create detailed issue reports with test results


---

**Last Updated**: July 23, 2025  
**Maintainer**: Morteza Heidari  
**Focus**: AI-PC Acceleration Problem Solving
