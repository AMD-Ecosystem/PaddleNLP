# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import subprocess

# Compatibility patch for Paddle DCU 3.4.0 + ROCm 10.x
# ROCm 10.0+ removed __HIP_PLATFORM_HCC__ in favour of __HIP_PLATFORM_AMD__,
# and replaced the -amdgpu-target= flag with --offload-arch=.
# Paddle's cpp_extension.py still emits the old flags -- patch them at import time.
# SYMPTOM: #error ("Must define exactly one of __HIP_PLATFORM_AMD__ or __HIP_PLATFORM_NVIDIA__")
# ROOT CAUSE: -D__HIP_PLATFORM_HCC__ passed by Paddle, but ROCm 10.x hip_runtime.h
#             only accepts __HIP_PLATFORM_AMD__ or __HIP_PLATFORM_NVIDIA__.
#             Also -amdgpu-target= is an unknown argument in ROCm 10.x clang++.
# FIX: Patch Paddle's extension_utils and cpp_extension in-process before building.
def _patch_paddle_rocm10_compat():
    import importlib, sys
    try:
        import paddle.utils.cpp_extension.extension_utils as eu
        import paddle.utils.cpp_extension.cpp_extension as ce
        import inspect, types

        # Patch extension_utils: replace -amdgpu-target= with --offload-arch=
        eu_src = inspect.getsource(eu)
        if '-amdgpu-target=' in eu_src:
            # We cannot reload the source, but we can monkey-patch the functions
            # that emit these flags. Look for the flag list in module-level code.
            pass  # Will handle via sed-like substitution below

        # Direct attribute patch approach: find all string attributes/lists
        # that contain '-amdgpu-target=' or '-D__HIP_PLATFORM_HCC__'
        for mod in [eu, ce]:
            for attr_name in dir(mod):
                try:
                    val = getattr(mod, attr_name)
                except Exception:
                    continue
                if isinstance(val, list):
                    new_val = []
                    changed = False
                    for item in val:
                        if isinstance(item, str):
                            if item.startswith('-amdgpu-target='):
                                item = item.replace('-amdgpu-target=', '--offload-arch=')
                                changed = True
                            elif item == '-D__HIP_PLATFORM_HCC__':
                                item = '-D__HIP_PLATFORM_AMD__'
                                changed = True
                        new_val.append(item)
                    if changed:
                        setattr(mod, attr_name, new_val)
    except Exception as e:
        print(f"WARNING: paddle ROCm10 compat patch failed: {e}")

# Also patch at the file level by editing the installed module files
import os, re as _re

def _patch_paddle_files():
    """Directly patch the Paddle extension Python files for ROCm 10.x compat."""
    try:
        import paddle.utils.cpp_extension.extension_utils as eu
        import paddle.utils.cpp_extension.cpp_extension as ce
        eu_path = eu.__file__
        ce_path = ce.__file__

        # Patch extension_utils.py:
        # 1. -amdgpu-target=X -> --offload-arch=X  (ROCm 10.x flag rename)
        # 2. gfx906/gfx926/gfx928 (DCU-specific, unsupported in public ROCm) -> gfx942
        with open(eu_path, 'r') as f:
            content = f.read()
        changed = False
        if '-amdgpu-target=' in content:
            content = content.replace('-amdgpu-target=', '--offload-arch=')
            changed = True
        # Replace DCU-specific archs (gfx906/926/928) with gfx942 (MI300X for public ROCm)
        # The extension_utils.py list looks like: '--offload-arch=gfx906', '--offload-arch=gfx926', ...
        if '--offload-arch=gfx906' in content or '--offload-arch=gfx926' in content:
            # Replace gfx906 with gfx942 (first entry becomes the target)
            content = content.replace('--offload-arch=gfx906', '--offload-arch=gfx942')
            # Remove gfx926 and gfx928 entries entirely (empty string replacement)
            content = content.replace(', \'--offload-arch=gfx926\'', '')
            content = content.replace('\'--offload-arch=gfx926\',', '')
            content = content.replace('--offload-arch=gfx926', '--offload-arch=gfx942')
            content = content.replace(', \'--offload-arch=gfx928\'', '')
            content = content.replace('\'--offload-arch=gfx928\',', '')
            content = content.replace('--offload-arch=gfx928', '--offload-arch=gfx942')
            changed = True
        # Also handle if -amdgpu-target= wasn't already replaced
        if '-amdgpu-target=gfx906' in content or '-amdgpu-target=gfx926' in content:
            content = content.replace('-amdgpu-target=gfx906', '--offload-arch=gfx942')
            content = content.replace('-amdgpu-target=gfx926', '--offload-arch=gfx942')
            content = content.replace('-amdgpu-target=gfx928', '--offload-arch=gfx942')
            changed = True
        if changed:
            with open(eu_path, 'w') as f:
                f.write(content)
            print(f"Patched {eu_path}: amdgpu-target -> offload-arch + gfx942")

        # Patch cpp_extension.py: __HIP_PLATFORM_HCC__ -> __HIP_PLATFORM_AMD__
        with open(ce_path, 'r') as f:
            content = f.read()
        if '__HIP_PLATFORM_HCC__' in content:
            content = content.replace('__HIP_PLATFORM_HCC__', '__HIP_PLATFORM_AMD__')
            with open(ce_path, 'w') as f:
                f.write(content)
            print(f"Patched {ce_path}: __HIP_PLATFORM_HCC__ -> __HIP_PLATFORM_AMD__")
    except Exception as e:
        print(f"WARNING: paddle file patch failed: {e}")

_patch_paddle_files()

# Create a glog stub so Paddle's miopen.h header compiles without a glog installation.
# SYMPTOM: fatal error: glog/logging.h: No such file or directory
# ROOT CAUSE: Paddle's miopen.h unconditionally includes glog/logging.h, but
#             the manylinux ROCm 10.1 container does not have glog installed.
# FIX: Create a stub header with no-op logging macros in a local include dir.
def _create_glog_stub():
    import os, pathlib
    stub_dir = pathlib.Path('/tmp/glog_stub/glog')
    stub_dir.mkdir(parents=True, exist_ok=True)
    stub_file = stub_dir / 'logging.h'
    if not stub_file.exists():
        stub_file.write_text("""
#pragma once
#include <iostream>
// Minimal glog stub for Paddle ROCm10.x build compatibility
#define GLOG_NO_ABBREVIATED_SEVERITIES
namespace google {
    struct LogMessage {
        std::ostream& stream() { return std::cout; }
        ~LogMessage() { std::cout << std::endl; }
    };
    struct LogMessageFatal : LogMessage { ~LogMessageFatal() { abort(); } };
    struct LogMessageVoidify { void operator&(std::ostream&) {} };
    struct NullStream { template<typename T> NullStream& operator<<(T) { return *this; } };
}
#define LOG_INFO  google::LogMessage().stream()
#define LOG_WARNING google::LogMessage().stream()
#define LOG_ERROR google::LogMessage().stream()
#define LOG_FATAL google::LogMessageFatal().stream()
#define LOG(severity) LOG_##severity
#define VLOG(n) if(false) google::NullStream()
#define DLOG(severity) if(false) google::NullStream()
#define CHECK(cond) if(!(cond)) LOG(FATAL) << "CHECK failed: " #cond " "
#define CHECK_EQ(a,b) CHECK((a)==(b))
#define CHECK_NE(a,b) CHECK((a)!=(b))
#define CHECK_LT(a,b) CHECK((a)<(b))
#define CHECK_LE(a,b) CHECK((a)<=(b))
#define CHECK_GT(a,b) CHECK((a)>(b))
#define CHECK_GE(a,b) CHECK((a)>=(b))
#define DCHECK(cond) ((void)0)
#define DCHECK_EQ(a,b) ((void)0)
#define DCHECK_NE(a,b) ((void)0)
""")
        print(f"Created glog stub at {stub_file}")
    return str(stub_dir.parent)

_glog_stub_dir = _create_glog_stub()

# Create hiprand stub: helper.h includes <hiprand.h> and <hiprand_kernel.h> unconditionally
# but none of the PaddleNLP custom ops actually call hiprand functions.
# hiprand is not installed in the ROCm 10.1 manylinux container.
# SYMPTOM: fatal error: 'hiprand.h' file not found
# ROOT CAUSE: helper.h unconditionally includes hiprand, but hiprand library is not
#             installed in the manylinux 10.1 container (only in full ROCm GPU installs).
#             The ops compiled here do not use hiprand at runtime.
# FIX: Create a minimal stub with just enough declarations to satisfy the compiler.
def _create_hiprand_stub():
    import os, pathlib
    stub_dir = pathlib.Path('/tmp/hiprand_stub')
    stub_dir.mkdir(parents=True, exist_ok=True)
    # hiprand.h stub
    (stub_dir / 'hiprand.h').write_text("""
#pragma once
// Minimal hiprand stub for PaddleNLP csrc compilation (hiprand not used at runtime)
typedef struct hiprandGenerator_st* hiprandGenerator_t;
typedef struct hiprandRngState_t { unsigned long long seed; } hiprandRngStateHost_t;
typedef enum {
    HIPRAND_RNG_PSEUDO_DEFAULT = 400, HIPRAND_RNG_PSEUDO_XORWOW = 401,
    HIPRAND_RNG_PSEUDO_MRG32K3A = 402, HIPRAND_RNG_PSEUDO_MTGP32 = 403,
    HIPRAND_RNG_PSEUDO_MT19937 = 404, HIPRAND_RNG_QUASI_DEFAULT = 500,
    HIPRAND_RNG_PSEUDO_PHILOX4_32_10 = 405,
} hiprandRngType_t;
typedef enum { HIPRAND_STATUS_SUCCESS = 0, HIPRAND_STATUS_INTERNAL_ERROR = 999 } hiprandStatus_t;
static inline hiprandStatus_t hiprandCreateGenerator(hiprandGenerator_t*, hiprandRngType_t) { return HIPRAND_STATUS_SUCCESS; }
static inline hiprandStatus_t hiprandDestroyGenerator(hiprandGenerator_t) { return HIPRAND_STATUS_SUCCESS; }
static inline hiprandStatus_t hiprandSetPseudoRandomGeneratorSeed(hiprandGenerator_t, unsigned long long) { return HIPRAND_STATUS_SUCCESS; }
static inline hiprandStatus_t hiprandGenerateUniform(hiprandGenerator_t, float*, size_t) { return HIPRAND_STATUS_SUCCESS; }
static inline hiprandStatus_t hiprandGenerateNormal(hiprandGenerator_t, float*, size_t, float, float) { return HIPRAND_STATUS_SUCCESS; }
""")
    # hiprand_kernel.h stub
    (stub_dir / 'hiprand_kernel.h').write_text("""
#pragma once
// Minimal hiprand_kernel stub for PaddleNLP csrc compilation (not used at runtime)
#include "hiprand.h"
typedef struct { unsigned int s[4]; } hiprandStateXORWOW_t;
typedef hiprandStateXORWOW_t hiprandState_t;
typedef struct { unsigned long long s[2]; } hiprandStatePhilox4_32_10_t;
""")
    print(f"Created hiprand stub at {stub_dir}")
    return str(stub_dir)

_hiprand_stub_dir = _create_hiprand_stub()

# Create a pre-include header to fix rocprim host-compilation issue with g++
# SYMPTOM: arch.hpp:49:12: error: '__builtin_amdgcn_wavefrontsize' was not declared in this scope
# ROOT CAUSE: ROCm 10.1 rocprim 4.6 arch.hpp calls __builtin_amdgcn_wavefrontsize() inside
#             wavefront::size() without guarding it for host-only (non-device) compilation.
#             When .cc files include paddle/extension.h which chains to rocprim via thrust,
#             g++ (the host C++ compiler) sees this device-only builtin and fails.
# FIX: Provide a preprocessor macro override before the header is included.
#      __builtin_amdgcn_wavefrontsize is a clang AMD GPU extension; on g++ we define it
#      as a macro returning 64 (wave64 for CDNA) so the header compiles on host paths.
def _create_rocprim_host_fix():
    import os, pathlib
    fix_dir = pathlib.Path('/tmp/rocprim_host_fix')
    fix_dir.mkdir(parents=True, exist_ok=True)
    fix_file = fix_dir / 'rocprim_host_fix.h'
    fix_file.write_text("""
#pragma once
// Fix: rocprim 4.6 arch.hpp uses __builtin_amdgcn_wavefrontsize without guarding for host
// This macro is only needed when NOT in device-compilation mode (i.e., when g++ compiles .cc)
// Safe to define as 64 here because rocprim host paths never actually call wavefront::size()
// at runtime in our build (it's only called from device kernels through hipcc).
#if defined(__GNUC__) && !defined(__clang__) && !defined(__HIP__)
#define __builtin_amdgcn_wavefrontsize() (64u)
#endif
""")
    print(f"Created rocprim host fix at {fix_file}")
    return str(fix_dir / 'rocprim_host_fix.h')

_rocprim_fix_file = _create_rocprim_host_fix()

# Now reload the patched modules
import importlib
try:
    import paddle.utils.cpp_extension.extension_utils as _eu
    import paddle.utils.cpp_extension.cpp_extension as _ce
    importlib.reload(_eu)
    importlib.reload(_ce)
except Exception as e:
    print(f"WARNING: module reload failed: {e}")

from paddle.utils.cpp_extension import CUDAExtension, setup


def update_git_submodule():
    try:
        subprocess.run(["git", "submodule", "update", "--init"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while updating git submodule: {str(e)}")
        raise


update_git_submodule()
# Add glog and hiprand stubs to the include path, and rocprim fix as pre-include for cxx
glog_include = f"-I{_glog_stub_dir}"
hiprand_include = f"-I{_hiprand_stub_dir}"
rocprim_fix_include = f"-include{_rocprim_fix_file}"

setup(
    name="paddlenlp_ops",
    ext_modules=CUDAExtension(
        sources=[
            "./gpu/save_with_output.cc",
            "./gpu/set_value_by_flags.cu",
            "./gpu/token_penalty_multi_scores.cu",
            "./gpu/token_penalty_multi_scores_v2.cu",
            "./gpu/stop_generation_multi_ends.cu",
            "./gpu/fused_get_rope.cu",
            "./gpu/get_padding_offset.cu",
            "./gpu/qkv_transpose_split.cu",
            "./gpu/rebuild_padding.cu",
            "./gpu/transpose_removing_padding.cu",
            "./gpu/write_cache_kv.cu",
            "./gpu/encode_rotary_qk.cu",
            "./gpu/get_padding_offset_v2.cu",
            "./gpu/rebuild_padding_v2.cu",
            "./gpu/set_value_by_flags_v2.cu",
            "./gpu/stop_generation_multi_ends_v2.cu",
            "./gpu/get_output.cc",
            "./gpu/save_with_output_msg.cc",
            "./gpu/write_int8_cache_kv.cu",
            "./gpu/step.cu",
            "./gpu/quant_int8.cu",
            "./gpu/dequant_int8.cu",
            "./gpu/flash_attn_bwd.cc",
            "./gpu/update_inputs_v2.cu",
            "./gpu/set_preids_token_penalty_multi_scores.cu",
        ],
        extra_compile_args={
            "cxx": [
                "-O3",
                glog_include,
                hiprand_include,
                rocprim_fix_include,  # pre-include: defines __builtin_amdgcn_wavefrontsize macro for g++
            ],
            "hipcc": [
                "-O3",
                "--gpu-max-threads-per-block=1024",
                "-U__HIP_NO_HALF_OPERATORS__",
                "-U__HIP_NO_HALF_CONVERSIONS__",
                "-U__HIP_NO_BFLOAT16_OPERATORS__",
                "-U__HIP_NO_BFLOAT16_CONVERSIONS__",
                "-U__HIP_NO_BFLOAT162_OPERATORS__",
                "-U__HIP_NO_BFLOAT162_CONVERSIONS__",
                "-Ithird_party/cutlass/include",
                "-Ithird_party/nlohmann_json/single_include",
                glog_include,
                hiprand_include,
            ],
        },
    ),
)
