from __future__ import annotations

import importlib.util
import os
from pathlib import Path

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension


ROOT = Path(__file__).parent


def _is_hip_build() -> bool:
    import torch

    return bool(getattr(torch.version, "hip", None))


def _check_toolchain() -> None:
    path = ROOT / "python" / "freetoken" / "kernel" / "_toolchain.py"
    spec = importlib.util.spec_from_file_location("_freetoken_toolchain", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.check_nvcc_matches_torch()


def _cuda_runtime_paths() -> tuple[list[str], list[str]]:
    # HIP torch builds expose no CUDA_HOME (torch.utils.cpp_extension leaves it
    # None when torch.version.cuda is falsy); the ROCm prefix comes from
    # ROCM_HOME instead, with the usual env/default fallbacks.
    if _is_hip_build():
        from torch.utils.cpp_extension import ROCM_HOME

        home = ROCM_HOME or os.environ.get("ROCM_PATH") or os.environ.get("ROCM_HOME")
        if home is None and Path("/opt/rocm").is_dir():
            home = "/opt/rocm"
        if home is None:
            raise RuntimeError(
                "A ROCm prefix is required to build the freetoken kernel "
                "extensions on a HIP torch build (set ROCM_PATH)."
            )
    else:
        from torch.utils.cpp_extension import CUDA_HOME

        if CUDA_HOME is None:
            raise RuntimeError(
                "CUDA_HOME is required to build freetoken.kernel._pinned_tensor "
                "because it links against the CUDA runtime API."
            )
        home = CUDA_HOME
    runtime_home = Path(home)
    library_dirs = [str(runtime_home / "lib64")]
    if (runtime_home / "lib").exists():
        library_dirs.append(str(runtime_home / "lib"))
    return [str(runtime_home / "include")], library_dirs


cuda_include_dirs, cuda_library_dirs = _cuda_runtime_paths()
_check_toolchain()

# HIP (ROCm) builds: CUDA_HOME is the ROCm prefix on a HIP torch build (torch's
# cpp_extension resolves it from ROCM_HOME/ROCM_PATH), the extensions consume
# the HIP runtime API through a source-level name shim (see csrc/*.cpp), and --
# because CppExtension compiles with the host compiler, not hipcc --
# __HIP_PLATFORM_AMD__ must be defined here for that shim to engage.
if _is_hip_build():
    runtime_libs = ["amdhip64"]
    platform_args = ["-D__HIP_PLATFORM_AMD__"]
else:
    runtime_libs = ["cudart"]
    platform_args = []


setup(
    ext_modules=[
        CppExtension(
            name="freetoken.kernel._pinned_tensor",
            sources=[
                "python/freetoken/kernel/csrc/pinned_tensor.cpp",
            ],
            include_dirs=cuda_include_dirs,
            library_dirs=cuda_library_dirs,
            libraries=runtime_libs,
            extra_compile_args=["-O3", "-std=c++17"] + platform_args,
        ),
        # CPU-compute MoE executor for --moe-backend cpu. Links the GPU runtime for the
        # cudaLaunchHostFunc submit/sync graph nodes; the bf16 GEMV microkernels
        # use per-function target attributes (avx512bf16/avx512f) + a runtime
        # __builtin_cpu_supports dispatch, so the single binary stays portable
        # (scalar fallback) -- no global -march is set.
        CppExtension(
            name="freetoken.kernel._cpu_moe",
            sources=[
                "python/freetoken/kernel/csrc/cpu_moe/cpu_moe_ext.cpp",
            ],
            include_dirs=cuda_include_dirs,
            library_dirs=cuda_library_dirs,
            libraries=runtime_libs,
            extra_compile_args=["-O3", "-std=c++17", "-pthread"] + platform_args,
        ),
    ],
    cmdclass={"build_ext": BuildExtension.with_options(use_ninja=True)},
)
