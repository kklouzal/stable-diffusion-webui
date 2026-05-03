from pathlib import Path


def replace_once(text: str, old: str, new: str, label: str) -> str:
    if old not in text:
        raise RuntimeError(f"anchor not found while patching {label}: {old!r}")
    return text.replace(old, new, 1)

cpp_extension = Path('/usr/local/lib/python3.12/dist-packages/torch/utils/cpp_extension.py')
if cpp_extension.exists():
    text = cpp_extension.read_text()
    text = text.replace(
        "logger.warning(\"No CUDA runtime is found, using CUDA_HOME='%s'\", cuda_home)",
        "logger.debug(\"No CUDA runtime is found, using CUDA_HOME='%s'\", cuda_home)",
    )
    text = text.replace(
        "logger.warning('There are no %s version bounds defined for CUDA version %s', compiler_name, cuda_str_version)",
        "logger.debug('There are no %s version bounds defined for CUDA version %s', compiler_name, cuda_str_version)",
    )
    cpp_extension.write_text(text)

root = Path('/opt/build/SageAttention')
for rel in ['setup.py', 'sageattention3_blackwell/setup.py']:
    path = root / rel
    text = path.read_text()
    text = text.replace('-std=c++17', '-std=c++20')
    text = text.replace('--threads=8', '--threads=4')
    text = text.replace('        "-Xptxas=-v",\n', '')
    text = text.replace('        "--ptxas-options=--verbose,--warn-on-local-memory-usage",  # printing out number of registers\n', '')
    text = text.replace('        "-lineinfo",\n', '')
    text = text.replace(
        '        "-diag-suppress=174",\n',
        '        "-diag-suppress=174",\n'
        '        "-Xcudafe=--diag_suppress=68",\n'
        '        "-Xcompiler=-Wno-interference-size",\n'
        '        "-Xcompiler=-Wno-narrowing",\n',
    )

    if rel == 'sageattention3_blackwell/setup.py':
        text = text.replace(
            '"cxx": ["-O3", "-std=c++20"],',
            '"cxx": ["-O3", "-std=c++20", "-Wno-interference-size", "-Wno-narrowing"],',
        )
        text = text.replace(
            '        "-DCUTLASS_DEBUG_TRACE_LEVEL=0",',
            '        "-Xcudafe=--diag_suppress=68",\n'
            '        "-Xcompiler=-Wno-interference-size",\n'
            '        "-Xcompiler=-Wno-narrowing",\n'
            '        "-DCUTLASS_DEBUG_TRACE_LEVEL=0",',
        )
        arch_probe = '    cc_major, cc_minor = torch.cuda.get_device_capability()'
        forced_arch = '\n'.join([
            '    forced_cc = os.getenv("SAGEATTN3_CUDA_ARCH") or os.getenv("TORCH_CUDA_ARCH_LIST", "")',
            '    if forced_cc:',
            '        normalized_cc = forced_cc.split(";")[0].strip().lower().removesuffix("a")',
            '        cc_major, cc_minor = [int(part) for part in normalized_cc.split(".", 1)]',
            '    else:',
            '        cc_major, cc_minor = torch.cuda.get_device_capability()',
        ])
        text = replace_once(text, arch_probe, forced_arch, str(path))

    path.write_text(text)

api = root / 'sageattention3_blackwell/sageattn3/blackwell/api.cu'
if api.exists():
    text = api.read_text().replace(
        'at::cuda::CUDAGuard device_guard{(char)q.get_device()};',
        'at::cuda::CUDAGuard device_guard{q.get_device()};',
    )
    api.write_text(text)

fused = root / 'csrc/fused/fused.cu'
text = fused.read_text().replace('  float block_sum_val;', '  float block_sum_val = 0.0f;')
fused.write_text(text)

manifest = root / 'MANIFEST.in'
if manifest.exists():
    text = manifest.read_text()
    text = text.replace('recursive-include csrc *.h *.hpp *.cuh *.cu *.cpp\n', 'recursive-include csrc *.h *.cuh *.cu *.cpp\n')
    text = text.replace('global-exclude __pycache__\n', '')
    text = text.replace('global-exclude *.py[cod]\n', '')
    manifest.write_text(text)

utils = root / 'csrc/utils.cuh'
text = utils.read_text()
if not text.endswith('\n'):
    utils.write_text(text + '\n')
