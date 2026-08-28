# ROCm 10.0 / Qwen4Exp experimental integration

Date: 2026-08-28

## Git state

- Base: `6d6b697cd53885de8f6f4e1b80902e3559817d7b` (`origin/master`)
- Qwen4Exp base present: `6c84c7d5d8833c6e0df69628f75a0f599797934e`
- Branch/worktree: `qwen4exp-rocm10-prs` / this directory
- PR #27803: `e756c8f6` (source `019a25d7`)
- PR #27836: `93f8be2e`, `10981d2b`, `bd3932df` (sources `d303eec9`, `d7262001`, `1d8de7c1`)
- PR #27825: `5572930a` (source `4c34edd7`)

All changes were cherry-picked in dependency order. `git diff --check` passes.
The original `/home/ignacia/llama.cpp` and
`/home/ignacia/llama-qwen38-rocmfp4-pr27742` worktrees remain unmodified by
this integration.

## ROCm status

- Debian 13, kernel `7.1.3+deb13-amd64`
- Preserved installation: `/opt/rocm-7.2.4`
- Active link resolves to `/opt/rocm-7.2.4`
- ROCm 10.0.0 was not installed: this execution environment has no
  `/dev/kfd` or `/dev/dri`, the user is not in `video`/`render`, and `/opt` is
  not writable by the user. No destructive replacement was attempted.
- `rocminfo`/HIP and real `gfx1151` validation therefore remain pending on the
  host with the GPU and appropriate privileges.

## Build validation

CPU fallback build (separate directory):

```text
build-cpu-validation
GGML_HIP=OFF, LLAMA_BUILD_SERVER=ON, BUILD_SHARED_LIBS=ON,
GGML_NATIVE=OFF, Release
```

Passed:

- `llama-server`
- `test-llama-archs`
- `llama-server --list-devices` (reports no devices in this container)

The requested HIP configuration was created in `build-rocm10-gfx1151` but
configuration stopped because `/opt/rocm-10.0.0/llvm/bin/clang++` does not
exist and CMake could not find a ROCm root. No HIP binary was produced.

The UI asset download was unavailable due to DNS, so the CPU server was built
without embedded UI assets.

## Pending GPU validation

On the actual host, install ROCm 10.0 using AMD's Debian 13 runfile procedure,
keeping `/opt/rocm-7.2.4` available for rollback, then rerun the planned HIP
build for `gfx1151`. Only after that build and normal Qwen4Exp loading pass
should MTP (`--spec-type draft-mtp`) and AllReduce be benchmarked.
