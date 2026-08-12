# AGENTS.md — Watermark-Conditioned Diffusion Experiment

## Conda Environment

All Python commands must run under the **WaDiff** conda environment.

- Python path: `D:\Anaconda_envs\envs\wadiff\python.exe`
- Conda env name: `wadiff`

### How to invoke

- Use the full Python path for running scripts:
  ```
  D:\Anaconda_envs\envs\wadiff\python.exe train_watermark_diffusion.py --config configs/watermark_stage1.yaml
  ```
- Use the full pip path for installing packages:
  ```
  D:\Anaconda_envs\envs\wadiff\python.exe -m pip install <package>
  ```
- Never use the system `python` or `python3` command — always use the full path above.

## File Safety

- Do not delete files or directories recursively.
- Do not use `Remove-Item -Recurse`, `rm -rf`, `del /s`, or similar bulk deletion commands.
- Delete only explicitly specified single files when necessary.
