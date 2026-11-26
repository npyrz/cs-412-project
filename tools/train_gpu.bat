@echo off
title CS412 LIAR - transformer training (leave this window open)
cd /d D:\412
set LOG=D:\412\logs\setup.log
set HF_HOME=D:\412\.hf_cache
set TQDM_DISABLE=1
set PYTHONUTF8=1
set HF_HUB_DISABLE_SYMLINKS_WARNING=1
if not exist logs mkdir logs
echo [%date% %time%] setup start > %LOG%

if not exist .venv\Scripts\python.exe (
  echo Creating Python environment...
  py -3.13 -m venv .venv >> %LOG% 2>&1
)
echo Installing PyTorch with CUDA (about 3 GB, first time only)...
.venv\Scripts\python -m pip install -q -U pip >> %LOG% 2>&1
.venv\Scripts\python -m pip install -q torch --index-url https://download.pytorch.org/whl/cu128 >> %LOG% 2>&1
echo Installing transformers and friends...
.venv\Scripts\python -m pip install -q "transformers>=4.46" datasets accelerate scikit-learn pandas >> %LOG% 2>&1
.venv\Scripts\python -c "import torch,transformers;print('torch',torch.__version__,'transformers',transformers.__version__,'CUDA',torch.cuda.is_available(),torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')" >> %LOG% 2>&1
.venv\Scripts\python -c "import torch,sys;sys.exit(0 if torch.cuda.is_available() else 1)"
if errorlevel 1 (
  echo SETUP FAILED: CUDA not available. See logs\setup.log >> %LOG%
  echo SETUP FAILED: CUDA not available. See D:\412\logs\setup.log
  pause
  exit /b 1
)
echo SETUP OK >> %LOG%
echo.
echo Setup OK. Training starts now. Progress: D:\412\logs\train.log
echo Leave this window open and keep the laptop plugged in and awake.
echo.
cd src
..\.venv\Scripts\python run_grid.py
echo.
echo Training finished. You can close this window.
pause
