@echo off
set LOG=D:\412\logs\gpu_check.txt
echo === nvidia-smi === > %LOG%
nvidia-smi >> %LOG% 2>&1
echo === python on PATH === >> %LOG%
where python >> %LOG% 2>&1
where py >> %LOG% 2>&1
python --version >> %LOG% 2>&1
py -0p >> %LOG% 2>&1
echo === torch === >> %LOG%
python -c "import torch;print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')" >> %LOG% 2>&1
echo === conda === >> %LOG%
where conda >> %LOG% 2>&1
echo === disk === >> %LOG%
wmic logicaldisk get caption,freespace,size >> %LOG% 2>&1
echo === net === >> %LOG%
curl -s -o NUL -w "huggingface %%{http_code}\n" https://huggingface.co/roberta-base/resolve/main/config.json -L >> %LOG% 2>&1
echo DONE >> %LOG%
