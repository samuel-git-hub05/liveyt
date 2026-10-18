$ErrorActionPreference = "Continue"

Write-Host "LiveYT Windows setup check" -ForegroundColor Cyan
Write-Host "--------------------------"

$python = Get-Command py -ErrorAction SilentlyContinue
if ($python) {
    py -3.11 --version
} else {
    Write-Warning "Python Launcher (py) not found. Install Python 3.11 and enable the launcher."
}

foreach ($tool in @("ffmpeg", "ffprobe", "yt-dlp")) {
    $found = Get-Command $tool -ErrorAction SilentlyContinue
    if ($found) {
        Write-Host "$tool : $($found.Source)" -ForegroundColor Green
    } else {
        Write-Warning "$tool not found on PATH (required for media processing / YouTube mode)."
    }
}

if (Test-Path ".venv\Scripts\python.exe") {
    & .\.venv\Scripts\python.exe -c "import torch; print('PyTorch:', torch.__version__, 'CUDA:', torch.cuda.is_available())" 2>$null
    & .\.venv\Scripts\python.exe main.py --list-devices
} else {
    Write-Host "Virtual environment .venv not found. Create it using the README setup steps." -ForegroundColor Yellow
}
