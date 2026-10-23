# NueroFly launcher (Windows PowerShell).
# Usage (from this folder):   .\run_nuerofly.ps1
#         if Godot is not found: .\run_nuerofly.ps1 -GodotPath "C:\path\to\Godot_v4.7-stable_win64.exe"
param([string]$GodotPath = "", [int]$Port = 8765)
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

# 1. Python
$py = $null
foreach ($c in @("py", "python")) { if (Get-Command $c -ErrorAction SilentlyContinue) { $py = $c; break } }
if (-not $py) { Write-Host "ERROR: Python 3.10+ not found. Install it from python.org (tick 'Add to PATH')." -ForegroundColor Red; exit 1 }

# 2. Virtual environment + dependencies (first run only)
if (-not (Test-Path ".venv\Scripts\python.exe")) {
    Write-Host "Creating virtual environment (first run)..."
    & $py -m venv .venv
}
$venvPy = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
& $venvPy -m pip install --quiet -r requirements.txt
if ($LASTEXITCODE -ne 0) { Write-Host "ERROR: pip install failed. Check your internet connection." -ForegroundColor Red; exit 1 }

# 3. Find Godot
if (-not $GodotPath) {
    foreach ($n in @("godot", "godot4", "Godot")) {
        $cmd = Get-Command $n -ErrorAction SilentlyContinue
        if ($cmd) { $GodotPath = $cmd.Source; break }
    }
}
if (-not $GodotPath -or -not (Test-Path $GodotPath)) {
    Write-Host "ERROR: Godot 4 executable not found." -ForegroundColor Red
    Write-Host "Re-run with:  .\run_nuerofly.ps1 -GodotPath `"C:\full\path\to\Godot_v4.7-stable_win64.exe`""
    exit 1
}

# 4. Start backend, then Godot; always stop the backend when Godot closes.
$env:PYTHONPATH = $PSScriptRoot
$backend = Start-Process -FilePath $venvPy -ArgumentList @("-m", "backend.sim.server", "--port", $Port) -PassThru -NoNewWindow
Start-Sleep -Seconds 1
if ($backend.HasExited) { Write-Host "ERROR: backend failed to start (is port $Port already in use?)." -ForegroundColor Red; exit 1 }
Write-Host "Backend running (PID $($backend.Id)). Starting Godot..."
try {
    & $GodotPath --path (Join-Path $PSScriptRoot "godot") -- "--port=$Port"
} finally {
    if (-not $backend.HasExited) { Stop-Process -Id $backend.Id -Force }
    Write-Host "Backend stopped."
}
