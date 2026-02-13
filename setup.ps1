$ErrorActionPreference = "Stop"

$PythonVersion = "3.12.3"
$VenvDir       = "watsonx-agent-skeletton\venv"
$ReqFile       = "watsonx-agent-skeletton\requirements.txt"
$EnvExample    = "watsonx-agent-skeletton\config\.env.example"
$EnvFile       = "watsonx-agent-skeletton\config\.env"

function Write-Step {
    param([string]$Message)
    Write-Host ""
    Write-Host ">> $Message" -ForegroundColor Cyan
}

# ── 1. Check / install pyenv-win ──────────────
Write-Step "Checking for pyenv..."

if (Get-Command pyenv -ErrorAction SilentlyContinue) {
    Write-Host "   pyenv is already installed."
} else {
    Write-Host "   pyenv not found. Installing pyenv-win..."
    Write-Host ""

    # Try scoop first, then chocolatey, then manual install
    if (Get-Command scoop -ErrorAction SilentlyContinue) {
        Write-Host "   Installing via scoop..."
        scoop install pyenv
    } elseif (Get-Command choco -ErrorAction SilentlyContinue) {
        Write-Host "   Installing via chocolatey..."
        choco install pyenv-win -y
    } else {
        Write-Host "   No package manager found. Installing pyenv-win manually..."
        $PyenvHome = "$env:USERPROFILE\.pyenv"
        if (-not (Test-Path $PyenvHome)) {
            git clone https://github.com/pyenv-win/pyenv-win.git $PyenvHome
        }

        # Add to user PATH
        $PyenvBin = "$PyenvHome\pyenv-win\bin"
        $PyenvShims = "$PyenvHome\pyenv-win\shims"
        $CurrentPath = [Environment]::GetEnvironmentVariable("PATH", "User")
        if ($CurrentPath -notlike "*$PyenvBin*") {
            [Environment]::SetEnvironmentVariable("PATH", "$PyenvBin;$PyenvShims;$CurrentPath", "User")
            $env:PATH = "$PyenvBin;$PyenvShims;$env:PATH"
            Write-Host "   Added pyenv-win to user PATH."
        }
    }

    # Verify
    if (-not (Get-Command pyenv -ErrorAction SilentlyContinue)) {
        Write-Host ""
        Write-Host "ERROR: pyenv is still not available." -ForegroundColor Red
        Write-Host "Please restart your terminal and run this script again," -ForegroundColor Red
        Write-Host "or install pyenv-win manually: https://github.com/pyenv-win/pyenv-win" -ForegroundColor Red
        exit 1
    }
}

# ── 2. Install Python ─────────────────────────
Write-Step "Installing Python $PythonVersion via pyenv..."

pyenv install -q $PythonVersion 2>$null
pyenv local $PythonVersion

$PythonCmd = "python"
$ActualVersion = & $PythonCmd --version 2>&1
Write-Host "   Using: $ActualVersion"

# ── 3. Create virtual environment ─────────────
Write-Step "Creating virtual environment..."

if (Test-Path $VenvDir) {
    Write-Host "   $VenvDir already exists — skipping."
} else {
    & $PythonCmd -m venv $VenvDir
    Write-Host "   Created $VenvDir"
}

# ── 4. Install dependencies ───────────────────
Write-Step "Installing Python dependencies..."

$PipCmd = "$VenvDir\Scripts\pip.exe"
& $PipCmd install --upgrade pip
& $PipCmd install -r $ReqFile

# ── 5. Create .env file ───────────────────────
Write-Step "Setting up environment file..."

if (Test-Path $EnvFile) {
    Write-Host "   $EnvFile already exists — skipping."
} else {
    Copy-Item $EnvExample $EnvFile
    Write-Host "   Created $EnvFile from template."
}

# ── Done ──────────────────────────────────────
Write-Host ""
Write-Host "============================================" -ForegroundColor Green
Write-Host " Setup complete!" -ForegroundColor Green
Write-Host "" -ForegroundColor Green
Write-Host " 1. Fill in your credentials:" -ForegroundColor Green
Write-Host "      $EnvFile" -ForegroundColor Green
Write-Host "" -ForegroundColor Green
Write-Host " 2. Activate the virtual environment:" -ForegroundColor Green
Write-Host "      $VenvDir\Scripts\Activate.ps1" -ForegroundColor Green
Write-Host "" -ForegroundColor Green
Write-Host " 3. Run the agent:" -ForegroundColor Green
Write-Host "      python watsonx-agent-skeletton\main.py" -ForegroundColor Green
Write-Host "============================================" -ForegroundColor Green
