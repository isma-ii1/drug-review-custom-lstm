param(
    [string]$TrainCsv = "$env:USERPROFILE\Downloads\archive\drugsComTrain_raw.csv",
    [string]$TestCsv = "$env:USERPROFILE\Downloads\archive\drugsComTest_raw.csv",
    [switch]$Train,
    [switch]$BuildDocuments
)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
if (-not (Test-Path -LiteralPath '.venv\Scripts\python.exe')) {
    if (Get-Command uv -ErrorAction SilentlyContinue) {
        uv venv --python 3.11 .venv
        if ($LASTEXITCODE) { throw 'Virtual environment creation failed' }
        uv pip install --python .venv\Scripts\python.exe -e '.[dev]'
    } else {
        py -3.11 -m venv .venv
        if ($LASTEXITCODE) { throw 'Install Python 3.11 or uv first' }
        .venv\Scripts\python.exe -m pip install -e '.[dev]'
    }
    if ($LASTEXITCODE) { throw 'Dependency installation failed' }
}
.venv\Scripts\python.exe -m pytest -q
if ($LASTEXITCODE) { throw 'Implementation checks failed' }
if ($Train) {
    if (-not (Test-Path -LiteralPath 'data\processed\encoded.npz')) {
        .venv\Scripts\python.exe -m drug_review.experiment prepare --train $TrainCsv --test $TestCsv
        if ($LASTEXITCODE) { throw 'Data preparation failed' }
    }
    .venv\Scripts\python.exe -m drug_review.experiment train --threads 2
    if ($LASTEXITCODE) { throw 'Training or evaluation failed' }
}
if ($BuildDocuments) {
    .venv\Scripts\python.exe scripts\build_deliverables.py --repo-url https://github.com/isma-ii1/drug-review-custom-lstm
    if ($LASTEXITCODE) { throw 'Document generation failed' }
}
Write-Output 'Project checks finished. Open README.md for commands and output locations.'
