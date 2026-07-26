# Download CiferAI/Cifer-Fraud-Detection-Dataset-AF from Hugging Face
$ErrorActionPreference = "Stop"

$dest = Join-Path $PSScriptRoot "..\data\Cifer-Fraud-Detection-Dataset-AF"
New-Item -ItemType Directory -Force -Path $dest | Out-Null

$baseUrl = "https://huggingface.co/datasets/CiferAI/Cifer-Fraud-Detection-Dataset-AF/resolve/main"
$files = @("README.md") + (1..14 | ForEach-Object {
    "Cifer-Fraud-Detection-Dataset-AF-part-$_-14.csv"
})

foreach ($file in $files) {
    $out = Join-Path $dest $file
    $minCsvBytes = 120MB
    if (Test-Path $out) {
        $size = (Get-Item $out).Length
        $isComplete = ($file -eq "README.md") -or ($size -ge $minCsvBytes)
        if ($isComplete) {
            Write-Host "Skipping $file (complete, $([math]::Round($size / 1MB, 1)) MB)"
            continue
        }
        Write-Host "Resuming $file (partial, $([math]::Round($size / 1MB, 1)) MB) ..."
    } else {
        Write-Host "Downloading $file ..."
    }
    curl.exe --ssl-no-revoke -L --fail --retry 5 --retry-delay 3 -C - `
        -o $out `
        "$baseUrl/$file"
    if ($LASTEXITCODE -ne 0) {
        Write-Warning "Failed to download $file (exit code $LASTEXITCODE)"
        continue
    }
    $size = (Get-Item $out).Length
    Write-Host "  Done ($([math]::Round($size / 1MB, 1)) MB)"
}

Write-Host ""
Write-Host "All files saved to: $dest"
