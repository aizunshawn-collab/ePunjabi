# FFmpeg Installation Script for Windows
# This will download and install FFmpeg for Whisper AI

Write-Host "`n============================================" -ForegroundColor Cyan
Write-Host "  FFmpeg Installation for Whisper AI" -ForegroundColor Cyan
Write-Host "============================================`n" -ForegroundColor Cyan

# Step 1: Download FFmpeg
Write-Host "Step 1: Downloading FFmpeg (~80MB, please wait)..." -ForegroundColor Yellow
$ffmpegUrl = "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip"
$downloadPath = "$env:USERPROFILE\Downloads\ffmpeg.zip"

try {
    Invoke-WebRequest -Uri $ffmpegUrl -OutFile $downloadPath -UseBasicParsing
    Write-Host "✅ Download complete!" -ForegroundColor Green
} catch {
    Write-Host "❌ Download failed: $_" -ForegroundColor Red
    Write-Host "`nPlease manually download from:" -ForegroundColor Yellow
    Write-Host $ffmpegUrl -ForegroundColor Cyan
    exit 1
}

# Step 2: Extract FFmpeg
Write-Host "`nStep 2: Extracting FFmpeg to C:\ffmpeg..." -ForegroundColor Yellow
$extractPath = "C:\ffmpeg"

# Create directory if it doesn't exist
if (-not (Test-Path $extractPath)) {
    New-Item -ItemType Directory -Path $extractPath -Force | Out-Null
}

try {
    Expand-Archive -Path $downloadPath -DestinationPath $extractPath -Force
    Write-Host "✅ Extraction complete!" -ForegroundColor Green
} catch {
    Write-Host "❌ Extraction failed: $_" -ForegroundColor Red
    exit 1
}

# Step 3: Find the bin folder
$binFolder = Get-ChildItem -Path $extractPath -Recurse -Directory -Filter "bin" | Select-Object -First 1
if ($binFolder) {
    $ffmpegBinPath = $binFolder.FullName
    Write-Host "✅ Found FFmpeg at: $ffmpegBinPath" -ForegroundColor Green
} else {
    Write-Host "❌ Could not find FFmpeg bin folder" -ForegroundColor Red
    exit 1
}

# Step 4: Add to PATH
Write-Host "`nStep 3: Adding FFmpeg to Windows PATH..." -ForegroundColor Yellow

# Get current PATH
$currentPath = [Environment]::GetEnvironmentVariable("Path", "User")

# Check if already in PATH
if ($currentPath -like "*$ffmpegBinPath*") {
    Write-Host "✅ FFmpeg already in PATH!" -ForegroundColor Green
} else {
    # Add to user PATH
    $newPath = "$currentPath;$ffmpegBinPath"
    [Environment]::SetEnvironmentVariable("Path", $newPath, "User")
    Write-Host "✅ Added to PATH!" -ForegroundColor Green
    
    # Update current session PATH
    $env:Path = "$env:Path;$ffmpegBinPath"
}

# Step 5: Verify installation
Write-Host "`nStep 4: Verifying FFmpeg installation..." -ForegroundColor Yellow
try {
    $ffmpegVersion = & "$ffmpegBinPath\ffmpeg.exe" -version 2>&1 | Select-Object -First 1
    Write-Host "✅ $ffmpegVersion" -ForegroundColor Green
} catch {
    Write-Host "❌ Verification failed: $_" -ForegroundColor Red
    exit 1
}

# Clean up
Write-Host "`nStep 5: Cleaning up..." -ForegroundColor Yellow
Remove-Item $downloadPath -Force -ErrorAction SilentlyContinue
Write-Host "✅ Cleanup complete!" -ForegroundColor Green

# Final message
Write-Host "`n============================================" -ForegroundColor Cyan
Write-Host "  ✅ FFmpeg Installation Complete!" -ForegroundColor Green
Write-Host "============================================`n" -ForegroundColor Cyan

Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "1. Restart your AI backend server (Ctrl+C then run again)" -ForegroundColor White
Write-Host "2. Open: http://localhost:8080/mobile-app.html" -ForegroundColor White
Write-Host "3. Test voice input - should use Whisper AI now!" -ForegroundColor White
Write-Host "`nPress any key to continue..." -ForegroundColor Gray
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
