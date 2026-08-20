@echo off
chcp 65001 >nul
setlocal EnableExtensions
where ffmpeg >nul 2>nul || (echo [ERROR] ffmpeg not found in PATH & pause & exit /b 1)
if "%~1"=="" (
  echo Drag and drop a video onto this .bat
  pause
  exit /b 1
)
set "IN=%~1"

powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "Add-Type -AssemblyName System.Windows.Forms;" ^
  "$f = New-Object System.Windows.Forms.Form;" ^
  "$f.Text = 'Output preset';" ^
  "$f.Size = New-Object System.Drawing.Size(360,230);" ^
  "$f.StartPosition = 'CenterScreen';" ^
  "$f.FormBorderStyle = 'FixedDialog';" ^
  "$f.MaximizeBox = $false;" ^
  "$font = New-Object System.Drawing.Font('Segoe UI',11);" ^
  "$b1 = New-Object System.Windows.Forms.Button;" ^
  "$b1.Text = '1920 x 1080  (16:9)';" ^
  "$b1.Font = $font;" ^
  "$b1.Size = New-Object System.Drawing.Size(320,40);" ^
  "$b1.Location = New-Object System.Drawing.Point(15,15);" ^
  "$b1.Add_Click({ $f.Tag = '1'; $f.Close() });" ^
  "$b2 = New-Object System.Windows.Forms.Button;" ^
  "$b2.Text = '1280 x 720  (16:9)';" ^
  "$b2.Font = $font;" ^
  "$b2.Size = New-Object System.Drawing.Size(320,40);" ^
  "$b2.Location = New-Object System.Drawing.Point(15,65);" ^
  "$b2.Add_Click({ $f.Tag = '2'; $f.Close() });" ^
  "$b3 = New-Object System.Windows.Forms.Button;" ^
  "$b3.Text = '720 x 1280  (9:16)';" ^
  "$b3.Font = $font;" ^
  "$b3.Size = New-Object System.Drawing.Size(320,40);" ^
  "$b3.Location = New-Object System.Drawing.Point(15,115);" ^
  "$b3.Add_Click({ $f.Tag = '3'; $f.Close() });" ^
  "$f.Controls.AddRange(@($b1,$b2,$b3));" ^
  "[void]$f.ShowDialog();" ^
  "Write-Output $f.Tag" > "%TEMP%\_preset.txt"

set /p PRESET=<"%TEMP%\_preset.txt"
del "%TEMP%\_preset.txt" >nul 2>nul

if "%PRESET%"=="1" set "W=1920" & set "H=1080" & set "TAG=1080p"
if "%PRESET%"=="2" set "W=1280" & set "H=720"  & set "TAG=720p"
if "%PRESET%"=="3" set "W=720"  & set "H=1280" & set "TAG=720x1280"
if "%PRESET%"=="" (echo No preset selected & pause & exit /b 1)

set "OUT=%~dpn1_%TAG%.mp4"

set "VF=fps=24,scale=%W%:%H%:force_original_aspect_ratio=decrease:in_range=tv:out_range=tv,pad=%W%:%H%:(ow-iw)/2:(oh-ih)/2,format=yuv420p"

ffmpeg -y -hide_banner -loglevel warning -stats ^
  -i "%IN%" ^
  -map 0:v:0 -map 0:a? ^
  -vf "%VF%" ^
  -c:v libx264 -pix_fmt yuv420p -preset medium -crf 17 ^
  -profile:v high -level 4.1 ^
  -colorspace bt709 -color_primaries bt709 -color_trc bt709 -color_range tv ^
  -c:a copy ^
  -movflags +faststart ^
  "%OUT%"

if errorlevel 1 (
  echo.
  echo [WARN] audio copy failed, retrying with AAC re-encode
  ffmpeg -y -hide_banner -loglevel warning -stats ^
    -i "%IN%" ^
    -map 0:v:0 -map 0:a? ^
    -vf "%VF%" ^
    -c:v libx264 -pix_fmt yuv420p -preset medium -crf 17 ^
    -profile:v high -level 4.1 ^
    -colorspace bt709 -color_primaries bt709 -color_trc bt709 -color_range tv ^
    -c:a aac -b:a 192k ^
    -movflags +faststart ^
    "%OUT%"
)

pause
endlocal