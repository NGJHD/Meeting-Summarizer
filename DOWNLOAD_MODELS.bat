@echo off
setlocal EnableDelayedExpansion
cd /d "%~dp0"

rem ---------------------------------------------------------------------------
rem  Meeting Summariser - model downloader
rem
rem  Run this ONCE on a machine that has internet access, then copy the whole
rem  folder to the offline machine. The application itself never touches the
rem  network; this script exists only so the models do not have to ship inside
rem  the folder.
rem
rem  Needs curl.exe and tar.exe, both included with Windows 10 1803 and later.
rem ---------------------------------------------------------------------------

echo.
echo   Meeting Summariser - downloading models
echo   ---------------------------------------
echo.
echo   Total download: about 29 GB across 6 models. This will take a while.
echo   Already-downloaded files are skipped, so it is safe to re-run
echo   this script if the connection drops.
echo.

if not exist "models" mkdir "models"

where curl.exe >nul 2>&1
if errorlevel 1 (
  echo   ERROR: curl.exe was not found.
  echo   It ships with Windows 10 version 1803 and later. On an older
  echo   version, download the five files listed in BUILD_NOTES.md by hand.
  echo.
  pause
  exit /b 1
)

set "FAILED="

rem  get <target file> <minimum size in bytes> <url> <description>
call :get "models\ggml-large-v3-turbo.bin" 1500000000 ^
  "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3-turbo.bin?download=true" ^
  "Speech recognition model - 1.6 GB"

call :get "models\ggml-silero-v5.1.2.bin" 800000 ^
  "https://huggingface.co/ggml-org/whisper-vad/resolve/main/ggml-silero-v5.1.2.bin?download=true" ^
  "Voice activity detector - 1 MB"

call :get "models\speaker-embedding.onnx" 90000000 ^
  "https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/nemo_en_titanet_large.onnx" ^
  "Speaker embedding model - 97 MB"

rem  Both language models ship: the app picks by VRAM at startup and the UI
rem  lets the user override, so either may be selected on any machine.
call :get "models\Qwen3.8-27B-UD-IQ3_XXS.gguf" 10000000000 ^
  "https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/resolve/main/Qwen3.8-27B-UD-IQ3_XXS.gguf?download=true" ^
  "Language model, Low Quality - 10.9 GB"

call :get "models\Qwen3.8-27B-UD-Q4_K_M.gguf" 16000000000 ^
  "https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/resolve/main/Qwen3.8-27B-UD-Q4_K_M.gguf?download=true" ^
  "Language model, High Quality - 16.5 GB, this is the long one"

rem  The speaker segmentation model is only published inside an archive.
if exist "models\segmentation-3.0.onnx" (
  echo   [skip] Speaker segmentation model - already present
) else (
  echo   [....] Speaker segmentation model ^(7 MB^)
  curl.exe -L --fail --retry 5 --retry-delay 5 -# ^
    -o "models\_segmentation.tar.bz2" ^
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2"
  if errorlevel 1 (
    echo   [FAIL] Speaker segmentation model
    set "FAILED=1"
  ) else (
    tar -xf "models\_segmentation.tar.bz2" -C "models"
    if exist "models\sherpa-onnx-pyannote-segmentation-3-0\model.onnx" (
      copy /Y "models\sherpa-onnx-pyannote-segmentation-3-0\model.onnx" ^
              "models\segmentation-3.0.onnx" >nul
      rmdir /S /Q "models\sherpa-onnx-pyannote-segmentation-3-0"
      del /Q "models\_segmentation.tar.bz2"
      echo   [ ok ] Speaker segmentation model
    ) else (
      echo   [FAIL] Speaker segmentation model - archive did not contain model.onnx
      set "FAILED=1"
    )
  )
)

echo.
echo   ---------------------------------------
if defined FAILED (
  echo   SOME DOWNLOADS FAILED. Re-run this script; finished files are skipped.
) else (
  echo   All models downloaded.
  echo.
  echo   You can now copy this entire folder to the offline machine
  echo   and start it with run.bat.
)
echo.
pause
exit /b 0

rem ---------------------------------------------------------------------------
:get
rem  %1 target  %2 minimum bytes  %3 url  %4 description
rem  NOTE: the description must not contain ( or ) -- it is echoed inside
rem  a parenthesised if-block, and cmd would treat them as block delimiters.
set "TARGET=%~1"
set "MINSIZE=%~2"
set "URL=%~3"
set "DESC=%~4"

if exist "%TARGET%" (
  for %%A in ("%TARGET%") do set "SIZE=%%~zA"
  if !SIZE! GEQ %MINSIZE% (
    echo   [skip] %DESC% - already present
    exit /b 0
  )
  echo   [redo] %DESC% - previous file was incomplete
  del /Q "%TARGET%"
)

echo   [....] %DESC%
rem  -C - resumes a partial download rather than starting over.
curl.exe -L --fail --retry 5 --retry-delay 5 -C - -# -o "%TARGET%" "%URL%"
if errorlevel 1 (
  echo   [FAIL] %DESC%
  set "FAILED=1"
  exit /b 1
)

for %%A in ("%TARGET%") do set "SIZE=%%~zA"
if !SIZE! LSS %MINSIZE% (
  echo   [FAIL] %DESC% - file is smaller than expected ^(!SIZE! bytes^)
  set "FAILED=1"
  exit /b 1
)
echo   [ ok ] %DESC%
exit /b 0
