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
echo   Total download: about 30 GB - 6 models plus the inference binaries
echo   for every graphics vendor. This will take a while.
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

rem ---------------------------------------------------------------------------
rem  Inference binaries, one folder per backend.
rem
rem  Every backend is downloaded, not just the one this machine needs: the
rem  folder is meant to be copied to a different computer, and picking the
rem  wrong set at download time would only be discovered over there. Together
rem  they add about 1.1 GB against 29 GB of models, which is not worth a
rem  choice the user could get wrong.
rem
rem  There is deliberately no whisper-vulkan below. whisper.cpp publishes no
rem  Vulkan binary for Windows -- it has to be built from source -- so on a
rem  machine with no NVIDIA card the app runs the language model on Vulkan and
rem  falls back to the CPU build for transcription. Drop a Vulkan build into
rem  bin\whisper-vulkan\ and it will be picked up automatically.
rem ---------------------------------------------------------------------------

set "LLAMA_BUILD=b10852"
set "WHISPER_BUILD=b4938"
set "GH=https://github.com/ggml-org"

call :getbin "llama-cuda"   90000000  ^
  "%GH%/llama.cpp/releases/download/%LLAMA_BUILD%/llama-%LLAMA_BUILD%-bin-win-cuda-12.4-x64.zip" ^
  "llama.cpp for NVIDIA - 254 MB"

call :getbin "llama-vulkan" 20000000  ^
  "%GH%/llama.cpp/releases/download/%LLAMA_BUILD%/llama-%LLAMA_BUILD%-bin-win-vulkan-x64.zip" ^
  "llama.cpp for AMD and Intel - 36 MB"

call :getbin "llama-cpu"    10000000  ^
  "%GH%/llama.cpp/releases/download/%LLAMA_BUILD%/llama-%LLAMA_BUILD%-bin-win-cpu-x64.zip" ^
  "llama.cpp without a GPU - 18 MB"

call :getbin "whisper-cuda" 300000000 ^
  "%GH%/whisper.cpp/releases/download/%WHISPER_BUILD%/whisper-cublas-12.4.0-bin-x64.zip" ^
  "whisper.cpp for NVIDIA - 671 MB"

call :getbin "whisper-cpu"  5000000   ^
  "%GH%/whisper.cpp/releases/download/%WHISPER_BUILD%/whisper-bin-x64.zip" ^
  "whisper.cpp without a GPU - 8 MB"

rem  The CUDA runtime is shared by both NVIDIA builds and sits in bin\ itself,
rem  where Windows finds it on PATH for either engine.
call :getbin "" 380000000 ^
  "%GH%/llama.cpp/releases/download/%LLAMA_BUILD%/cudart-llama-bin-win-cuda-12.4-x64.zip" ^
  "NVIDIA CUDA runtime - 391 MB"

echo.
echo   ---------------------------------------
if defined FAILED (
  echo   SOME DOWNLOADS FAILED. Re-run this script; finished files are skipped.
) else (
  echo   All models and binaries downloaded.
  echo.
  echo   You can now copy this entire folder to the offline machine
  echo   and start it with run.bat.
)
echo.
pause
exit /b 0

rem ---------------------------------------------------------------------------
:getbin
rem  %1 folder under bin\ - blank means bin\ itself   %2 minimum bytes
rem  %3 url   %4 description
set "SUB=%~1"
set "MINSIZE=%~2"
set "URL=%~3"
set "DESC=%~4"
if "%SUB%"=="" ( set "DEST=bin" ) else ( set "DEST=bin\%SUB%" )

if exist "%DEST%\llama-server.exe" goto :binskip
if exist "%DEST%\whisper-cli.exe"  goto :binskip
if "%SUB%"=="" if exist "bin\cudart64_12.dll" goto :binskip

echo   [....] %DESC%
if not exist "temp" mkdir "temp"
curl.exe -L --fail --retry 5 --retry-delay 5 -C - -# -o "temp\_bin.zip" "%URL%"
if errorlevel 1 (
  echo   [FAIL] %DESC%
  set "FAILED=1"
  exit /b 1
)
for %%A in ("temp\_bin.zip") do set "SIZE=%%~zA"
if !SIZE! LSS %MINSIZE% (
  echo   [FAIL] %DESC% - file is smaller than expected
  set "FAILED=1"
  del /Q "temp\_bin.zip"
  exit /b 1
)

if not exist "%DEST%" mkdir "%DEST%"
rem  bsdtar ships with Windows and reads zip. The absolute System32 path is
rem  deliberate: a `tar` on PATH may be GNU tar, which cannot.
"%SystemRoot%\System32\tar.exe" -xf "temp\_bin.zip" -C "%DEST%"
if errorlevel 1 (
  echo   [FAIL] %DESC% - could not unpack
  set "FAILED=1"
  del /Q "temp\_bin.zip"
  exit /b 1
)
del /Q "temp\_bin.zip"

rem  The whisper zips wrap everything in Release\; flatten it so every backend
rem  folder has the same shape.
if exist "%DEST%\Release\whisper-cli.exe" (
  move /Y "%DEST%\Release\*" "%DEST%\" >nul 2>&1
  rmdir /S /Q "%DEST%\Release" 2>nul
)
echo   [ ok ] %DESC%
exit /b 0

:binskip
echo   [skip] %DESC% - already present
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
