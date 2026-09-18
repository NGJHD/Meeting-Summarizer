@echo off
setlocal EnableDelayedExpansion
cd /d "%~dp0"

set "PYTHONDONTWRITEBYTECODE=1"
set "PYTHONUNBUFFERED=1"
set "PATH=%~dp0bin;%PATH%"

if not exist "temp"   mkdir "temp"
if not exist "output" mkdir "output"

rem ---- preflight: name the missing file plainly, do not vanish -------------

set "MISSING="
if not exist "runtime\python.exe"                     set "MISSING=!MISSING! runtime\python.exe"
if not exist "bin\ffmpeg.exe"                         set "MISSING=!MISSING! bin\ffmpeg.exe"
rem  One engine folder per backend. Only *some* must be present: an NVIDIA
rem  machine needs the cuda pair, an AMD or Intel one the vulkan/cpu pair.
rem  The server picks per engine at startup; here we only insist that each
rem  engine has at least one build to run.
set "HAVE_LLAMA="
for %%B in (cuda vulkan cpu) do if exist "bin\llama-%%B\llama-server.exe"  set "HAVE_LLAMA=1"
if exist "bin\llama\llama-server.exe" set "HAVE_LLAMA=1"
if not defined HAVE_LLAMA set "MISSING=!MISSING! bin\llama-*\llama-server.exe"

set "HAVE_WHISPER="
for %%B in (cuda vulkan cpu) do if exist "bin\whisper-%%B\whisper-cli.exe" set "HAVE_WHISPER=1"
if exist "bin\whisper\whisper-cli.exe" set "HAVE_WHISPER=1"
if not defined HAVE_WHISPER set "MISSING=!MISSING! bin\whisper-*\whisper-cli.exe"
if not exist "models\ggml-large-v3-turbo.bin"         set "MISSING=!MISSING! models\ggml-large-v3-turbo.bin"
if not exist "models\ggml-silero-v5.1.2.bin"          set "MISSING=!MISSING! models\ggml-silero-v5.1.2.bin"
if not exist "models\segmentation-3.0.onnx"           set "MISSING=!MISSING! models\segmentation-3.0.onnx"
if not exist "models\speaker-embedding.onnx"          set "MISSING=!MISSING! models\speaker-embedding.onnx"
rem The language model weights are 25GB of the download, and they are not
rem needed at all when the Model dropdown is set to "Port" -- that sends the
rem work to a server the user is already running. Ask config.json rather than
rem pattern-matching it here: "enabled": true appears under diarization too.
set "EXTERNAL_LLM="
"%~dp0runtime\python.exe" -c "import json,sys;sys.exit(0 if json.load(open('config.json')).get('llm',{}).get('external',{}).get('enabled') else 1)" 2>nul && set "EXTERNAL_LLM=1"
rem At least one language model, not every one. A copy updated from 1.0.x has
rem Q4_K_M and no UD-IQ4_XS -- the update payload carries no models\ -- and
rem naming a file the updater cannot deliver would stop it starting.
if not defined EXTERNAL_LLM (
  set "HAVE_LLM="
  if exist "models\Qwen3.8-27B-UD-IQ4_XS.gguf"  set "HAVE_LLM=1"
  if exist "models\Qwen3.8-27B-UD-IQ3_XXS.gguf" set "HAVE_LLM=1"
  if exist "models\Qwen3.8-27B-UD-Q4_K_M.gguf"  set "HAVE_LLM=1"
  if not defined HAVE_LLM set "MISSING=!MISSING! models\Qwen3.8-27B-UD-IQ4_XS.gguf"
)

if not "!MISSING!"=="" (
  echo.
  echo   Meeting Summariser cannot start.
  echo.
  echo   These files are missing from the application folder:
  for %%F in (!MISSING!) do echo       %%F
  echo.
  echo   If this is a fresh copy, run DOWNLOAD_MODELS.bat first - it fetches
  echo   the Python runtime, ffmpeg, the models and the inference binaries.
  echo   Otherwise the folder may not have copied completely; copy it again.
  echo.
  pause
  exit /b 1
)

rem ---- find a free port ---------------------------------------------------

set "PORT="
for %%P in (8000 8001 8002 8003 8004 8005) do (
  if not defined PORT (
    netstat -ano -p tcp | findstr /r /c:"LISTENING" | findstr /c:":%%P " >nul 2>&1
    if errorlevel 1 set "PORT=%%P"
  )
)

if not defined PORT (
  echo.
  echo   Meeting Summariser cannot start: ports 8000-8005 are all in use.
  echo   Restart the computer and try again.
  echo.
  pause
  exit /b 1
)

echo.
echo   Meeting Summariser
echo.
echo   Keep this window open while you work.
echo   Closing it stops the application.
echo.
rem A shortcut beside run.bat is the only reliable way to hand a
rem non-technical user a clickable link: whether a console linkifies a URL
rem depends on which terminal Windows happens to be using, and on this machine
rem it did not. Rewritten each launch so it always points at the live port.
set "SHORTCUT=%~dp0Open Meeting Summariser.url"
> "%SHORTCUT%" echo [InternetShortcut]
rem The version is in the URL deliberately: a browser that cached the page
rem from an older build may never revalidate it, and would show that old page
rem after an update. A new query string is a new cache key.
for /f "delims=" %%V in ('"%~dp0runtime\python.exe" -c "from server.version import APP_VERSION;print(APP_VERSION)"') do set "APPVER=%%V"
>>"%SHORTCUT%" echo URL=http://127.0.0.1:%PORT%/?v=%APPVER%

echo   [1/3] Program files found.
echo   [2/3] Port %PORT% is free.
echo   [3/3] Starting server; the browser opens by itself when it is ready.
echo.
echo   If it does not open, either double-click
echo   "Open Meeting Summariser" in this folder, or type this
echo   address into your browser:
echo.
echo       http://127.0.0.1:%PORT%
echo.

rem Open the browser only once the server actually answers. Opening it first
rem races the server and leaves the page retrying against a dead socket, which
rem looks identical to a hang.
start "" /B "%~dp0runtime\python.exe" -m server.open_browser %PORT%

rem --timeout-keep-alive covers the long idle gaps between SSE events.
rem --limit-max-requests is left unset; uvicorn imposes no body size cap of
rem its own, so a 400MB upload streams straight through (BUILD_NOTES.md).
"%~dp0runtime\python.exe" -m uvicorn server.main:app ^
  --host 127.0.0.1 ^
  --port %PORT% ^
  --timeout-keep-alive 300 ^
  --log-level warning

echo.
rem An update stops the server on purpose and opens a new window. Pausing
rem here would leave the old console sitting behind it saying the app had
rem stopped, which reads as a crash. The marker says it was intentional.
if exist "temp\updating.flag" (
  del /q "temp\updating.flag" >nul 2>&1
  echo   Updating - this window will close.
  exit /b 0
)
echo   Meeting Summariser has stopped.
pause
