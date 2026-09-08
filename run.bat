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
if not exist "bin\whisper\whisper-cli.exe"            set "MISSING=!MISSING! bin\whisper\whisper-cli.exe"
if not exist "bin\llama\llama-server.exe"             set "MISSING=!MISSING! bin\llama\llama-server.exe"
if not exist "models\ggml-large-v3-turbo.bin"         set "MISSING=!MISSING! models\ggml-large-v3-turbo.bin"
if not exist "models\ggml-silero-v5.1.2.bin"          set "MISSING=!MISSING! models\ggml-silero-v5.1.2.bin"
if not exist "models\segmentation-3.0.onnx"           set "MISSING=!MISSING! models\segmentation-3.0.onnx"
if not exist "models\speaker-embedding.onnx"          set "MISSING=!MISSING! models\speaker-embedding.onnx"
if not exist "models\Qwen3.8-27B-UD-IQ3_XXS.gguf"    set "MISSING=!MISSING! models\Qwen3.8-27B-UD-IQ3_XXS.gguf"
if not exist "models\Qwen3.8-27B-UD-Q4_K_M.gguf"     set "MISSING=!MISSING! models\Qwen3.8-27B-UD-Q4_K_M.gguf"

if not "!MISSING!"=="" (
  echo.
  echo   Meeting Summariser cannot start.
  echo.
  echo   These files are missing from the application folder:
  for %%F in (!MISSING!) do echo       %%F
  echo.
  echo   The folder may not have copied completely. Copy it again.
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
>>"%SHORTCUT%" echo URL=http://127.0.0.1:%PORT%

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
echo   Meeting Summariser has stopped.
pause
