@echo off
setlocal
where python >nul 2>nul
if %ERRORLEVEL%==0 (
  python "%~dp0start_resa.py" %*
) else (
  py "%~dp0start_resa.py" %*
)
