@echo off
echo 🚀 Multi-Modal Stock Analysis Framework - Quick Test
echo ===============================================

REM Try different Python commands
echo Testing Python installation...

python --version >nul 2>&1
if %errorlevel% == 0 (
    echo ✅ Python found: python command
    set PYTHON_CMD=python
    goto :run_tests
)

py --version >nul 2>&1
if %errorlevel% == 0 (
    echo ✅ Python found: py command
    set PYTHON_CMD=py
    goto :run_tests
)

python3 --version >nul 2>&1
if %errorlevel% == 0 (
    echo ✅ Python found: python3 command
    set PYTHON_CMD=python3
    goto :run_tests
)

echo ❌ Python not found in PATH
echo Please install Python from https://python.org or add it to PATH
pause
exit /b 1

:run_tests
echo.
echo 📋 Running component tests...
%PYTHON_CMD% test_components.py

echo.
echo 📊 Checking system status...
%PYTHON_CMD% main.py --status

echo.
echo 🧪 Running basic analysis test...
echo This may take a few minutes...
%PYTHON_CMD% main.py --ticker AAPL

echo.
echo ✅ Testing completed!
echo Check the output above for any errors.
echo.
pause

