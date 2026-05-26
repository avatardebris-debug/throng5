@echo off
REM Build stable-retro for Python 3.11 on Windows with VS2022 Build Tools
REM Run this from a VS 2022 x64 Developer Command Prompt

call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"

REM Add cmake/ninja from the venv to PATH
set PATH=C:\Users\avata\aicompete\retro_env\Scripts;%PATH%
set CMAKE_GENERATOR=Ninja

REM Install build deps
C:\Users\avata\aicompete\retro_env\Scripts\pip.exe install --upgrade pip setuptools wheel scikit-build

REM Clone and build from source
cd /d C:\Users\avata\aicompete
if not exist stable-retro-src (
    git clone --recurse-submodules https://github.com/Farama-Foundation/stable-retro.git stable-retro-src
)
cd stable-retro-src

REM Install with no isolation so cmake/ninja on PATH are used
C:\Users\avata\aicompete\retro_env\Scripts\pip.exe install -e . --no-build-isolation

echo.
echo Done! Test with:
echo C:\Users\avata\aicompete\retro_env\Scripts\python.exe -c "import retro; print('OK')"
