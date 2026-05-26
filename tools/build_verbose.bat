@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
set PATH=C:\Users\avata\aicompete\retro_env\Scripts;%PATH%
set CMAKE_GENERATOR=Ninja
cd /d C:\Users\avata\aicompete\stable-retro-src
echo.
echo === Python version ===
C:\Users\avata\aicompete\retro_env\Scripts\python.exe --version
echo.
echo === cmake version ===
cmake --version
echo.
echo === ninja version ===
ninja --version
echo.
echo === cl.exe check ===
where cl
echo.
echo === Attempting editable install (verbose) ===
C:\Users\avata\aicompete\retro_env\Scripts\pip.exe install -e . --no-build-isolation -v 2>&1
