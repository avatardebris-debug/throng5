@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
set PATH=C:\Users\avata\aicompete\retro_env\Scripts;%PATH%
set CMAKE_GENERATOR=Ninja
cd /d C:\Users\avata\aicompete\stable-retro-src

echo === Running setup.py build directly ===
C:\Users\avata\aicompete\retro_env\Scripts\python.exe setup.py build_ext --inplace 2>&1
