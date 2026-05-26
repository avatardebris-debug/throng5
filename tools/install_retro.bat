@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"

REM Add cmake/ninja from the retro venv to PATH
set PATH=C:\Users\avata\aicompete\retro_env\Scripts;%PATH%

REM Override CMake generator from "Unix Makefiles" to "Ninja"
set CMAKE_GENERATOR=Ninja

C:\Users\avata\aicompete\retro_env\Scripts\pip.exe install stable-retro pygame numpy --no-build-isolation
