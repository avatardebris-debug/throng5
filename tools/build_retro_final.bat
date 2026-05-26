@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
set PATH=C:\Users\avata\aicompete\retro_env\Scripts;%PATH%
set CMAKE_GENERATOR=Ninja

set PYTHON311=C:\Users\avata\AppData\Local\Programs\Python\Python311
set RETRO_VENV=C:\Users\avata\aicompete\retro_env

cd /d C:\Users\avata\aicompete\stable-retro-src

REM Clean stale cmake cache
if exist CMakeCache.txt del /f CMakeCache.txt
if exist CMakeFiles rmdir /s /q CMakeFiles

echo === Installing zlib via vcpkg or nuget ===
REM Try to find zlib - VS has it bundled
set ZLIB_INC=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\VS\include
set ZLIB_LIB=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\VS\lib\x64

echo === cmake configure ===
cmake . -G Ninja ^
  -DPython_EXECUTABLE=%RETRO_VENV%\Scripts\python.exe ^
  -DPython_ROOT_DIR=%PYTHON311% ^
  -DPython_FIND_STRATEGY=LOCATION ^
  -DPython_INCLUDE_DIR=%PYTHON311%\Include ^
  -DPython_LIBRARY=%PYTHON311%\libs\python311.lib ^
  -DPYEXT_SUFFIX=.cp311-win_amd64.pyd ^
  -DPYLIB_DIRECTORY=. ^
  -DBUILD_N64=OFF ^
  -DBUILD_Jaguar=OFF

if errorlevel 1 (
  echo CMake configure FAILED - see output above
  exit /b 1
)

echo === ninja build ===
ninja -j4 stable_retro

if errorlevel 1 (
  echo Ninja build FAILED
  exit /b 1
)

echo.
echo === Install Python package wrapper ===
%RETRO_VENV%\Scripts\pip.exe install gymnasium "pyglet>=1.3.2,<2" farama-notifications --quiet
%RETRO_VENV%\Scripts\python.exe setup.py install --skip-build

echo.
echo === Test import ===
%RETRO_VENV%\Scripts\python.exe -c "import stable_retro; print('stable_retro OK:', stable_retro.__version__)"
