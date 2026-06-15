# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Build instructions for the Raven hydrological modelling framework.

Builds the Raven C++ engine (CSHS-CWRA/RavenHydroFramework) from source via CMake
so it can be bundled in SYMFLUENCE's release binaries. RavenPy is pointed at the
resulting binary at runtime (see ``runner.py``), so a separate conda raven-hydro
install is not required.
"""
from __future__ import annotations

from symfluence.cli.services import get_common_build_environment, get_netcdf_detection
from symfluence.core.registries import R


@R.build_instructions.add('raven')
def get_raven_build_instructions():
    """Build configuration for Raven (CMake C++ build, NetCDF-enabled)."""
    common_env = get_common_build_environment()
    netcdf_detect = get_netcdf_detection()

    return {
        'description': 'Raven flexible hydrological modelling framework (U. Waterloo)',
        'config_path_key': 'RAVEN_INSTALL_PATH',
        'config_exe_key': 'RAVEN_EXE',
        'default_path_suffix': 'installs/raven/bin',
        'default_exe': 'raven',
        'repository': 'https://github.com/CSHS-CWRA/RavenHydroFramework.git',
        'branch': 'main',
        'install_dir': 'raven',
        'build_commands': [
            common_env,
            netcdf_detect,
            r'''
set -e
echo "=== Raven build starting ==="
# Build the Raven executable with CMake; enable NetCDF I/O when the libs are present.
NETCDF_FLAG="-DCOMPILE_WITH_NETCDF=ON"
if [ -z "${NETCDF_C:-}${NETCDF:-}" ]; then
  echo "NetCDF not detected — building without NetCDF support"
  NETCDF_FLAG="-DCOMPILE_WITH_NETCDF=OFF"
fi

mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release ${NETCDF_FLAG}
cmake --build . -j"$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 2)"

# Install the produced executable as install/bin/raven (lowercase, stable name).
cd ..
mkdir -p install/bin
RAVEN_BIN="$(find build -maxdepth 2 -type f \( -name 'Raven' -o -name 'Raven.exe' -o -name 'raven' \) -perm -u+x 2>/dev/null | head -1)"
if [ -z "$RAVEN_BIN" ]; then
  echo "ERROR: Raven executable not found after build" >&2
  exit 1
fi
cp "$RAVEN_BIN" install/bin/raven
chmod +x install/bin/raven
echo "=== Raven build complete: install/bin/raven ==="
'''
        ],
        'dependencies': [],
        'verify_install': {
            'file_paths': ['install/bin/raven', 'install/bin/raven.exe', 'bin/raven'],
            'check_type': 'exists_any',
        },
        'order': 5,
        'optional': True,
    }
