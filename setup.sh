#!/usr/bin/env bash
# Source this file from the ANNIE Python 3.9 environment:
#   source setup.sh

if [ -n "${BASH_VERSION:-}" ] && [ "${BASH_SOURCE[0]}" = "$0" ]; then
    echo "Run this script with: source setup.sh" >&2
    exit 1
fi

_pionid_setup_source="${BASH_SOURCE[0]:-$0}"
_pionid_setup_dir="$(cd "$(dirname "${_pionid_setup_source}")" && pwd)"
_pionid_shared_site="/exp/annie/app/users/dajana/myboy/lib/python3.9/site-packages"

if [ ! -d "${_pionid_shared_site}" ]; then
    echo "PionID setup error: shared site-packages directory not found:" >&2
    echo "  ${_pionid_shared_site}" >&2
    return 1 2>/dev/null || exit 1
fi

_pionid_python_version="$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
if [ "${_pionid_python_version}" != "3.9" ]; then
    echo "PionID setup error: the shared libraries are built for Python 3.9," >&2
    echo "but python3 is Python ${_pionid_python_version}." >&2
    return 1 2>/dev/null || exit 1
fi

# The ANNIE shared directory contains one tested TensorFlow 2.20 scientific
# Python stack. Put the complete stack first so TensorFlow, NumPy, Awkward,
# Uproot, and their compiled extensions always come from the same environment.
case "${PYTHONPATH:-}" in
    "${_pionid_shared_site}"|"${_pionid_shared_site}:"*) ;;
    *) export PYTHONPATH="${_pionid_shared_site}${PYTHONPATH:+:${PYTHONPATH}}" ;;
esac

python3 "${_pionid_setup_dir}/verify_python_environment.py"
_pionid_verify_status=$?
if [ "${_pionid_verify_status}" -ne 0 ]; then
    echo "PionID setup failed its compatibility check." >&2
    return "${_pionid_verify_status}" 2>/dev/null || exit "${_pionid_verify_status}"
fi

echo "PionID Python environment configured."

unset _pionid_setup_source
unset _pionid_setup_dir
unset _pionid_shared_site
unset _pionid_python_version
unset _pionid_verify_status
