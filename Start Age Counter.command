#!/bin/zsh
cd -- "${0:A:h}" || exit 1
if [[ -x .venv/bin/live-camera ]]; then
    live_camera=.venv/bin/live-camera
else
    print 'Please follow the Python setup steps in README.md first.'
    read 'reply?Press Enter to close: '
    exit 1
fi
"$live_camera" --age "$@"
result=$?
if (( result != 0 )); then
    read 'reply?Press Enter to close: '
fi
exit $result
