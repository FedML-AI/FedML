#!/bin/bash
# Launch supervisor
BASENAME="${0##*/}"
log () {
  echo "${BASENAME} - ${1}"
}
EXIT_CODE_FILE="/tmp/batch-exit-code"
supervisord -n -c "/etc/supervisor/supervisord.conf"

log "Reading exit code from batch script stored at $EXIT_CODE_FILE"
if [ ! -f $EXIT_CODE_FILE ]; then
    echo "Exit code file not found , returning with exit code 1!" >&2
    exit 1
fi
exit $(cat $EXIT_CODE_FILE)
