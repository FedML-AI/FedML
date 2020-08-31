#! /bin/sh
# start or stop or restart
opt=$1

PROGRAM_NAME='FebMLTraining'

start() {
  program_num=$(ps -ef | grep $PROGRAM_NAME | grep -cv grep)
  if [ "$program_num" -eq 0 ]; then
    echo "start $PROGRAM_NAME"
    today=$(date +%Y%m%d%H)
    nohup gunicorn -n $PROGRAM_NAME -c conf/gunicornconf.py app:app>>console_$((today)).log app:client>>console_$((today)).log 2>&1 &
  else
    echo "$PROGRAM_NAME is running..."
    ps -ef | grep $PROGRAM_NAME | grep -v grep
  fi
}

stop() {
  program_num=$(ps -ef | grep $PROGRAM_NAME | grep -cv grep)
  if [ "$program_num" -eq 0 ]; then
    echo "$PROGRAM_NAME is not running"
    return
  fi
  # find all process id
  program_ids=$(ps -ef | grep $PROGRAM_NAME | grep -v grep | awk '{print $2}')
  for pid in $program_ids; do
    kill -9 "$pid"
  done
  echo "Stopped $PROGRAM_NAME"
}

status() {
  program_num=$(ps -ef | grep $PROGRAM_NAME | grep -cv grep)
  if [ "$program_num" -eq 0 ]; then
    echo "$PROGRAM_NAME is not running"
  else
    echo "$PROGRAM_NAME is running ..."
  fi
}

if [ x"${opt}" = x ]; then
  opt=start
fi

case $1 in
"start")
  start
  status
  ;;
"stop")
  stop
  ;;
"status")
  status
  ;;
"restart")
  stop
  start
  ;;
*)
  echo "please input: start, stop, status, restart"
  ;;
esac
