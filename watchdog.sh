#!/bin/sh
# Restarts gunicorn (via supervisorctl) if it stops responding to /health after
# having previously started up successfully. Cold starts can legitimately take
# a long time (downloading/loading several GB of models), so failures are only
# treated as a real problem - and trigger a restart - once the app has been
# healthy at least once before.
CHECK_INTERVAL=30
MAX_FAILURES=3
FAILURES=0
EVER_HEALTHY=0

while true; do
  sleep "$CHECK_INTERVAL"
  if curl -fsS --max-time 10 http://localhost:8080/health > /dev/null 2>&1; then
    FAILURES=0
    EVER_HEALTHY=1
  elif [ "$EVER_HEALTHY" -eq 1 ]; then
    FAILURES=$((FAILURES + 1))
    echo "watchdog: health check failed ($FAILURES/$MAX_FAILURES) after having been healthy"
    if [ "$FAILURES" -ge "$MAX_FAILURES" ]; then
      echo "watchdog: restarting gunicorn - unresponsive for $((FAILURES * CHECK_INTERVAL))s"
      supervisorctl restart gunicorn
      FAILURES=0
      EVER_HEALTHY=0
    fi
  else
    echo "watchdog: still starting up, health check not yet passing"
  fi
done
