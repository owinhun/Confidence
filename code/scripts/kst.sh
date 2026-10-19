#!/bin/bash
# tzdata is absent on this machine, so TZ=Asia/Seoul is silently ignored and `date` returns
# UTC. Compute the offset explicitly instead.
python3 -c "import datetime;print((datetime.datetime.utcnow()+datetime.timedelta(hours=9)).strftime('%H:%M'))"
