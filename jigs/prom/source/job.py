import time

import prometheus_client as prom

prom.start_http_server(9001)

# proc_collector = prom.ProcessCollector()
# prom.REGISTER.register(proc_collector)

c = prom.Counter("loop_counter", "Just something to test")

for i in range(1000000000000):

    c.inc()

    time.sleep(10)
