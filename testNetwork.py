import sacn
import time

sender = sacn.sACNsender()
sender.start()
sender.activate_output(1)
sender[1].multicast = False
sender[1].destination = "10.8.220.23"
sender[1].dmx_data = (0,)*12 + (
    175,
    183,
    255,
    0,
    0,
    0,
    200
)



time.sleep(10)
sender.stop()
#10.8.220.40 - camera
#10.8.220.41 - camera controller

