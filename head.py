class Head:
    pan = 100
    tilt = 100
    index = -1
    time = 0
    state = 0# 0 - turned off, 1 - active, 2 - standby (no person, but still on)
    def setPos(self, pan_, tilt_):
        self.pan = pan_
        self.tilt = tilt_
