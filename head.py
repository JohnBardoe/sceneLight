class Head:
    def __init__(self):
        self.index = -1
        self.state = 0
        # 0 - turned off, 1 - active, 2 - standby (no person, but still on)
        self.time = 0
        self.pan = 100
        self.tilt = 100
