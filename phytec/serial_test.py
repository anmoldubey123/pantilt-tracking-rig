import os, termios, time

DEV = "/dev/ttyACM0"
BAUD = 115200

class SerialPort:
    def __init__(self, dev, baud=BAUD):
        self.fd = os.open(dev, os.O_RDWR | os.O_NOCTTY)
        attrs = termios.tcgetattr(self.fd)
        speed = getattr(termios, f"B{baud}")
        attrs[4] = speed
        attrs[5] = speed
        attrs[2] = (attrs[2] & ~termios.CSIZE) | termios.CS8
        attrs[2] &= ~termios.PARENB
        attrs[2] &= ~termios.CSTOPB
        attrs[2] |= (termios.CLOCAL | termios.CREAD)
        attrs[0] = 0
        attrs[1] = 0
        attrs[3] = 0
        termios.tcsetattr(self.fd, termios.TCSANOW, attrs)
        termios.tcflush(self.fd, termios.TCIOFLUSH)
    def write(self, data):
        os.write(self.fd, data)
    def read_reply(self, timeout=1.0):
        buf = b""
        t0 = time.time()
        while time.time() - t0 < timeout:
            try:
                chunk = os.read(self.fd, 128)
                if chunk:
                    buf += chunk
                    if b"\n" in buf:
                        break
            except BlockingIOError:
                pass
            time.sleep(0.01)
        return buf.decode(errors="replace").strip()
    def handshake(self, timeout=4.0):
        # soft-reboot (Ctrl-D) to recover a REPL-dropped Pico, wait for READY
        os.write(self.fd, b"\x04")
        buf = ""
        t0 = time.time()
        while time.time() - t0 < timeout:
            r = self.read_reply(0.3)
            if r:
                buf += r
                if "READY" in buf:
                    print("handshake: Pico ready")
                    return True
        # fallback: PING
        os.write(self.fd, b"PING\n")
        return "PONG" in (self.read_reply(1.0) or "")

    def close(self):
        os.close(self.fd)

ser = SerialPort(DEV)
time.sleep(0.2)
ser.handshake()

print("PING ->", repr(ser.read_reply(0.2)))  # drain
ser.write(b"PING\n")
print("PING reply:", repr(ser.read_reply()))

ser.write(b"CENTER\n")
print("CENTER reply:", repr(ser.read_reply()))
time.sleep(1.0)

for pan, tilt in [(0.6, 0.6), (-0.6, -0.6), (0.0, 0.0)]:
    ser.write(f"AIM {pan:.3f} {tilt:.3f}\n".encode())
    print(f"AIM {pan} {tilt} reply:", repr(ser.read_reply()))
    time.sleep(1.5)

ser.close()
print("done")
