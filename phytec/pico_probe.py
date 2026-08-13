import os, termios, time

DEV = "/dev/ttyACM0"
BAUD = 115200

fd = os.open(DEV, os.O_RDWR | os.O_NOCTTY | os.O_NONBLOCK)
attrs = termios.tcgetattr(fd)
speed = getattr(termios, f"B{BAUD}")
attrs[4] = speed
attrs[5] = speed
attrs[2] = (attrs[2] & ~termios.CSIZE) | termios.CS8
attrs[2] &= ~termios.PARENB
attrs[2] &= ~termios.CSTOPB
attrs[2] |= (termios.CLOCAL | termios.CREAD)
attrs[0] = 0
attrs[1] = 0
attrs[3] = 0
# non-canonical: return immediately
attrs[6][termios.VMIN] = 0
attrs[6][termios.VTIME] = 0
termios.tcsetattr(fd, termios.TCSANOW, attrs)
termios.tcflush(fd, termios.TCIOFLUSH)

def drain(secs, label):
    buf = b""
    t0 = time.time()
    while time.time() - t0 < secs:
        try:
            c = os.read(fd, 256)
            if c:
                buf += c
        except BlockingIOError:
            pass
        time.sleep(0.02)
    print(f"[{label}] {len(buf)} bytes:")
    print(buf.decode(errors="replace"))
    print("-" * 40)

# 1) Ctrl-C to interrupt any running loop, then see what prints
os.write(fd, b"\x03")
drain(1.0, "after Ctrl-C")

# 2) Ctrl-D soft reboot: should print MicroPython banner + main.py boot output (or traceback)
os.write(fd, b"\x04")
drain(3.0, "after Ctrl-D soft reboot")

# 3) newline: if at REPL we'd see >>> ; if main.py loop is running, likely ERR or nothing
os.write(fd, b"\r\n")
drain(1.0, "after newline")

os.close(fd)
print("done")
