from machine import I2C, Pin
import time, sys, select

# --- Hardware setup ---
i2c = I2C(0, sda=Pin(4), scl=Pin(5), freq=100000)
PCA = 0x40

PAN = 0
TILT = 1

# Safe range limits (tilt UNLOADED - re-measure after camera mount)
PAN_MIN, PAN_MAX = 110, 570
TILT_MIN, TILT_MAX = 110, 570
CENTER = 307

# --- Slew tuning: smaller step / larger delay = slower, gentler motion ---
SLEW_STEP = 8        # pulse units per increment
SLEW_DELAY_MS = 18   # pause between increments

def pca_init():
    i2c.writeto_mem(PCA, 0x00, b'\x10')
    i2c.writeto_mem(PCA, 0xFE, bytes([121]))
    i2c.writeto_mem(PCA, 0x00, b'\x20')
    time.sleep_ms(5)

def set_pwm(ch, off):
    off = int(off)
    base = 0x06 + 4 * ch
    i2c.writeto_mem(PCA, base,   bytes([0]))
    i2c.writeto_mem(PCA, base+1, bytes([0]))
    i2c.writeto_mem(PCA, base+2, bytes([off & 0xFF]))
    i2c.writeto_mem(PCA, base+3, bytes([off >> 8]))

def clamp(val, lo, hi):
    return max(lo, min(hi, val))

# Map normalized [-1,1] to pulse, centered on CENTER, symmetric within limits
def norm_to_pulse(n, lo, hi):
    n = clamp(n, -1.0, 1.0)
    half = min(CENTER - lo, hi - CENTER)   # largest symmetric swing that fits
    return int(CENTER + n * half)

# current committed pulse per channel (start centered, matching boot)
cur = {PAN: CENTER, TILT: CENTER}

# Slew both channels together toward their targets in small steps
def slew_to(pan_target, tilt_target):
    pt = clamp(pan_target, PAN_MIN, PAN_MAX)
    tt = clamp(tilt_target, TILT_MIN, TILT_MAX)
    while cur[PAN] != pt or cur[TILT] != tt:
        for ch, tgt in ((PAN, pt), (TILT, tt)):
            d = tgt - cur[ch]
            if d != 0:
                step = SLEW_STEP if d > 0 else -SLEW_STEP
                if abs(d) < SLEW_STEP:
                    step = d
                cur[ch] += step
                set_pwm(ch, cur[ch])
        time.sleep_ms(SLEW_DELAY_MS)
    return cur[PAN], cur[TILT]

def aim(pan_n, tilt_n):
    return slew_to(norm_to_pulse(pan_n, PAN_MIN, PAN_MAX),
                   norm_to_pulse(tilt_n, TILT_MIN, TILT_MAX))

def center():
    return slew_to(CENTER, CENTER)

# --- Command handling ---
def handle(line):
    parts = line.strip().split()
    if not parts:
        return None
    cmd = parts[0].upper()
    if cmd == "PING":
        return "PONG"
    if cmd == "CENTER":
        p, t = center()
        return "OK %d %d" % (p, t)
    if cmd == "AIM" and len(parts) == 3:
        try:
            pan_n = float(parts[1]); tilt_n = float(parts[2])
        except ValueError:
            return "ERR"
        p, t = aim(pan_n, tilt_n)
        return "OK %d %d" % (p, t)
    return "ERR"

# --- Boot ---
pca_init()
# snap to center once at boot (cur already CENTER, so write directly)
set_pwm(PAN, CENTER)
set_pwm(TILT, CENTER)
print("READY pan:[%d,%d] tilt:[%d,%d]" % (PAN_MIN, PAN_MAX, TILT_MIN, TILT_MAX))

# --- Serial command loop ---
poll = select.poll()
poll.register(sys.stdin, select.POLLIN)
buf = ""
while True:
    if poll.poll(0):
        ch = sys.stdin.read(1)
        if ch == "\n" or ch == "\r":
            if buf:
                resp = handle(buf)
                if resp is not None:
                    print(resp)
                buf = ""
        else:
            buf += ch
    else:
        time.sleep_ms(5)
