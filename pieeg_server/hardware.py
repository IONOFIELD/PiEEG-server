"""
Low-level hardware interface for PiEEG.

Manages one or two ADS1299 ADC chips via SPI (8 channels each)
and GPIO lines for chip-select and data-ready signaling.

Requires: spidev (pip), Linux GPIO chardev (kernel, no pip package needed)
Must run on Raspberry Pi with SPI enabled and PiEEG shield connected.
"""

import logging
import os
import select
import struct
import sys
import time

try:
    import fcntl
except ImportError:
    fcntl = None  # not on Linux — hardware methods will fail, mock mode still works

from . import _native, drdy_reader
from .profiles import HardwareProfile, get_profile

logger = logging.getLogger("pieeg.hardware")

try:
    import spidev
except ImportError:
    spidev = None


def _require_hardware_libs():
    """Check that spidev is available, exit with a clear message if not."""
    if spidev is None:
        print(
            "\n  ERROR: Missing hardware library: spidev\n\n"
            "  This is a Raspberry Pi-only package.\n"
            "  Install it inside the project venv:\n"
            "    cd PiEEG-server && ./setup.sh\n\n"
            "  Or for testing without hardware:\n"
            "    pieeg-server --mock\n",
            file=sys.stderr,
        )
        sys.exit(1)

# --- ADC register addresses ---
WHO_I_AM = 0x00
CONFIG1 = 0x01
CONFIG2 = 0x02
CONFIG3 = 0x03
CH1SET = 0x05
CH2SET = 0x06
CH3SET = 0x07
CH4SET = 0x08
CH5SET = 0x09
CH6SET = 0x0A
CH7SET = 0x0B
CH8SET = 0x0C

# --- Lead-off (electrode contact) registers, per ADS1299 datasheet ---
LOFF = 0x04         # comparator thresholds + lead-off current + AC/DC mode
LOFF_SENSP = 0x0F   # per-channel enable, positive input lead-off sense
LOFF_SENSN = 0x10   # per-channel enable, negative input lead-off sense
LOFF_STATP = 0x12   # read-only status: positive input off/floating (per bit)
LOFF_STATN = 0x13   # read-only status: negative input off/floating (per bit)
CONFIG4 = 0x17      # lead-off comparator power + single-shot

# LOFF register value: COMP_TH=000 (comparator trips at 95%/5% of supply),
# ILEAD_OFF=00 (6 nA lead-off current), FLEAD_OFF=00 (DC lead-off). DC lead-off
# is the simplest/most robust mode: a floating high-impedance electrode drifts
# to a rail and trips the comparator; a connected one stays mid-supply.
# AC lead-off (an injected current for an impedance MAGNITUDE in kΩ) is a
# harder follow-on — TODO, not implemented here.
LOFF_DC_95_5 = 0x00
LOFF_SENSE_ALL = 0xFF          # enable lead-off sensing on all 8 channels
CONFIG4_PD_LOFF_COMP = 0x02    # bit 1: power up the lead-off comparators

# --- Bias drive (driven right leg) ---
# BIAS_SENSP/N choose which inputs the bias amplifier averages into the
# common-mode estimate it inverts and drives back onto the body through the
# BIAS pin (the "BIO" electrode). With none selected the BIAS pin only sits at
# mid-supply, so common-mode pickup is not cancelled. TI SBAS499C 9.3.2.4.5.
BIAS_SENSP = 0x0D
BIAS_SENSN = 0x0E
# CONFIG3: PD_REFBUF=1, reserved 11, BIAS_MEAS=0, BIASREF_INT=1, PD_BIAS=1,
# BIAS_LOFF_SENS=0 -> 0xEC (TI's sample sequence and OpenBCI). The old 0xFF
# also set BIAS_MEAS and BIAS_LOFF_SENS, which are for bias diagnostics only.
CONFIG3_BIAS_DRIVE = 0xEC
CONFIG3_LEGACY = 0xFF
# Register values with the bias drive on / off (off = the original PiEEG
# block, which is also what impedance_cal.json was fitted with).
BIAS_DRIVE_ON = {CONFIG3: CONFIG3_BIAS_DRIVE, BIAS_SENSP: 0xFF, BIAS_SENSN: 0xFF}
BIAS_DRIVE_OFF = {CONFIG3: CONFIG3_LEGACY, BIAS_SENSP: 0x00, BIAS_SENSN: 0x00}


def bias_drive_wanted(num_channels: int) -> bool:
    """Bias drive on? PIEEG_BIAS_DRIVE=1/0 forces it; unset, it is on for
    the 8-channel board (bench-tested) and off for the daisy-chained boards,
    whose BIAS wiring between chips has not been checked."""
    env = os.environ.get("PIEEG_BIAS_DRIVE", "").strip()
    if env:
        return env not in ("0", "off", "no", "false")
    return num_channels == 8

# Oversampling (see decimate.py): the chip runs at OUTPUT_RATE x k and the
# acquisition loop decimates back to OUTPUT_RATE. CONFIG1 codes by k.
OUTPUT_RATE = 250
OVERSAMPLE_CONFIG1 = {1: 0x96, 2: 0x95, 4: 0x94}


def oversample_factor(num_channels: int) -> int:
    """k from PIEEG_OVERSAMPLE (1, 2 or 4). Unset: 4 on the 8-channel board
    (bench-tested 2026-09-23), 1 on the daisy-chained boards, which read
    in-thread and haven't been tested faster — and 1 whenever PIEEG_CONFIG1
    picks the rate by hand."""
    env = os.environ.get("PIEEG_OVERSAMPLE", "").strip()
    if env:
        k = int(env)
    elif os.environ.get("PIEEG_CONFIG1", "").strip() or num_channels != 8:
        k = 1
    else:
        k = 4
    if k not in OVERSAMPLE_CONFIG1:
        raise ValueError(f"PIEEG_OVERSAMPLE={env!r}: use 1, 2 or 4")
    if k > 1 and num_channels != 8:
        raise ValueError("PIEEG_OVERSAMPLE is for the 8-channel board only")
    return k


# STATUS word sync marker: every ADS1299 data frame begins with a 24-bit STATUS
# word whose top 4 bits are fixed 1100. The remaining bits carry LOFF_STATP[7:0]
# + LOFF_STATN[7:0] + GPIO[3:0], which now VARY once lead-off sensing is on — so
# frame-sync validation must key on the fixed nibble only, not the whole word.
STATUS_SYNC_MASK = 0xF0
STATUS_SYNC_VALUE = 0xC0

# --- ADC commands ---
CMD_WAKEUP = 0x02
CMD_RESET = 0x06
CMD_START = 0x08
CMD_STOP = 0x0A
CMD_RDATAC = 0x10
CMD_SDATAC = 0x11
CMD_RDATA = 0x12

# --- Conversion constants ---
# 24-bit ADC: 2^23 - 1 = 0x7FFFFF, 2^24 - 1 = 0xFFFFFF
SIGN_TEST = 0x7FFFFF
FULL_SCALE = 0xFFFFFF
FULL_SCALE_PLUS_1 = 16777215
NEGATIVE_OFFSET = 16777214
FULL_SCALE_23 = (1 << 23) - 1   # 8388607: signed 24-bit positive full scale
VREF_UV = 4.5e6  # 4.5V reference in microvolts
# PiEEG-16: don't start a chip 2 read this close to its next conversion
# (see PiEEGHardware._wait_drdy2); a 16-channel frame read takes ~0.2 ms.
DRDY2_MARGIN_NS = 300_000
# ...and when its newest sample was already read, wait at most this long for
# the next one before reading the same sample again.
DRDY2_WAIT_NS = 1_000_000

# CHnSET register (0x05..0x0C) bit layout, per ADS1299 datasheet (TI SBAS499):
#   bit 7    PDn     0 = channel powered on
#   bits 6:4 GAINn   000=x1 001=x2 010=x4 011=x6 100=x8 101=x12 110=x24
#   bit 3    SRB2    0 = SRB2 open
#   bits 2:0 MUXn    000 = normal electrode input
# Clinical EEG uses PGA gain x24. 0x60 = 0b0110_0000 -> PD=0, GAIN=110 (x24),
# SRB2=0, MUX=000.
GAIN_CODE_X24 = 0b110
CHNSET_GAIN_X24_NORMAL = 0x60
PGA_GAIN_X24 = 24            # numeric multiplier for the x24 gain code

# --- GPIO pins ---
CS_PIN = 19
DRDY_PIN = 26      # DRDY chip 1
DRDY_PIN_2 = 13    # DRDY chip 2

# --- SPI settings ---
SPI_SPEED_HZ = 4_000_000
# Register reads/writes (RREG/WREG) must run MUCH slower than the streaming
# clock. Each command byte needs ~4 tCLK (tCLK = 2.048 MHz internal osc) to be
# decoded before the next byte; at 4 MHz that window is violated, so WREG never
# latches and RREG returns 0x00. Measured on this board (Pi 4): register access
# is unreliable at >=2 MHz and rock-solid (40/40) at 500 kHz. Streaming
# (readbytes) has no per-byte command decode and stays at the full 4 MHz.
REGISTER_SPEED_HZ = 500_000
SPI_MODE = 0b01
SPI_BITS = 8
BYTES_PER_READ = 27  # 3 status + 8 channels * 3 bytes

# --- Expected status header from ADC chip 2 ---
EXPECTED_STATUS = (192, 0, 8)  # 0xC0, 0x00, 0x08

# --- Spike detection defaults (matches not_spike script) ---
SPIKE_THRESHOLD = 5000  # max allowed jump in raw 24-bit signed value
SPIKE_RESET_AFTER = 50  # re-sync baseline after this many consecutive rejections

# --- Linux GPIO chardev v1 ioctl constants ---
# See include/uapi/linux/gpio.h in the Linux kernel source.
# ioctl numbers: _IOWR(type=0xB4, nr, size) = (3<<30)|(size<<16)|(0xB4<<8)|nr
_GPIO_GET_LINEHANDLE  = 0xC16CB403   # _IOWR(0xB4, 0x03, 364) – gpiohandle_request
_GPIOHANDLE_GET_VALUES = 0xC040B408  # _IOWR(0xB4, 0x08, 64)  – gpiohandle_data
_GPIOHANDLE_SET_VALUES = 0xC040B409  # _IOWR(0xB4, 0x09, 64)  – gpiohandle_data
_GPIOHANDLE_REQUEST_INPUT  = 1 << 0
_GPIOHANDLE_REQUEST_OUTPUT = 1 << 1
_HANDLE_REQUEST_SIZE = 364  # sizeof(struct gpiohandle_request)
_HANDLE_DATA_SIZE    = 64   # sizeof(struct gpiohandle_data)

# Line EVENTS (edge interrupts): requested via drdy_reader, which the reader
# process also runs standalone.
_EVENT_DATA_SIZE = drdy_reader.EVENT_DATA_SIZE   # u64 timestamp + u32 id


def _status_sync_ok(raw: list[int]) -> bool:
    """True if the frame's STATUS word carries the fixed 1100 sync marker."""
    return (raw[0] & STATUS_SYNC_MASK) == STATUS_SYNC_VALUE


def config1_sample_rate(config1: int) -> int | None:
    """Output data rate (SPS) selected by an ADS1299 CONFIG1 register value.

    The low three bits DR[2:0] divide the 16 kSPS modulator rate by a power
    of two: 000 = 16000 ... 101 = 500, 110 = 250. 111 is reserved (None).
    """
    dr = config1 & 0x07
    return None if dr == 0x07 else 16000 >> dr


def leadoff_state(p_off: bool, n_off: bool = False) -> str:
    """Green/red electrode contact verdict from a channel's lead-off flags.

    Only the P (electrode) flag counts. On the PiEEG-8 every N input is tied
    to SRB1, and on the bench (2026-09-16) the N flags read "off" on every
    channel whether REF was connected or not, so they say nothing about
    contact. A missing REF or GND shows up in the signal instead; see
    classify_contact(). n_off is accepted for call compatibility and ignored.
    """
    return "red" if p_off else "green"


# A channel whose recent samples reach this share of full scale is "at the
# rail" for contact purposes (a floating input with GND present reads ~94-100%).
CONTACT_RAIL_FRACTION = 0.9
# A floating REF puts one shared, drifting signal under every channel. When
# the connected leads' signals are this share identical and at least this
# large (µV std, per-channel DC removed), REF is taken as off. Bench: REF out
# gave 1222 µV identical on all 8; REF in gave 0.2-0.4 µV.
REF_FLOAT_COMMON_UV = 500.0
REF_FLOAT_COMMON_SHARE = 0.9
# Mains hum is shared too: REF through 20 kΩ picked up 1.3 mV of 60 Hz on every
# channel and was called floating. Averaging over 100 ms (a whole number of
# 50 Hz and 60 Hz cycles) removes it before the shared signal is judged.
MAINS_AVERAGE_S = 0.1
# A REF that has only just come out drifts before it gets large: bench
# 2026-09-16, it read -13 mV shared at first, then -67 to -84 mV over 6 s
# (2.4-3.8 mV/s). A connected REF gave well under 0.1 mV/s.
REF_FLOAT_DRIFT_UV_S = 1500.0
# ... and sits far from zero with every lead at the same level.
REF_FLOAT_DC_UV = 30_000.0
REF_FLOAT_DC_SPREAD = 0.2
# GND (BIO) out on WALL power (saline bath, 2026-09-23): nothing holds the
# body near the amplifier's reference, so every lead carries 0.3-33 mV of
# mains (median 3.5-4.5 mV; 2-10 µV with BIO in), and the all-off lead-off
# signature only flashes every 10-15 s. mains_uv at or above this, after such
# a flash, keeps GND off (see acq_viewer.ContactTracker).
GND_OFF_MAINS_UV = 500.0


def classify_contact(status, railed, common_uv=0.0, drift_uv_s=0.0,
                     shared_dc_uv=0.0):
    """Lead, REF and GND (BIO) contact from lead-off flags plus signal rails.

    Signatures measured on the PiEEG-8 with the leads, REF and BIO wires
    joined and then pulled one at a time (2026-09-16):

      all connected -> every lead flags on, signals in range
      one lead off  -> that lead flags off and its signal rails
      REF off       -> connected leads flag on, but their signals rail or all
                       carry one large identical signal (a floating REF
                       drifts; it doesn't always reach the rail)
      GND off       -> every lead flags off, yet the signals stay in range
                       (the DC test currents return through GND)

    status: leadoff_status() list. railed: per-channel bools, True when that
    channel's recent signal is at the rail (see CONTACT_RAIL_FRACTION).
    common_uv: size of the slow signal shared by the connected leads (see
    contact_from_signal); REF_FLOAT_COMMON_UV or more means REF is floating.
    drift_uv_s: how fast that shared signal moves; REF_FLOAT_DRIFT_UV_S or
    more means REF is floating. shared_dc_uv: the DC level every connected
    lead shares (0 when they differ); REF_FLOAT_DC_UV or more away from zero
    means REF is floating.
    Returns {"leads": [...], "ref": v, "gnd": v}; leads are "green"/"red",
    ref and gnd are "green"/"red" or None when the wiring can't tell.
    """
    p_off = [bool(c.get("p_off")) for c in status]
    n = len(p_off)
    at_rail = [bool(railed[i]) if i < len(railed) else False for i in range(n)]
    leads = ["red" if off else "green" for off in p_off]
    on = [i for i in range(n) if not p_off[i]]
    if not on:
        in_range = sum(1 for i in range(n) if not at_rail[i])
        return {"leads": leads, "ref": None,
                "gnd": "red" if n and in_range * 2 > n else None}
    ref_off = (sum(1 for i in on if at_rail[i]) * 2 > len(on)
               or common_uv >= REF_FLOAT_COMMON_UV
               or drift_uv_s >= REF_FLOAT_DRIFT_UV_S
               or abs(shared_dc_uv) >= REF_FLOAT_DC_UV)
    return {"leads": leads, "ref": "red" if ref_off else "green", "gnd": "green"}


def contact_from_signal(status, block, full_scale_uv, fs=250):
    """classify_contact() from a recent DC-mode signal block.

    block: (N samples x channels) µV; 0.25-0.5 s is enough. fs: its sample
    rate. Works out which channels are at the rail, then, for the connected
    leads averaged over MAINS_AVERAGE_S (so mains hum doesn't count): how
    large and how fast-moving their shared signal is (counted only when it is
    REF_FLOAT_COMMON_SHARE of their own signal), and the DC level they all
    share (0 when their levels differ).
    """
    import numpy as np

    x = np.asarray(block, dtype=float)
    railed = (np.max(np.abs(x), axis=0)
              >= CONTACT_RAIL_FRACTION * full_scale_uv).tolist()
    on = [i for i, c in enumerate(status)
          if not c.get("p_off") and i < x.shape[1]]
    common = drift = shared_dc = 0.0
    mains = 0.0
    live = on if len(on) >= 2 else [i for i in range(x.shape[1])
                                     if not railed[i]]
    k = int(round(MAINS_AVERAGE_S * fs))
    if live and 1 < k < x.shape[0]:
        # fast part: the signal minus its MAINS_AVERAGE_S running mean
        w = x[:, live]
        c = np.vstack([np.zeros((1, w.shape[1])), np.cumsum(w, axis=0)])
        fast = w[k - 1:] - (c[k:] - c[:-k]) / k
        mains = float(np.median(fast.std(axis=0)))
    if len(on) >= 2:
        z = x[:, on]
        if 1 < k < z.shape[0]:
            c = np.vstack([np.zeros((1, z.shape[1])), np.cumsum(z, axis=0)])
            z = (c[k:] - c[:-k]) / k
        dc = z.mean(axis=0)
        y = z - dc
        own = float(y.std(axis=0).mean())
        shared = y.mean(axis=1)
        if own > 0 and float(shared.std()) / own >= REF_FLOAT_COMMON_SHARE:
            common = float(shared.std())
            if shared.size >= 3:
                drift = abs(float(np.polyfit(np.arange(shared.size) / fs,
                                             shared, 1)[0]))
        level = float(np.median(dc))
        if np.all(np.abs(dc - level) <= REF_FLOAT_DC_SPREAD * abs(level)):
            shared_dc = level
    verdict = classify_contact(status, railed, common, drift, shared_dc)
    verdict["mains_uv"] = mains     # median rms of the leads' fast part
    return verdict


def parse_leadoff_status(status_bytes, channel_offset: int = 0) -> list[dict]:
    """Decode one ADS1299 24-bit STATUS word into per-channel lead-off flags.

    STATUS layout (MSB first): 1100 + LOFF_STATP[7:0] + LOFF_STATN[7:0]
    + GPIO[3:0]. LOFF_STATP/N bit i (0-indexed) maps to channel i+1; a set bit
    means that electrode's positive (P) or negative (N) input is off/floating
    (impedance above the comparator threshold). ``channel_offset`` shifts the
    reported channel numbers, e.g. 8 for the second ADS1299 in 16-channel mode.

    Returns a list of 8 dicts: {"ch", "off", "p_off", "n_off", "state"} where
    "off" is the electrode (P) flag and "state" its green/red verdict from
    ``leadoff_state``. n_off is passed through raw but is not meaningful on
    the PiEEG-8 (see leadoff_state).
    """
    word = (status_bytes[0] << 16) | (status_bytes[1] << 8) | status_bytes[2]
    statp = (word >> 12) & 0xFF
    statn = (word >> 4) & 0xFF
    channels = []
    for i in range(8):
        p_off = bool(statp & (1 << i))
        n_off = bool(statn & (1 << i))
        channels.append({
            "ch": channel_offset + i + 1,
            "off": p_off,
            "p_off": p_off,
            "n_off": n_off,
            "state": leadoff_state(p_off),
        })
    return channels


class PiEEGHardware:
    """Hardware abstraction for PiEEG shields (8 or 16 channels)."""

    # Channel register addresses for convenience
    CH_REGS = (CH1SET, CH2SET, CH3SET, CH4SET, CH5SET, CH6SET, CH7SET, CH8SET)

    def __init__(self, gpio_chip: str = "/dev/gpiochip4",
                 num_channels: int = 16,
                 profile: HardwareProfile | str | None = None):
        if num_channels not in (8, 16):
            raise ValueError(f"num_channels must be 8 or 16, got {num_channels}")
        self._num_channels = num_channels
        self._gpio_chip_name = gpio_chip
        # Resolve profile: accept a HardwareProfile, a name string, or None.
        # None / "auto" trigger auto-detection; unknown setups fall back to
        # the safe default (= pre-existing behavior).
        if isinstance(profile, HardwareProfile):
            self._profile = profile
        else:
            self._profile = get_profile(profile)
        # 16-ch mode always needs GPIO19 toggling for chip 2's CS line,
        # regardless of the profile's manage_cs_pin setting.
        self._manage_cs = self._profile.manage_cs_pin or num_channels == 16
        if num_channels == 16 and not self._profile.manage_cs_pin:
            logger.warning(
                "Profile %r disables CS pin management, but 16-channel mode "
                "requires GPIO%d for chip 2. Forcing CS management on.",
                self._profile.name, CS_PIN,
            )
        self._chip_fd = -1
        self._cs_fd = -1
        self._drdy_fd = -1
        self._drdy2_fd = -1
        self._drdy2_event_fd = -1  # chip 2 falling-edge events (16-ch)
        self._drdy2_last_ns = 0    # newest chip 2 edge seen
        self._drdy2_read_ns = 0    # chip 2 edge whose sample was last read
        self._drdy2_filled = 0     # chip 2 edges the kernel dropped (filled in)
        self._drdy2_repeats = 0    # chip 2 samples read twice (phase wrap)
        self._drdy_event_fd = -1   # falling-edge interrupt fd (interrupt mode)
        self._spi1 = None
        self._spi2 = None
        self._last_valid_value: int | None = None
        self._spike_count = 0
        self._consecutive_rejects = 0
        self._spike_threshold = SPIKE_THRESHOLD
        self._spike_reset_after = SPIKE_RESET_AFTER
        self._register_state: dict[int, int] = {}
        # PGA gain verified by register readback in _configure_adc.
        self._pga_gain: int | None = None
        # CONFIG1 value written in _configure_adc; the sample rate derives from
        # it. None until open() has configured the chip.
        self._config1: int | None = None
        # Latest per-channel lead-off (electrode contact) state, parsed from the
        # STATUS word of the live data stream. Empty until the first read.
        self._leadoff: list[dict] = []

    @property
    def num_channels(self) -> int:
        return self._num_channels

    @property
    def pga_gain(self) -> int | None:
        """PGA gain confirmed by register readback (None until configured)."""
        return self._pga_gain

    @property
    def chip_rate(self) -> int | None:
        """Conversion rate (SPS) programmed into CONFIG1 (None until
        configured): the DRDY edge rate the acquisition loop times against."""
        config1 = getattr(self, "_config1", None)
        return None if config1 is None else config1_sample_rate(config1)

    @property
    def _period_ns(self) -> int:
        """Nanoseconds between conversions at the programmed chip rate."""
        return 1_000_000_000 // (self.chip_rate or 250)

    @property
    def oversample(self) -> int:
        """Chip samples per output sample (1 = no decimation)."""
        return getattr(self, "_oversample", 1)

    @property
    def config1(self) -> int | None:
        """The CONFIG1 value streaming runs with (restored after the
        impedance check, which needs 250 SPS)."""
        return getattr(self, "_config1", None)

    @property
    def sample_rate(self) -> int | None:
        """Rate (SPS) of the samples the server hands on (None until
        configured): the chip rate, or with oversampling the decimated rate.

        Consumers (server welcome, filters, journal, the Scope viewer) read
        this instead of assuming 250, so PIEEG_CONFIG1 changes stay consistent.
        """
        rate = self.chip_rate
        return None if rate is None else rate // self.oversample

    @property
    def spike_threshold(self) -> int:
        return self._spike_threshold

    @spike_threshold.setter
    def spike_threshold(self, value: int):
        v = int(value)
        self._spike_threshold = v if v == -1 else max(0, v)

    @property
    def spike_reset_after(self) -> int:
        return self._spike_reset_after

    @spike_reset_after.setter
    def spike_reset_after(self, value: int):
        self._spike_reset_after = max(1, int(value))

    # --- lifecycle ---

    def open(self):
        """Initialize GPIO and SPI, configure ADC chip(s)."""
        _require_hardware_libs()
        self._init_gpio()
        self._init_spi()
        self._configure_adc(chip_num=1)
        if self._num_channels == 16:
            self._configure_adc(chip_num=2)

    def close(self):
        """Release all hardware resources."""
        if self._spi1:
            self._spi1.close()
        if self._spi2 and self._num_channels == 16:
            self._spi2.close()
        if self._cs_fd >= 0:
            os.close(self._cs_fd)
            self._cs_fd = -1
        if self._drdy_fd >= 0:
            os.close(self._drdy_fd)
            self._drdy_fd = -1
        if self._drdy_event_fd >= 0:
            os.close(self._drdy_event_fd)
            self._drdy_event_fd = -1
        if self._drdy2_fd >= 0:
            os.close(self._drdy2_fd)
            self._drdy2_fd = -1
        if self._drdy2_event_fd >= 0:
            os.close(self._drdy2_event_fd)
            self._drdy2_event_fd = -1
        if self._chip_fd >= 0:
            os.close(self._chip_fd)
            self._chip_fd = -1

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *exc):
        self.close()

    # --- public API ---

    def read_sample(self):
        """
        Read one sample from the ADC (8 or 16 channels).

        Returns None on status header mismatch.
        Returns a list of floats (microvolts) on success.

        Caller (acquisition loop) is responsible for DRDY polling.
        """
        # Read 27 bytes from ADC chip 1
        raw1 = self._spi1.readbytes(BYTES_PER_READ)

        if self._num_channels == 16:
            # Wait for chip 2 DRDY falling edge before reading
            self._wait_drdy2()
            self._cs_set(0)
            raw2 = self._spi2.readbytes(BYTES_PER_READ)
            self._cs_set(1)

            # Update lead-off status from both STATUS words. Done BEFORE the
            # spike/validity gates below so a floating electrode (which reads as
            # huge noise and is spike-rejected) still reports its "off" state.
            self._update_leadoff(raw1, raw2)

            # Spike detection: check last channel of chip 2 (bytes 24-26)
            if not self._is_valid_frame(raw2):
                return None

            # Validate the frame sync marker on chip 2. Only the fixed 1100
            # nibble is checked — the rest of the STATUS word now carries live
            # lead-off/GPIO bits (see STATUS_SYNC_MASK) and legitimately varies.
            if not _status_sync_ok(raw2):
                return None

            channels = []
            channels.extend(self._decode_channels(raw1))
            channels.extend(self._decode_channels(raw2))
            return channels
        else:
            return self.decode_frame(raw1)

    def decode_frame(self, raw1):
        """Decode one 27-byte 8-channel frame into µV (None if rejected).

        Split out of read_sample() so the separate reader process
        (drdy_reader) can do the SPI read and hand the bytes back here.
        """
        # 8-channel mode: spike detection on chip 1
        self._update_leadoff(raw1)
        if not self._is_valid_frame(raw1):
            return None
        # Reject reads without the 1100 sync marker. A read that starts
        # after the chip has shifted the frame out (a late or duplicate
        # read) returns all zeros, and a misaligned one garbage; both
        # passed straight through before and showed up as huge spikes.
        if not _status_sync_ok(raw1):
            return None
        return self._decode_channels(raw1)

    def _update_leadoff(self, raw1: list[int], raw2: list[int] | None = None):
        """Refresh cached lead-off state from the STATUS word(s) of a frame.

        Only trusts a frame whose sync marker is intact; a desynced/corrupt
        frame leaves the previous good state in place rather than reporting
        garbage contact readings.
        """
        if not _status_sync_ok(raw1):
            return
        leadoff = parse_leadoff_status(raw1)
        if raw2 is not None:
            if not _status_sync_ok(raw2):
                return
            leadoff = leadoff + parse_leadoff_status(raw2, channel_offset=8)
        self._leadoff = leadoff

    def leadoff_status(self) -> list[dict] | None:
        """Latest per-channel electrode lead-off state, or None before any read.

        Each entry: {"ch": int, "off": bool, "p_off": bool, "n_off": bool}.
        ``off`` True means the electrode is floating / high-impedance (no good
        contact). Derived from the DC lead-off comparators via the data-stream
        STATUS word; updated continuously as samples are read.
        """
        if not self._leadoff:
            return None
        return [dict(c) for c in self._leadoff]

    def _is_valid_frame(self, raw: list[int]) -> bool:
        """Spike detection matching the original not_spike script.

        Checks the last 3 bytes (bytes 24-26) of the SPI read as a signed
        24-bit integer. If the jump from the previous valid value exceeds
        SPIKE_THRESHOLD, the frame is considered corrupted.

        A threshold of -1 disables spike rejection entirely.
        """
        if self._spike_threshold == -1:
            return True

        combined = (raw[24] << 16) | (raw[25] << 8) | raw[26]
        if raw[24] & 0x80:
            combined -= 1 << 24

        if self._last_valid_value is None:
            self._last_valid_value = combined
            return False  # first frame is always skipped, matching original

        if abs(combined - self._last_valid_value) > self._spike_threshold:
            self._spike_count += 1
            self._consecutive_rejects += 1
            if self._consecutive_rejects >= self._spike_reset_after:
                # Electrode contact likely changed — accept new baseline
                logger.info(
                    "Spike filter reset after %d consecutive rejects "
                    "(old=%d, new=%d)",
                    self._consecutive_rejects,
                    self._last_valid_value,
                    combined,
                )
                self._last_valid_value = combined
                self._consecutive_rejects = 0
                return True
            logger.debug("Spike detected (count: %d)", self._spike_count)
            return False

        self._last_valid_value = combined
        self._consecutive_rejects = 0
        return True

    def wait_for_drdy(self):
        """
        Block until DRDY goes high (pre-condition for a valid read).
        Returns True when DRDY is high, meaning data will be ready
        on the next low transition.
        """
        return self._drdy_get() == 1

    # --- register configuration ---

    def configure_registers(self, reg_map: dict[int, int], start: bool = True):
        """Write arbitrary registers with STOP → SDATAC → write → RDATAC → START.

        Applies to both chips in 16-ch mode, chip 1 only in 8-ch mode.
        Updates the shadow register state.
        Resets spike filter baseline so the first post-config frames aren't
        rejected due to the signal level changing (e.g. normal → shorted).
        start=False leaves conversions stopped, for a caller that wants to
        send START itself once it is ready to catch the first DRDY edge (the
        impedance check counts samples from START to know the phase of the
        test signal).
        """
        chips = [1, 2] if self._num_channels == 16 else [1]
        for chip in chips:
            self._send_command(chip, CMD_STOP)
            self._send_command(chip, CMD_SDATAC)
            for addr, value in reg_map.items():
                self._write_register(chip, int(addr), int(value) & 0xFF)
            self._send_command(chip, CMD_RDATAC)
            if start:
                self._send_command(chip, CMD_START)
        self._register_state.update(reg_map)
        # Reset spike filter — signal level changes after config, old baseline
        # would cause false rejections (e.g. normal→shorted: ±50µV → ±2µV)
        self._last_valid_value = None
        self._consecutive_rejects = 0
        logger.info("Registers configured: %s", {hex(k): hex(v) for k, v in reg_map.items()})

    def drain_drdy_events(self):
        """Throw away any DRDY edges already queued, so the next one read is
        the next one that happens."""
        while self._drdy_event_fd >= 0 and select.select(
                [self._drdy_event_fd], [], [], 0)[0]:
            os.read(self._drdy_event_fd, _EVENT_DATA_SIZE)

    def start_conversions(self, chip_num: int = 1):
        """START on its own, for a caller that configured with start=False."""
        self._send_command(chip_num, CMD_START)

    def read_raw_frame(self):
        """The 27 raw bytes of one 8-channel frame (decode with
        decode_frame). The caller waits for DRDY itself."""
        return self._spi1.readbytes(BYTES_PER_READ)

    def set_input_short(self):
        """Set all CHnSET registers to 0x01 (input shorted for noise test)."""
        reg_map = {reg: 0x01 for reg in self.CH_REGS}
        self.configure_registers(reg_map)

    def set_input_normal(self):
        """Set all CHnSET registers to 0x00 (normal electrode input)."""
        reg_map = {reg: 0x00 for reg in self.CH_REGS}
        self.configure_registers(reg_map)

    @property
    def register_state(self) -> dict[int, int]:
        """Return a copy of the shadow register state."""
        return dict(self._register_state)

    # --- GPIO helpers (direct Linux chardev ioctl) ---

    def _cs_set(self, value: int):
        """Set chip-select line: 1 = high (deselect), 0 = low (select).

        No-op when the active profile delegates CS to the kernel SPI driver
        (e.g. Pi 5 in 8-channel mode). In 16-channel mode this always toggles
        GPIO19 because chip 2's CS is wired there on the PiEEG shield.
        """
        if not self._manage_cs or self._cs_fd < 0:
            return
        buf = bytearray(_HANDLE_DATA_SIZE)
        buf[0] = value & 1
        fcntl.ioctl(self._cs_fd, _GPIOHANDLE_SET_VALUES, buf)

    def _drdy_get(self) -> int:
        """Read chip 1 data-ready line. Returns 1 when high."""
        buf = bytearray(_HANDLE_DATA_SIZE)
        fcntl.ioctl(self._drdy_fd, _GPIOHANDLE_GET_VALUES, buf)
        return buf[0]

    def _drdy2_get(self) -> int:
        """Read chip 2 data-ready line. Returns 1 when high."""
        buf = bytearray(_HANDLE_DATA_SIZE)
        fcntl.ioctl(self._drdy2_fd, _GPIOHANDLE_GET_VALUES, buf)
        return buf[0]

    def _wait_drdy2(self):
        """Wait until chip 2 holds a sample that is safe to read now.

        The two ADS1299s on the PiEEG-16 run on their own oscillators, so
        chip 2's conversions slide against chip 1's (measured ~0.4 ms/s, a
        full 4 ms period every ~10 s); software can't lock them. Their DRDY
        flags can't say "unread sample" either: the chips share SCLK, so
        reading chip 1 also knocks chip 2's DRDY high.

        With DRDY2 edge events on (interrupt mode) the kernel timestamps
        chip 2's conversions, so per chip 1 frame:

        * an unread chip 2 sample is read now, unless its successor is due
          within DRDY2_MARGIN_NS (a read could straddle that update); then
          it waits for the successor;
        * if chip 2's newest sample was already read, it waits for the next
          one only if that is due within DRDY2_WAIT_NS, else it reads the
          same sample again.

        So the phase wrap costs one repeated (or, if chip 2 runs fast, one
        skipped) chip 2 sample every ~10 s and at most DRDY2_WAIT_NS of
        extra latency, and channels 9-16 stay within one period of 1-8.
        The old wait always skipped the ready sample for the next one, up to
        a whole period; chip 1's next edge then landed mid-read and ~89% of
        16-ch frames were thrown away as torn.

        Without events (busy-poll loops) it keeps the old edge wait.
        """
        if self._drdy2_event_fd < 0:
            while self._drdy2_get() == 0:   # wait out previous low
                pass
            while self._drdy2_get() == 1:   # wait for HIGH→LOW
                pass
            return
        fd = self._drdy2_event_fd
        period = self._period_ns
        last = self._drdy2_last_ns
        while select.select([fd], [], [], 0)[0]:
            last = self._read_edge(fd)
        if not last:                        # first frame: no edge seen yet
            if select.select([fd], [], [], period / 1e9)[0]:
                last = self._read_edge(fd)
            self._drdy2_last_ns = self._drdy2_read_ns = last
            return
        # The kernel now and then drops a DRDY2 edge while both chips are
        # being read (~1 in 200, sometimes a few in a row). Chip 2's clock is
        # steady, so a missing edge is put back where it must have been.
        now = time.monotonic_ns()
        while now - last >= period:
            last += period
            self._drdy2_filled += 1
        to_next = last + period - now
        # "Unread" by more than half a period, so a real edge that turns up
        # just after its filled-in stand-in isn't read as a new sample.
        unread = last - self._drdy2_read_ns > period // 2
        if to_next < (DRDY2_MARGIN_NS if unread else DRDY2_WAIT_NS):
            if select.select([fd], [], [], to_next / 1e9 + 2e-4)[0]:
                last = self._read_edge(fd)
            else:
                last += period
                self._drdy2_filled += 1
        elif not unread:
            self._drdy2_repeats += 1
        self._drdy2_last_ns = self._drdy2_read_ns = last

    # --- private helpers ---

    def _init_gpio(self):
        """Initialize GPIO via Linux chardev v1 ioctl (no gpiod dependency)."""
        logger.info("Opening GPIO chip %s", self._gpio_chip_name)
        self._chip_fd = os.open(self._gpio_chip_name, os.O_RDWR | os.O_CLOEXEC)

        # Chip-select line (output, default high). Skipped when the kernel
        # SPI driver owns the line (e.g. Pi 5 in 8-ch mode); see profiles.py.
        if self._manage_cs:
            self._cs_fd = self._request_line(
                self._chip_fd, CS_PIN, _GPIOHANDLE_REQUEST_OUTPUT,
                default_value=1, consumer=b"pieeg_cs")
        else:
            self._cs_fd = -1

        # Data-ready line chip 1 (input)
        self._drdy_fd = self._request_line(
            self._chip_fd, DRDY_PIN, _GPIOHANDLE_REQUEST_INPUT,
            consumer=b"pieeg_drdy")

        # Data-ready line chip 2 (input) — only for 16-ch mode
        if self._num_channels == 16:
            self._drdy2_fd = self._request_line(
                self._chip_fd, DRDY_PIN_2, _GPIOHANDLE_REQUEST_INPUT,
                consumer=b"pieeg_drdy2")

    @staticmethod
    def _request_line(chip_fd: int, pin: int, flags: int,
                      default_value: int = 0, consumer: bytes = b"pieeg") -> int:
        """Request a single GPIO line via GPIO_GET_LINEHANDLE_IOCTL.

        Returns the file descriptor for the requested line handle.
        """
        # struct gpiohandle_request layout (364 bytes):
        #   0..255   lineoffsets[64]   (u32 × 64)
        #   256..259 flags             (u32)
        #   260..323 default_values[64](u8 × 64)
        #   324..355 consumer_label[32](char × 32)
        #   356..359 lines             (u32)
        #   360..363 fd                (i32)
        buf = bytearray(_HANDLE_REQUEST_SIZE)
        struct.pack_into("I", buf, 0, pin)          # lineoffsets[0]
        struct.pack_into("I", buf, 256, flags)       # flags
        buf[260] = default_value & 1                  # default_values[0]
        label = consumer[:32]
        buf[324:324 + len(label)] = label             # consumer_label
        struct.pack_into("I", buf, 356, 1)           # lines = 1
        fcntl.ioctl(chip_fd, _GPIO_GET_LINEHANDLE, buf)
        return struct.unpack_from("i", buf, 360)[0]  # fd

    # --- DRDY interrupt (edge event) support ---

    @staticmethod
    def _request_event_line(chip_fd: int, pin: int,
                            consumer: bytes = b"pieeg_evt") -> int:
        """Request a GPIO line as a FALLING-EDGE event source.

        Returns a file descriptor that becomes readable on each edge; reading
        16 bytes yields one struct gpioevent_data (u64 kernel timestamp, u32 id).
        """
        return drdy_reader.request_falling_edge_events(chip_fd, pin, consumer)

    def reader_handles(self):
        """(gpiochip fd, DRDY pin, spidev fd) for the drdy_reader process.

        None when it can't be used: 16-channel boards read a second chip
        with its own DRDY and chip-select sequence, which stays in-thread.
        """
        if self._num_channels != 8 or self._spi1 is None or self._chip_fd < 0:
            return None
        return self._chip_fd, DRDY_PIN, self._spi1.fileno()

    def release_drdy_level(self):
        """Free the DRDY level handle so the reader process can request the
        line as an event source. disable_drdy_events() restores it."""
        if self._drdy_fd >= 0:
            os.close(self._drdy_fd)
            self._drdy_fd = -1

    def enable_drdy_events(self):
        """Switch chip-1 DRDY to interrupt mode (falling-edge events).

        A GPIO line can't be held as both a value-handle and an event source,
        so we release the level handle first, then request the event fd.
        """
        if self._drdy_fd >= 0:
            os.close(self._drdy_fd)
            self._drdy_fd = -1
        self._drdy_event_fd = self._request_event_line(
            self._chip_fd, DRDY_PIN, consumer=b"pieeg_drdy_evt")
        logger.info("DRDY interrupt mode enabled (falling-edge on GPIO%d)", DRDY_PIN)
        if self._num_channels == 16:
            if self._drdy2_fd >= 0:
                os.close(self._drdy2_fd)
                self._drdy2_fd = -1
            self._drdy2_event_fd = self._request_event_line(
                self._chip_fd, DRDY_PIN_2, consumer=b"pieeg_drdy2_evt")
            self._drdy2_last_ns = self._drdy2_read_ns = 0

    def disable_drdy_events(self):
        """Release the DRDY event fd and restore the level-read handle."""
        if self._drdy_event_fd >= 0:
            os.close(self._drdy_event_fd)
            self._drdy_event_fd = -1
        if self._drdy_fd < 0:
            self._drdy_fd = self._request_line(
                self._chip_fd, DRDY_PIN, _GPIOHANDLE_REQUEST_INPUT,
                consumer=b"pieeg_drdy")
        if self._drdy2_event_fd >= 0:
            os.close(self._drdy2_event_fd)
            self._drdy2_event_fd = -1
        if self._num_channels == 16 and self._drdy2_fd < 0:
            self._drdy2_fd = self._request_line(
                self._chip_fd, DRDY_PIN_2, _GPIOHANDLE_REQUEST_INPUT,
                consumer=b"pieeg_drdy2")

    def wait_drdy_event(self, timeout: float = 0.5):
        """Block until the next DRDY falling edge.

        Returns the kernel event timestamp in nanoseconds (CLOCK_MONOTONIC),
        or None if no edge arrived within ``timeout`` seconds (so the caller
        can re-check its stop flag).
        """
        ready, _, _ = select.select([self._drdy_event_fd], [], [], timeout)
        if not ready:
            return None
        return self._read_edge(self._drdy_event_fd)

    @staticmethod
    def _read_edge(fd: int) -> int:
        """One queued edge event's kernel timestamp (ns, CLOCK_MONOTONIC)."""
        data = os.read(fd, _EVENT_DATA_SIZE)
        # First 8 bytes = u64 timestamp (nanoseconds).
        return struct.unpack_from("Q", data, 0)[0]

    def stop_streaming(self, chip_num: int = 1):
        """Halt continuous conversion and return to a safe idle state.

        STOP ends conversions; SDATAC leaves the device ready for register
        access again (RREG/WREG are ignored while streaming).
        """
        self._send_command(chip_num, CMD_STOP)
        self._send_command(chip_num, CMD_SDATAC)

    def _init_spi(self):
        speed = self._profile.spi_speed_hz
        logger.info("Initializing SPI at %d Hz (profile=%s)",
                    speed, self._profile.name)
        self._spi1 = spidev.SpiDev()
        self._spi1.open(0, 0)
        self._spi1.max_speed_hz = speed
        self._spi1.lsbfirst = False
        self._spi1.mode = SPI_MODE
        self._spi1.bits_per_word = SPI_BITS

        if self._num_channels == 16:
            self._spi2 = spidev.SpiDev()
            self._spi2.open(0, 1)
            self._spi2.max_speed_hz = speed
            self._spi2.lsbfirst = False
            self._spi2.mode = SPI_MODE
            self._spi2.bits_per_word = SPI_BITS

    def _send_command(self, chip_num: int, command: int):
        # xfer2 keeps CS asserted for the whole transaction; commands run at the
        # slow register clock so the chip reliably decodes them.
        if chip_num == 1:
            self._spi1.xfer2([command], REGISTER_SPEED_HZ)
        else:
            self._cs_set(0)
            self._spi2.xfer2([command], REGISTER_SPEED_HZ)
            self._cs_set(1)

    def _write_register(self, chip_num: int, register: int, value: int):
        # WREG = [0x40|addr, num-1, value]. MUST use xfer2 (CS held low across
        # all three bytes) at the slow register clock, or the write won't latch.
        data = [0x40 | register, 0x00, value]
        if chip_num == 1:
            self._spi1.xfer2(data, REGISTER_SPEED_HZ)
        else:
            self._cs_set(0)
            self._spi2.xfer2(data, REGISTER_SPEED_HZ)
            self._cs_set(1)

    def _verify_comms(self, chip_num: int):
        """Confirm SPI register access works before trusting any config.

        The ID register (WHO_I_AM) reads a fixed ADS1299 value: bit4=1 and the
        low 5 bits = 0b1_1110 (0x1E), so 0x1E/0x3E/... are valid. Right after
        power-up/RESET the chip occasionally isn't settled and every read comes
        back 0x00; retry the reset a few times before giving up.
        """
        for attempt in range(1, 6):
            dev_id = self._rreg(chip_num, WHO_I_AM)
            if (dev_id & 0x1F) == 0x1E:
                logger.info("chip %d ID register = 0x%02X (ADS1299 comms OK)",
                            chip_num, dev_id)
                return
            logger.warning("chip %d ID read 0x%02X invalid; retrying reset (%d/5)",
                           chip_num, dev_id, attempt)
            self._send_command(chip_num, CMD_RESET)
            time.sleep(0.05)
            self._send_command(chip_num, CMD_SDATAC)
            time.sleep(2e-3)
        raise RuntimeError(
            f"ADS1299 chip {chip_num}: SPI comms failed -- ID register never "
            f"read a valid value. Check wiring, power, and SPI mode.")

    def _configure_adc(self, chip_num: int):
        """Send the full initialization sequence to one ADC chip."""
        self._send_command(chip_num, CMD_WAKEUP)
        self._send_command(chip_num, CMD_STOP)
        self._send_command(chip_num, CMD_RESET)
        time.sleep(0.05)                      # let the reset + oscillator settle
        self._send_command(chip_num, CMD_SDATAC)

        # Confirm SPI register access before writing anything we later trust.
        self._verify_comms(chip_num)

        # Register configuration (matches original PiEEG scripts)
        self._write_register(chip_num, 0x14, 0x80)  # GPIO
        # CONFIG1 selects the sample rate. 0x96 is 250 SPS, the board default
        # and an EEG choice; low three bits are the data-rate divider
        # (0x95=500, 0x94=1000, 0x93=2000, 0x92=4000). Surface EMG carries power
        # to ~450 Hz, so 250 SPS discards everything above 125 Hz.
        #
        # Read from the environment so this stays opt-in and reversible: unset,
        # the board behaves exactly as it always has. The register API refuses
        # CONFIG1 writes on purpose, because changing the rate under a running
        # filter chain is silent and destructive, so it has to happen here.
        #
        # PIEEG_OVERSAMPLE=k instead runs the chip k times faster and the
        # acquisition loop decimates to 250 SPS (sample_rate stays 250,
        # chip_rate is the real conversion rate).
        k = oversample_factor(self._num_channels)
        if k > 1 and os.environ.get("PIEEG_CONFIG1", "").strip():
            raise ValueError("set PIEEG_OVERSAMPLE or PIEEG_CONFIG1, not both")
        config1 = int(os.environ.get("PIEEG_CONFIG1", "")
                      or hex(OVERSAMPLE_CONFIG1[k]), 0)
        logger.info("WREG CONFIG1 <- 0x%02X (chip %s SPS%s)", config1,
                    config1_sample_rate(config1),
                    f", decimated x{k} to {OUTPUT_RATE}" if k > 1 else "")
        self._write_register(chip_num, CONFIG1, config1)
        self._config1 = config1
        self._oversample = k
        self._write_register(chip_num, CONFIG2, 0xD4)
        # Bias drive: all 8 P inputs and the shared reference (every N input
        # is tied to SRB1) feed the common-mode loop. An unconnected input
        # rails and would skew that average, so a partial montage wants
        # PIEEG_BIAS_DRIVE=0 until the sense mask follows lead-off status.
        self.bias_registers = dict(BIAS_DRIVE_ON if bias_drive_wanted(
            self._num_channels) else BIAS_DRIVE_OFF)
        logger.info("bias drive %s: %s",
                    "ON" if self.bias_registers[BIAS_SENSP] else "OFF",
                    {hex(k): hex(v) for k, v in self.bias_registers.items()})
        self._write_register(chip_num, CONFIG3, self.bias_registers[CONFIG3])
        self._write_register(chip_num, LOFF, LOFF_DC_95_5)  # DC lead-off, 95%/5%
        self._write_register(chip_num, BIAS_SENSP,
                             self.bias_registers[BIAS_SENSP])
        self._write_register(chip_num, BIAS_SENSN,
                             self.bias_registers[BIAS_SENSN])
        # Enable lead-off sensing on all 8 P and N inputs so the STATUS word
        # reports per-channel electrode contact. Additive: does not touch the
        # sample rate, gain, filtering, or channel data path.
        self._write_register(chip_num, LOFF_SENSP, LOFF_SENSE_ALL)
        self._write_register(chip_num, LOFF_SENSN, LOFF_SENSE_ALL)
        self._write_register(chip_num, 0x11, 0x00)          # LOFF_FLIP
        self._write_register(chip_num, 0x15, 0x20)          # MISC1
        # Power the lead-off comparators (PD_LOFF_COMP); without this the
        # LOFF_STATP/N status bits stay stuck and never flag a floating lead.
        self._write_register(chip_num, CONFIG4, CONFIG4_PD_LOFF_COMP)

        # Enable all 8 channels at PGA gain x24 (clinical EEG). Byte 0x60 is
        # decoded in the CHNSET_GAIN_X24_NORMAL comment above.
        for ch_reg in (CH1SET, CH2SET, CH3SET, CH4SET,
                       CH5SET, CH6SET, CH7SET, CH8SET):
            self._write_register(chip_num, ch_reg, CHNSET_GAIN_X24_NORMAL)
            logger.info("WREG CH@0x%02X <- 0x%02X (gain x24)",
                        ch_reg, CHNSET_GAIN_X24_NORMAL)

        # Verify the gain actually took. RREG works here because we are still
        # in SDATAC (continuous read not re-enabled yet). A wrong gain silently
        # corrupts every exported microvolt, so fail loudly on mismatch.
        self._assert_gain_x24(chip_num)

        self._send_command(chip_num, CMD_RDATAC)
        self._send_command(chip_num, CMD_START)

    def _rreg(self, chip_num: int, register: int) -> int:
        """Read one register value off the chip (must be in SDATAC).

        RREG opcode is 0x20|addr, followed by (count-1)=0 and one dummy byte
        that clocks the register value out. MUST use xfer2 (CS held low across
        all three bytes) at the slow register clock: with plain xfer / at 4 MHz
        this returns 0x00. The chip must not be streaming (RDATAC).
        """
        frame = [0x20 | register, 0x00, 0x00]
        if chip_num == 1:
            resp = self._spi1.xfer2(frame, REGISTER_SPEED_HZ)
        else:
            self._cs_set(0)
            resp = self._spi2.xfer2(frame, REGISTER_SPEED_HZ)
            self._cs_set(1)
        return resp[2] & 0xFF

    def _assert_gain_x24(self, chip_num: int):
        """Read back every CHnSET and confirm the PGA gain is x24."""
        codes = []
        for ch_reg in (CH1SET, CH2SET, CH3SET, CH4SET,
                       CH5SET, CH6SET, CH7SET, CH8SET):
            value = self._rreg(chip_num, ch_reg)
            code = (value >> 4) & 0b111
            codes.append(code)
            logger.info("RREG CH@0x%02X -> 0x%02X (gain code %d)",
                        ch_reg, value, code)
        if any(code != GAIN_CODE_X24 for code in codes):
            raise RuntimeError(
                f"ADS1299 chip {chip_num}: PGA gain readback FAILED. Expected "
                f"gain code {GAIN_CODE_X24} (x24) on all channels, got {codes}. "
                f"Refusing to run with an unverified gain.")
        self._pga_gain = PGA_GAIN_X24
        logger.info("chip %d PGA gain verified: x%d", chip_num, PGA_GAIN_X24)

    def _physical_lsb_uv(self) -> float:
        """Physically correct microvolts per ADC count for the programmed gain.

        Datasheet: uV = code * Vref / (gain * (2^23 - 1)). Uses the gain
        confirmed by register readback (falls back to x24 before configure).
        """
        gain = self._pga_gain or PGA_GAIN_X24
        return VREF_UV / (gain * FULL_SCALE_23)

    def _decode_channels(self, raw: list[int]) -> list[float]:
        """
        Decode 8 channels from a 27-byte SPI read into PHYSICAL microvolts.

        Bytes 0-2: status
        Bytes 3-26: 8 channels × 3 bytes (24-bit signed, MSB first)

        uV = code * Vref / (gain * (2^23 - 1)), using the PGA gain read back
        from the chip -- so the live stream, CSV, and LSL all carry physically
        correct microvolts. Kept to 4 decimals: that is far finer than one
        count (~0.0224 uV at gain 24), so the journal still inverts each value
        back to the exact integer ADC code (counts stay bit-for-bit 1:1).

        Uses the compiled ``pieeg_core.decode_channels`` (~30× faster) when
        available; that returns the older transport-scale uV (Vref/(2^24-1),
        no gain), so we rescale it to the physical scale with a single factor.
        """
        lsb_uv = self._physical_lsb_uv()

        if _native.HAS_NATIVE:
            # Native returns transport-scale uV; convert to physical. The factor
            # is (2^24-1) / (gain * (2^23-1)) -- i.e. add the gain and swap the
            # full-scale denominator, without re-reading the codes.
            gain = self._pga_gain or PGA_GAIN_X24
            correction = FULL_SCALE_PLUS_1 / (gain * FULL_SCALE_23)
            return [round(v * correction, 4) for v in _native.decode_channels(raw)]

        channels = []
        for i in range(3, 25, 3):
            raw_val = (raw[i] << 16) | (raw[i + 1] << 8) | raw[i + 2]

            # Two's complement conversion for 24-bit signed values
            if raw_val | SIGN_TEST == FULL_SCALE:
                signed_val = raw_val - NEGATIVE_OFFSET
            else:
                signed_val = raw_val

            channels.append(round(signed_val * lsb_uv, 4))

        return channels
