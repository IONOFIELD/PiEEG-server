#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""
Diagnose ADS1299 register READBACK-returns-zero on the PiEEG (Pi 4, spidev0.0).

Symptom: WREG writes CHnSET=0x60, but RREG reads back 0x00 on all channels.

This talks to the chip directly over /dev/spidev0.0 (chip 1 = hardware CE0), so
it is independent of the full driver. It follows the datasheet-ordered checks:

  1. SDATAC state  -- RREG/WREG are IGNORED while the device streams (RDATAC).
  2. ID register   -- read register 0x00; a valid ADS1299 ID is non-zero with
                      bit4=1 and the low bits 1_1110. If ID reads 0x00 the READ
                      PATH is broken (not the CHnSET write).
  3. Mode / timing -- try SPI speeds and a t_SDECODE gap between the RREG
                      command and the data byte (the ADS1299 needs ~4 tCLK to
                      decode a command before it drives MISO).
  4. CHnSET        -- only then, write 0x60 and read it back.

Every byte written/read is printed. Nothing here changes calibration or export.

DATASHEET NOTES (ADS1299, TI SBAS499; verified, not assumed)
  Commands:  WAKEUP=0x02 STANDBY=0x04 RESET=0x06 START=0x08 STOP=0x0A
             RDATAC=0x10 SDATAC=0x11 RDATA=0x12
  RREG:      first byte 0x20|addr, second byte (num_regs-1); register data is
             shifted out AFTER a command-decode delay of ~4 tCLK.
  WREG:      first byte 0x40|addr, second byte (num_regs-1), then data byte(s).
  ID reg:    address 0x00; bit4 reads 1, DEV_ID[3:2]=11, NU_CH[1:0]=10 (8ch) ->
             low 5 bits = 0b1_1110 = 0x1E; REV_ID in the top 3 bits (so 0x1E,
             0x3E, ... are all valid). tCLK (internal osc) = 2.048 MHz.
"""

import time

import spidev

# --- ADS1299 opcodes ------------------------------------------------------ #
WAKEUP, STOP, RESET, START = 0x02, 0x0A, 0x06, 0x08
RDATAC, SDATAC = 0x10, 0x11
ID_REG = 0x00
CH1SET = 0x05

# ~4 tCLK decode delay at tCLK = 2.048 MHz is ~2 us; use a comfortable margin.
T_SDECODE_S = 10e-6


def open_spi(speed_hz=4_000_000, mode=0b01):
    spi = spidev.SpiDev()
    spi.open(0, 0)                 # /dev/spidev0.0  (chip 1, hardware CE0)
    spi.max_speed_hz = speed_hz
    spi.mode = mode                # ADS1299 = SPI mode 1 (CPOL=0, CPHA=1)
    spi.bits_per_word = 8
    return spi


def cmd(spi, opcode, settle_s=1e-3):
    """Send a one-byte command and let the chip decode it."""
    spi.xfer2([opcode])
    time.sleep(settle_s)


def reset_to_known_state(spi):
    """Datasheet power-up/exit-stream sequence, ending in SDATAC (not streaming)."""
    cmd(spi, WAKEUP)
    cmd(spi, STOP)
    cmd(spi, RESET)
    time.sleep(1e-3)               # wait out the reset (>18 tCLK)
    cmd(spi, SDATAC)               # <-- REQUIRED before any RREG/WREG
    time.sleep(1e-3)


def rreg_contiguous(spi, reg):
    """RREG in one CS-low transaction: [0x20|reg, count-1, dummy]. Returns 3 bytes."""
    return spi.xfer2([0x20 | reg, 0x00, 0x00])


def rreg_split(spi, reg, gap_s=T_SDECODE_S):
    """RREG as command, then a decode gap, then clock the data byte separately."""
    cmdbytes = spi.xfer2([0x20 | reg, 0x00])   # opcode + (num-1)
    time.sleep(gap_s)                          # t_SDECODE
    data = spi.xfer2([0x00])                   # clock the register value out
    return cmdbytes, data


def wreg(spi, reg, value):
    """Write one register: [0x40|reg, count-1, value]."""
    spi.xfer2([0x40 | reg, 0x00, value & 0xFF])


def id_is_valid(byte):
    """ADS1299 ID: bit4 must be 1 and low 5 bits must be 0b1_1110 (0x1E)."""
    return (byte & 0x1F) == 0x1E


def hexb(bs):
    return "[" + ", ".join(f"0x{b:02X}" for b in bs) + "]"


def main():
    print("=" * 70)
    print("ADS1299 SPI register-readback diagnostic (Pi 4, /dev/spidev0.0)")
    print("=" * 70)

    spi = open_spi()
    print(f"SPI: mode={spi.mode} (expect 1 for ADS1299), "
          f"speed={spi.max_speed_hz} Hz, bits={spi.bits_per_word}")

    # ---- Step 1: SDATAC state ---------------------------------------- #
    print("\n[1] SDATAC / RDATAC state")
    reset_to_known_state(spi)
    print("    Issued WAKEUP, STOP, RESET, SDATAC -> device should NOT be streaming.")
    # Demonstrate the RDATAC trap: read ID while streaming, then after SDATAC.
    cmd(spi, RDATAC)
    id_in_rdatac = rreg_contiguous(spi, ID_REG)
    cmd(spi, SDATAC)
    id_in_sdatac = rreg_contiguous(spi, ID_REG)
    print(f"    ID read WHILE in RDATAC : {hexb(id_in_rdatac)} -> value 0x{id_in_rdatac[2]:02X}")
    print(f"    ID read AFTER SDATAC    : {hexb(id_in_sdatac)} -> value 0x{id_in_sdatac[2]:02X}")

    # ---- Step 2: ID register (read-path sanity) ---------------------- #
    print("\n[2] ID register (0x00) read-path sanity check")
    val = id_in_sdatac[2]
    print(f"    raw bytes = {hexb(id_in_sdatac)}   ID = 0x{val:02X}")
    print(f"    valid ADS1299 ID? {id_is_valid(val)}  "
          f"(need bit4=1 & low5=0x1E; e.g. 0x1E/0x3E)")

    # ---- Step 3: mode / timing sweep --------------------------------- #
    print("\n[3] SPI mode / speed / t_SDECODE timing sweep (reading ID)")
    for speed in (4_000_000, 2_000_000, 1_000_000, 500_000, 250_000):
        s = open_spi(speed_hz=speed)
        reset_to_known_state(s)
        contig = rreg_contiguous(s, ID_REG)
        _, split = rreg_split(s, ID_REG)
        print(f"    speed={speed:>8} Hz  contiguous=0x{contig[2]:02X} "
              f"({'ok' if id_is_valid(contig[2]) else 'BAD'})   "
              f"split+gap=0x{split[0]:02X} "
              f"({'ok' if id_is_valid(split[0]) else 'BAD'})")
        s.close()

    # Also try the other SPI modes in case CPOL/CPHA is wrong.
    print("    SPI mode sweep (contiguous ID read @ 1 MHz):")
    for mode in (0b00, 0b01, 0b10, 0b11):
        s = open_spi(speed_hz=1_000_000, mode=mode)
        reset_to_known_state(s)
        r = rreg_contiguous(s, ID_REG)
        print(f"      mode {mode:02b}: 0x{r[2]:02X} "
              f"({'ok' if id_is_valid(r[2]) else 'BAD'})")
        s.close()

    # ---- Step 4: CHnSET write -> readback ---------------------------- #
    print("\n[4] CHnSET write/readback (the reported failure)")
    s = open_spi(speed_hz=1_000_000)
    reset_to_known_state(s)
    wreg(s, CH1SET, 0x60)          # gain x24, normal input
    time.sleep(1e-3)
    contig = rreg_contiguous(s, CH1SET)
    _, split = rreg_split(s, CH1SET)
    print(f"    wrote CH1SET <- 0x60")
    print(f"    readback contiguous = 0x{contig[2]:02X}")
    print(f"    readback split+gap  = 0x{split[0]:02X}")
    gain_code = (split[0] >> 4) & 0b111
    print(f"    gain code from split readback = {gain_code} "
          f"({'x24 OK' if gain_code == 0b110 else 'MISMATCH'})")
    s.close()
    spi.close()

    print("\n" + "=" * 70)
    print("Read the ID line in [2] and the sweep in [3] to localize the fault.")
    print("=" * 70)


if __name__ == "__main__":
    main()
