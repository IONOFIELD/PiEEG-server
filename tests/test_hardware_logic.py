"""
Tests for PiEEGHardware that can run WITHOUT a Raspberry Pi.

Only tests logic that doesn't require actual GPIO/SPI hardware:
- ADC value decoding (24-bit signed → µV)
- Spike detection logic
- SPI frame validation (status byte checking)
- Register/command constants
- Register configuration & shadow state
"""

import pytest

from pieeg_server.hardware import (
    SIGN_TEST, FULL_SCALE, FULL_SCALE_PLUS_1, NEGATIVE_OFFSET,
    VREF_UV, SPIKE_THRESHOLD, SPIKE_RESET_AFTER,
    EXPECTED_STATUS, BYTES_PER_READ,
    CS_PIN, DRDY_PIN, DRDY_PIN_2, SPI_SPEED_HZ,
    CH1SET, CH2SET, CH3SET, CH4SET, CH5SET, CH6SET, CH7SET, CH8SET,
    LOFF, LOFF_SENSP, LOFF_SENSN, LOFF_STATP, LOFF_STATN, CONFIG4,
    CONFIG4_PD_LOFF_COMP, LOFF_SENSE_ALL,
    STATUS_SYNC_MASK, STATUS_SYNC_VALUE,
    parse_leadoff_status, leadoff_state, _status_sync_ok,
    config1_sample_rate, classify_contact, contact_from_signal,
    PiEEGHardware,
)


# We can't call PiEEGHardware.open() on Windows, but we can test its
# pure-logic methods by constructing an instance and bypassing open().


class TestADCConstants:
    """Verify ADC conversion constants are correct for ADS1299."""

    def test_sign_test_is_2_23_minus_1(self):
        assert SIGN_TEST == 0x7FFFFF == 2**23 - 1

    def test_full_scale_is_2_24_minus_1(self):
        assert FULL_SCALE == 0xFFFFFF == 2**24 - 1

    def test_vref_is_4_5V(self):
        assert VREF_UV == 4.5e6  # 4.5V in µV

    def test_spi_speed_is_4mhz(self):
        assert SPI_SPEED_HZ == 4_000_000

    def test_bytes_per_read_is_27(self):
        # 3 status + 8 channels × 3 bytes = 27
        assert BYTES_PER_READ == 27

    def test_gpio_pins(self):
        assert CS_PIN == 19
        assert DRDY_PIN == 26
        assert DRDY_PIN_2 == 13


class TestSpikeDetection:
    """Test the spike detection logic (critical for data quality)."""

    def _make_hw(self):
        """Create a PiEEGHardware without initializing GPIO/SPI."""
        hw = PiEEGHardware.__new__(PiEEGHardware)
        hw._last_valid_value = None
        hw._spike_count = 0
        hw._consecutive_rejects = 0
        hw._spike_threshold = SPIKE_THRESHOLD
        hw._spike_reset_after = SPIKE_RESET_AFTER
        return hw

    def _raw_with_last_3(self, b24, b25, b26):
        """Create a 27-byte raw frame with specified bytes at positions 24-26."""
        raw = [0] * 27
        raw[24] = b24
        raw[25] = b25
        raw[26] = b26
        return raw

    def test_first_frame_always_rejected(self):
        """First frame is skipped (no reference value to compare against)."""
        hw = self._make_hw()
        raw = self._raw_with_last_3(0, 0, 100)
        assert hw._is_valid_frame(raw) is False
        # But it sets the baseline
        assert hw._last_valid_value is not None

    def test_small_change_accepted(self):
        hw = self._make_hw()
        # First frame (sets baseline)
        hw._is_valid_frame(self._raw_with_last_3(0, 0, 100))
        # Second frame with small change
        assert hw._is_valid_frame(self._raw_with_last_3(0, 0, 110)) is True

    def test_large_spike_rejected(self):
        hw = self._make_hw()
        # Baseline at value 100
        hw._is_valid_frame(self._raw_with_last_3(0, 0, 100))
        # Jump of >5000 — this is a spike
        # Encode value 6000 as 24-bit: 0x001770
        val = 6000
        b24 = (val >> 16) & 0xFF
        b25 = (val >> 8) & 0xFF
        b26 = val & 0xFF
        assert hw._is_valid_frame(self._raw_with_last_3(b24, b25, b26)) is False
        assert hw._spike_count == 1

    def test_negative_values_handled(self):
        """Signed 24-bit negative values should be decoded correctly."""
        hw = self._make_hw()
        # Set baseline to 0
        hw._is_valid_frame(self._raw_with_last_3(0, 0, 0))
        # Small negative (-10): 24-bit two's complement = 0xFFFFF6
        raw = self._raw_with_last_3(0xFF, 0xFF, 0xF6)
        assert hw._is_valid_frame(raw) is True

    def test_spike_count_increments(self):
        hw = self._make_hw()
        hw._is_valid_frame(self._raw_with_last_3(0, 0, 0))  # baseline

        # Produce multiple spikes (fewer than SPIKE_RESET_AFTER)
        for i in range(5):
            val = 10000 + i * 1000  # all > threshold
            b24 = (val >> 16) & 0xFF
            b25 = (val >> 8) & 0xFF
            b26 = val & 0xFF
            hw._is_valid_frame(self._raw_with_last_3(b24, b25, b26))

        assert hw._spike_count == 5

    def test_spike_filter_resets_after_consecutive_rejects(self):
        """After SPIKE_RESET_AFTER consecutive rejects, baseline re-syncs."""
        hw = self._make_hw()
        hw._is_valid_frame(self._raw_with_last_3(0, 0, 0))  # baseline at 0

        # Send the same far-away value repeatedly (simulates electrode connect)
        val = 100_000
        b24 = (val >> 16) & 0xFF
        b25 = (val >> 8) & 0xFF
        b26 = val & 0xFF
        far_raw = self._raw_with_last_3(b24, b25, b26)

        for i in range(SPIKE_RESET_AFTER - 1):
            assert hw._is_valid_frame(far_raw) is False

        # The next one triggers the reset and is accepted
        assert hw._is_valid_frame(far_raw) is True
        assert hw._consecutive_rejects == 0
        assert hw._last_valid_value == val

        # Subsequent close values are accepted normally
        val2 = val + 10
        b24 = (val2 >> 16) & 0xFF
        b25 = (val2 >> 8) & 0xFF
        b26 = val2 & 0xFF
        assert hw._is_valid_frame(self._raw_with_last_3(b24, b25, b26)) is True

    def test_status_header_constant(self):
        """Expected status bytes should be (0xC0, 0x00, 0x08)."""
        assert EXPECTED_STATUS == (192, 0, 8)
        assert EXPECTED_STATUS == (0xC0, 0x00, 0x08)

    def test_custom_threshold_accepts_larger_jumps(self):
        """Raising spike_threshold allows larger jumps through."""
        hw = self._make_hw()
        hw.spike_threshold = 10000
        hw._is_valid_frame(self._raw_with_last_3(0, 0, 100))  # baseline
        # Jump of 6000 — within new threshold
        val = 6100
        b24 = (val >> 16) & 0xFF
        b25 = (val >> 8) & 0xFF
        b26 = val & 0xFF
        assert hw._is_valid_frame(self._raw_with_last_3(b24, b25, b26)) is True

    def test_custom_reset_after(self):
        """Lowering spike_reset_after resets sooner."""
        hw = self._make_hw()
        hw.spike_reset_after = 5
        hw._is_valid_frame(self._raw_with_last_3(0, 0, 0))  # baseline

        val = 100_000
        b24 = (val >> 16) & 0xFF
        b25 = (val >> 8) & 0xFF
        b26 = val & 0xFF
        far_raw = self._raw_with_last_3(b24, b25, b26)

        for _ in range(4):
            assert hw._is_valid_frame(far_raw) is False
        # 5th triggers reset
        assert hw._is_valid_frame(far_raw) is True
        assert hw._consecutive_rejects == 0

    def test_threshold_minus_one_disables_filter(self):
        """Setting threshold to -1 disables spike rejection entirely."""
        hw = self._make_hw()
        hw.spike_threshold = -1
        assert hw.spike_threshold == -1
        # Even a huge jump should be accepted
        assert hw._is_valid_frame(self._raw_with_last_3(0, 0, 0)) is True
        val = 8_000_000  # massive jump
        b24 = (val >> 16) & 0xFF
        b25 = (val >> 8) & 0xFF
        b26 = val & 0xFF
        assert hw._is_valid_frame(self._raw_with_last_3(b24, b25, b26)) is True

    def test_threshold_property_clamps_minimum(self):
        """spike_threshold property enforces min=0 (except -1)."""
        hw = self._make_hw()
        hw.spike_threshold = -100
        assert hw.spike_threshold == 0

    def test_reset_after_property_clamps_minimum(self):
        """spike_reset_after property enforces min=1."""
        hw = self._make_hw()
        hw.spike_reset_after = 0
        assert hw.spike_reset_after == 1


class TestADCDecoding:
    """Test the 24-bit signed integer to PHYSICAL µV conversion.

    _decode_channels is now an instance method (it needs the PGA gain), and it
    emits physically correct microvolts:
        µV = round(signed_val * Vref / (gain * (2^23 - 1)), 4)
    """

    GAIN = 24

    def _hw(self):
        """A PiEEGHardware instance without GPIO/SPI, gain fixed at x24."""
        hw = PiEEGHardware.__new__(PiEEGHardware)
        hw._pga_gain = self.GAIN
        hw._num_channels = 8
        return hw

    def _uv(self, signed_val):
        return round(signed_val * VREF_UV / (self.GAIN * (2**23 - 1)), 4)

    def test_decode_channels_returns_8_values(self):
        channels = self._hw()._decode_channels([0] * 27)
        assert len(channels) == 8

    def test_decode_channels_zero(self):
        channels = self._hw()._decode_channels([0] * 27)
        assert all(v == 0.0 for v in channels)

    def test_decode_channels_positive(self):
        """0x400000 (positive, below sign threshold) → expected physical µV."""
        raw = [0] * 27
        raw[3] = 0x40  # ch1 MSB
        channels = self._hw()._decode_channels(raw)
        # 0x400000 = 4194304, signed_val = 4194304
        assert channels[0] == self._uv(4194304)
        assert channels[0] > 0

    def test_decode_channels_negative(self):
        """0xFFFFFF (all ones) decodes to signed_val = +1 (per NEGATIVE_OFFSET)."""
        raw = [0] * 27
        raw[3] = raw[4] = raw[5] = 0xFF  # ch1
        channels = self._hw()._decode_channels(raw)
        # signed_val = 16777215 - 16777214 = 1
        assert channels[0] == self._uv(1)

    def test_decode_full_negative(self):
        """0x800000 (MSB set) decodes to signed_val = -8388606."""
        raw = [0] * 27
        raw[3] = 0x80
        channels = self._hw()._decode_channels(raw)
        # signed_val = 8388608 - 16777214 = -8388606
        assert channels[0] == self._uv(-8388606)
        assert channels[0] < 0

    def test_all_channels_independent(self):
        """Different values in different channel slots decode independently."""
        raw = [0] * 27
        raw[5] = 0x01   # Ch1 = 0x000001
        raw[8] = 0x02   # Ch2 = 0x000002
        channels = self._hw()._decode_channels(raw)
        assert channels[0] != channels[1]
        assert channels[0] == self._uv(1)
        assert channels[1] == self._uv(2)


class TestRegisterState:
    """Test register shadow state tracking (no SPI required)."""

    def _make_hw(self):
        """Create a PiEEGHardware without initializing GPIO/SPI."""
        hw = PiEEGHardware.__new__(PiEEGHardware)
        hw._register_state = {}
        hw._num_channels = 8
        hw._last_valid_value = None
        hw._spike_count = 0
        hw._consecutive_rejects = 0
        hw._spike_threshold = SPIKE_THRESHOLD
        hw._spike_reset_after = SPIKE_RESET_AFTER
        return hw

    def test_register_state_starts_empty(self):
        hw = self._make_hw()
        assert hw.register_state == {}

    def test_register_state_is_copy(self):
        """register_state property returns a copy, not a reference."""
        hw = self._make_hw()
        state = hw.register_state
        state[0xFF] = 0x99
        assert 0xFF not in hw.register_state

    def test_ch_regs_constant(self):
        """CH_REGS should contain all 8 channel register addresses."""
        assert PiEEGHardware.CH_REGS == (
            CH1SET, CH2SET, CH3SET, CH4SET,
            CH5SET, CH6SET, CH7SET, CH8SET,
        )
        assert len(PiEEGHardware.CH_REGS) == 8
        # Sequential from 0x05 to 0x0C
        assert PiEEGHardware.CH_REGS == tuple(range(0x05, 0x0D))


class TestSampleRate:
    """CONFIG1 DR bits -> sample rate, exposed as PiEEGHardware.sample_rate."""

    @pytest.mark.parametrize("config1, rate", [
        (0x96, 250), (0x95, 500), (0x94, 1000), (0x93, 2000),
        (0x92, 4000), (0x91, 8000), (0x90, 16000),
    ])
    def test_config1_rate_table(self, config1, rate):
        assert config1_sample_rate(config1) == rate

    def test_reserved_rate_code_is_none(self):
        assert config1_sample_rate(0x97) is None

    def test_none_before_configured(self):
        hw = PiEEGHardware.__new__(PiEEGHardware)
        assert hw.sample_rate is None
        hw._config1 = None
        assert hw.sample_rate is None

    def test_follows_written_config1(self):
        hw = PiEEGHardware.__new__(PiEEGHardware)
        hw._config1 = 0x95
        assert hw.sample_rate == 500


class TestMockRegisterConfig:
    """Test MockHardware register config and input mode switching."""

    def test_set_input_short_switches_mode(self):
        from pieeg_server.mock import MockHardware
        hw = MockHardware(num_channels=8)
        hw.open()

        hw.set_input_short()
        # All channels should be in shorted mode (0x01)
        assert all(m == 0x01 for m in hw._ch_modes[:8])
        for reg in hw.CH_REGS:
            assert hw.register_state.get(reg) == 0x01

    def test_set_input_normal_restores_mode(self):
        from pieeg_server.mock import MockHardware
        hw = MockHardware(num_channels=8)
        hw.open()

        hw.set_input_short()
        hw.set_input_normal()
        assert all(m == 0x00 for m in hw._ch_modes[:8])
        for reg in hw.CH_REGS:
            assert hw.register_state.get(reg) == 0x00

    def test_shorted_mode_produces_low_noise(self):
        """In shorted mode, samples should be very small (±few µV)."""
        import statistics
        from pieeg_server.mock import MockHardware
        hw = MockHardware(num_channels=8)
        hw.open()
        hw.set_input_short()

        samples = [hw.read_sample() for _ in range(500)]
        # Check all channels have low RMS
        for ch in range(8):
            values = [s[ch] for s in samples]
            rms = statistics.stdev(values)
            assert rms < 5, f"Channel {ch} RMS {rms} too high for shorted mode"

    def test_normal_mode_produces_alpha(self):
        """In normal mode, samples should have higher amplitude (alpha rhythm)."""
        import statistics
        from pieeg_server.mock import MockHardware
        hw = MockHardware(num_channels=8)
        hw.open()

        samples = [hw.read_sample() for _ in range(500)]
        values = [s[0] for s in samples]
        rms = statistics.stdev(values)
        assert rms > 5, f"Channel 0 RMS {rms} too low for normal mode"

    def test_configure_registers_updates_state(self):
        from pieeg_server.mock import MockHardware
        hw = MockHardware(num_channels=8)
        hw.open()

        hw.configure_registers({0x05: 0x05, 0x06: 0x05})
        assert hw.register_state[0x05] == 0x05
        assert hw.register_state[0x06] == 0x05

    def test_register_state_is_copy(self):
        from pieeg_server.mock import MockHardware
        hw = MockHardware(num_channels=8)
        hw.open()

        hw.configure_registers({0x05: 0x01})
        state = hw.register_state
        state[0x05] = 0xFF
        assert hw.register_state[0x05] == 0x01

    def test_per_channel_mode_individual(self):
        """Individual channel register changes affect only that channel."""
        from pieeg_server.mock import MockHardware
        hw = MockHardware(num_channels=8)
        hw.open()

        # Set only CH1 to shorted, rest stay normal
        hw.configure_registers({0x05: 0x01})
        assert hw._ch_modes[0] == 0x01  # CH1 shorted
        assert hw._ch_modes[1] == 0x00  # CH2 still normal

    def test_test_signal_mode_produces_square_wave(self):
        """Test signal mode should produce large ±1800 µV values."""
        from pieeg_server.mock import MockHardware
        hw = MockHardware(num_channels=8)
        hw.open()

        hw.configure_registers({0x05: 0x05})  # CH1 = test signal
        samples = [hw.read_sample()[0] for _ in range(250)]
        # Test signal alternates ±1800, check amplitude is large
        assert max(abs(v) for v in samples) > 1000

    def test_16ch_mirror_registers(self):
        """On 16-channel mock, CHnSET registers mirror to channels 9-16."""
        from pieeg_server.mock import MockHardware
        hw = MockHardware(num_channels=16)
        hw.open()

        hw.set_input_short()
        # Channels 0-7 AND 8-15 should all be shorted
        assert all(m == 0x01 for m in hw._ch_modes)

        hw.set_input_normal()
        assert all(m == 0x00 for m in hw._ch_modes)

    def test_16ch_individual_register_mirrors(self):
        """Setting CH1SET on 16ch affects ch0 AND ch8."""
        from pieeg_server.mock import MockHardware
        hw = MockHardware(num_channels=16)
        hw.open()

        hw.configure_registers({0x05: 0x05})  # CH1SET = test signal
        assert hw._ch_modes[0] == 0x05   # ch1
        assert hw._ch_modes[8] == 0x05   # ch9 (mirror)
        assert hw._ch_modes[1] == 0x00   # ch2 unchanged
        assert hw._ch_modes[9] == 0x00   # ch10 unchanged

    def test_16ch_shorted_produces_low_noise_all_channels(self):
        """16-channel shorted mode: all 16 channels should have low noise."""
        import statistics
        from pieeg_server.mock import MockHardware
        hw = MockHardware(num_channels=16)
        hw.open()
        hw.set_input_short()

        samples = [hw.read_sample() for _ in range(500)]
        for ch in range(16):
            values = [s[ch] for s in samples]
            rms = statistics.stdev(values)
            assert rms < 5, f"Channel {ch} RMS {rms} too high for shorted 16ch mode"


class TestLeadOffRegisters:
    """Verify the lead-off register addresses/values match the ADS1299 datasheet."""

    def test_register_addresses(self):
        assert LOFF == 0x04
        assert LOFF_SENSP == 0x0F
        assert LOFF_SENSN == 0x10
        assert LOFF_STATP == 0x12
        assert LOFF_STATN == 0x13
        assert CONFIG4 == 0x17

    def test_comparator_power_bit(self):
        # PD_LOFF_COMP is CONFIG4 bit 1.
        assert CONFIG4_PD_LOFF_COMP == 0x02

    def test_sense_all_channels(self):
        assert LOFF_SENSE_ALL == 0xFF

    def test_sync_marker(self):
        assert STATUS_SYNC_MASK == 0xF0
        assert STATUS_SYNC_VALUE == 0xC0
        # The historical fixed header must still pass the relaxed sync check.
        assert (EXPECTED_STATUS[0] & STATUS_SYNC_MASK) == STATUS_SYNC_VALUE


class TestLeadOffState:
    """green/red verdict from the electrode (P) flag; N is ignored because
    it reads off on every PiEEG-8 channel regardless of REF."""

    def test_connected_is_green(self):
        assert leadoff_state(False) == "green"
        assert leadoff_state(False, True) == "green"   # stuck N flag ignored

    def test_electrode_off_is_red(self):
        assert leadoff_state(True) == "red"
        assert leadoff_state(True, False) == "red"


class TestClassifyContact:
    """The wiring signatures measured on the bench (leads/REF/BIO joined,
    then pulled one at a time)."""

    @staticmethod
    def _status(p_off=()):
        return [{"ch": c, "p_off": c in p_off, "n_off": True} for c in range(1, 9)]

    def test_all_connected(self):
        r = classify_contact(self._status(), [False] * 8)
        assert r == {"leads": ["green"] * 8, "ref": "green", "gnd": "green"}

    def test_one_lead_off_rails_alone(self):
        railed = [True] + [False] * 7
        r = classify_contact(self._status(p_off={1}), railed)
        assert r["leads"][0] == "red" and r["leads"][1:] == ["green"] * 7
        assert (r["ref"], r["gnd"]) == ("green", "green")

    def test_ref_off_rails_the_connected_leads(self):
        # bench: E3/E4/E6 loose (flag off, in range), the rest on but railed
        loose = {3, 4, 6}
        railed = [c not in loose for c in range(1, 9)]
        r = classify_contact(self._status(p_off=loose), railed)
        assert r["ref"] == "red" and r["gnd"] == "green"

    def test_gnd_off_flags_everything_off_without_rails(self):
        r = classify_contact(self._status(p_off=set(range(1, 9))), [False] * 8)
        assert r["gnd"] == "red" and r["ref"] is None
        assert r["leads"] == ["red"] * 8

    def test_nothing_attached_is_unknown(self):
        r = classify_contact(self._status(p_off=set(range(1, 9))), [True] * 8)
        assert r["gnd"] is None and r["ref"] is None

    def test_large_shared_signal_means_ref_floating(self):
        r = classify_contact(self._status(), [False] * 8, common_uv=1222.0)
        assert r["ref"] == "red" and r["gnd"] == "green"


class TestContactFromSignal:
    """Signal features measured on the bench, synthesised."""

    FS = 4.5e6 / 24

    @staticmethod
    def _status(p_off=()):
        return [{"ch": c, "p_off": c in p_off, "n_off": True} for c in range(1, 9)]

    def _t(self):
        import numpy as np
        return np.arange(62) / 250

    def test_ref_floating_without_railing(self):
        import numpy as np
        rng = np.random.default_rng(0)
        shared = -67600 + 1700 * np.sin(2 * np.pi * 60 * self._t())
        block = shared[:, None] + rng.normal(0, 1, (62, 8))
        r = contact_from_signal(self._status(), block, self.FS)
        assert r["ref"] == "red"

    def test_quiet_short_is_ref_on(self):
        import numpy as np
        block = np.random.default_rng(0).normal(5, 0.3, (62, 8))
        assert contact_from_signal(self._status(), block, self.FS)["ref"] == "green"

    def test_big_but_independent_signals_are_not_ref_floating(self):
        import numpy as np
        block = np.random.default_rng(0).normal(0, 1500, (62, 8))
        assert contact_from_signal(self._status(), block, self.FS)["ref"] == "green"

    def test_railed_lead_and_loose_lead(self):
        import numpy as np
        block = np.random.default_rng(0).normal(0, 0.3, (62, 8))
        block[:, 0] = self.FS
        r = contact_from_signal(self._status(p_off={1}), block, self.FS)
        assert r["leads"][0] == "red" and r["ref"] == "green" and r["gnd"] == "green"


class TestLeadOffStatusParsing:
    """Parse a synthetic 24-bit STATUS word into per-channel off flags.

    STATUS layout (MSB first): 1100 + LOFF_STATP[7:0] + LOFF_STATN[7:0]
    + GPIO[3:0]. This encoder mirrors that so tests read like the register map.
    """

    @staticmethod
    def _status_bytes(statp=0, statn=0, gpio=0):
        word = (0xC << 20) | ((statp & 0xFF) << 12) | ((statn & 0xFF) << 4) | (gpio & 0x0F)
        return [(word >> 16) & 0xFF, (word >> 8) & 0xFF, word & 0xFF]

    def test_all_connected_reports_no_off(self):
        chans = parse_leadoff_status(self._status_bytes())
        assert len(chans) == 8
        assert [c["ch"] for c in chans] == list(range(1, 9))
        assert all(c["off"] is False for c in chans)
        assert all(c["p_off"] is False and c["n_off"] is False for c in chans)
        assert all(c["state"] == "green" for c in chans)

    def test_channel1_positive_off(self):
        chans = parse_leadoff_status(self._status_bytes(statp=0b0000_0001))
        assert chans[0]["p_off"] is True
        assert chans[0]["n_off"] is False
        assert chans[0]["off"] is True
        assert chans[0]["state"] == "red"
        # Only channel 1 flagged
        assert all(c["off"] is False for c in chans[1:])

    def test_channel8_negative_off(self):
        chans = parse_leadoff_status(self._status_bytes(statn=0b1000_0000))
        assert chans[7]["ch"] == 8
        assert chans[7]["n_off"] is True           # passed through raw
        assert chans[7]["p_off"] is False
        assert chans[7]["off"] is False            # N alone isn't contact loss
        assert chans[7]["state"] == "green"
        assert all(c["off"] is False for c in chans[:7])

    def test_mixed_pattern(self):
        # ch2 P off, ch5 N off, ch3 both off
        statp = (1 << 1) | (1 << 2)          # channels 2, 3 (P)
        statn = (1 << 4) | (1 << 2)          # channels 5, 3 (N)
        chans = parse_leadoff_status(self._status_bytes(statp=statp, statn=statn))
        off = {c["ch"] for c in chans if c["off"]}
        assert off == {2, 3}                             # electrode (P) flags
        assert chans[2]["p_off"] and chans[2]["n_off"]   # ch3 both
        assert chans[2]["state"] == "red"
        assert chans[1]["p_off"] and not chans[1]["n_off"]  # ch2 P only
        assert chans[1]["state"] == "red"
        assert chans[4]["n_off"] and not chans[4]["p_off"]  # ch5 N only
        assert chans[4]["state"] == "green"
        assert chans[0]["state"] == "green"              # ch1 untouched

    def test_gpio_bits_ignored(self):
        # GPIO nibble must not leak into any channel flag.
        chans = parse_leadoff_status(self._status_bytes(gpio=0b1111))
        assert all(c["off"] is False for c in chans)

    def test_channel_offset_for_second_chip(self):
        chans = parse_leadoff_status(self._status_bytes(statp=0b0000_0001),
                                     channel_offset=8)
        assert [c["ch"] for c in chans] == list(range(9, 17))
        assert chans[0]["ch"] == 9 and chans[0]["off"] is True

    def test_sync_ok_helper(self):
        assert _status_sync_ok(self._status_bytes()) is True
        assert _status_sync_ok(self._status_bytes(statp=0xFF, statn=0xFF, gpio=0xF)) is True
        # A frame that lost the 1100 marker is rejected.
        assert _status_sync_ok([0x00, 0x00, 0x00]) is False
        assert _status_sync_ok([0x30, 0x00, 0x00]) is False


class TestPiEEGLeadOffState:
    """PiEEGHardware caches lead-off state from the data-stream STATUS word."""

    def _make_hw(self, num_channels=8):
        hw = PiEEGHardware.__new__(PiEEGHardware)
        hw._num_channels = num_channels
        hw._leadoff = []
        return hw

    @staticmethod
    def _frame(statp=0, statn=0, gpio=0):
        raw = [0] * 27
        word = (0xC << 20) | ((statp & 0xFF) << 12) | ((statn & 0xFF) << 4) | (gpio & 0x0F)
        raw[0], raw[1], raw[2] = (word >> 16) & 0xFF, (word >> 8) & 0xFF, word & 0xFF
        return raw

    def test_status_none_before_first_read(self):
        assert self._make_hw().leadoff_status() is None

    def test_update_from_single_chip(self):
        hw = self._make_hw(8)
        hw._update_leadoff(self._frame(statp=0b0000_0001))
        status = hw.leadoff_status()
        assert len(status) == 8
        assert status[0]["off"] is True
        assert all(c["off"] is False for c in status[1:])

    def test_update_from_two_chips_offsets_channels(self):
        hw = self._make_hw(16)
        # chip1: ch1 off; chip2: ch1-of-chip2 off => reported as ch9
        hw._update_leadoff(self._frame(statp=0b0000_0001),
                           self._frame(statp=0b0000_0001))
        status = hw.leadoff_status()
        assert len(status) == 16
        assert [c["ch"] for c in status] == list(range(1, 17))
        off = {c["ch"] for c in status if c["off"]}
        assert off == {1, 9}

    def test_corrupt_frame_keeps_previous_state(self):
        hw = self._make_hw(8)
        hw._update_leadoff(self._frame(statp=0b0000_0001))
        # A desynced frame (no 1100 marker) must not overwrite good state.
        hw._update_leadoff([0x00, 0x00, 0x00])
        assert hw.leadoff_status()[0]["off"] is True

    def test_status_returns_copy(self):
        hw = self._make_hw(8)
        hw._update_leadoff(self._frame())
        status = hw.leadoff_status()
        status[0]["off"] = True
        assert hw.leadoff_status()[0]["off"] is False


class TestMockLeadOff:
    """Mock reports all-connected by default and honors a fixed pattern."""

    def test_all_connected_by_default(self):
        from pieeg_server.mock import MockHardware
        hw = MockHardware(num_channels=8)
        hw.open()
        status = hw.leadoff_status()
        assert len(status) == 8
        assert all(c["off"] is False for c in status)

    def test_set_pattern(self):
        from pieeg_server.mock import MockHardware
        hw = MockHardware(num_channels=8)
        hw.open()
        hw.set_leadoff_pattern([2, 5])
        status = hw.leadoff_status()
        off = {c["ch"] for c in status if c["off"]}
        assert off == {2, 5}
        # A flagged electrode is red; its neighbours stay green.
        assert status[1]["state"] == "red" and status[4]["state"] == "red"
        assert status[0]["state"] == "green"

    def test_16ch_reports_all_channels(self):
        from pieeg_server.mock import MockHardware
        hw = MockHardware(num_channels=16)
        hw.open()
        status = hw.leadoff_status()
        assert [c["ch"] for c in status] == list(range(1, 17))


class TestAcquisitionRestartWithConfig:
    """Test acquisition restart_with_config method."""

    def test_restart_with_config_calls_configure(self):
        import asyncio
        from pieeg_server.mock import MockHardware
        from pieeg_server.acquisition import AcquisitionLoop

        loop = asyncio.new_event_loop()
        hw = MockHardware(num_channels=8)
        hw.open()
        acq = AcquisitionLoop(hw, loop, mock=True)
        acq.start()

        reg_map = {0x05: 0x01, 0x06: 0x01}
        acq.restart_with_config(reg_map)

        assert hw.register_state.get(0x05) == 0x01
        assert hw.register_state.get(0x06) == 0x01

        acq.stop()
        loop.close()


class TestInterruptLoopRestart:
    """The interrupt acquisition loop must survive restart_with_config()."""

    class _EdgeHardware:
        """Fake SPI hardware: DRDY edges every 4 ms of kernel time."""

        num_channels = 8
        sample_rate = 250
        spike_threshold = -1

        def __init__(self):
            import threading
            self._ts = 1_000_000_000
            self._lock = threading.Lock()
            self.registers = {}

        def enable_drdy_events(self):
            pass

        def disable_drdy_events(self):
            pass

        def stop_streaming(self):
            pass

        def wait_drdy_event(self, timeout=0.5):
            import time
            time.sleep(0.004)
            with self._lock:
                self._ts += 4_000_000
                return self._ts

        def read_sample(self):
            return [0.0] * 8

        def configure_registers(self, reg_map):
            with self._lock:
                self._ts += 500_000_000   # the write pauses the edge stream
            self.registers.update(reg_map)

    def test_frames_keep_flowing_after_restart(self):
        import asyncio
        import time
        from pieeg_server.acquisition import AcquisitionLoop

        loop = asyncio.new_event_loop()
        hw = self._EdgeHardware()
        acq = AcquisitionLoop(hw, loop, interrupt=True)
        acq.start()
        time.sleep(0.2)
        acq.restart_with_config({0x05: 0x61})
        before = acq.capture_stats()["frames_read"]
        time.sleep(0.4)
        stats = acq.capture_stats()
        acq.stop()
        loop.close()

        assert acq._thread is not None and not acq._thread.is_alive()
        assert stats["frames_read"] > before, "acquisition died after restart"
        # The register-write pause is not counted as dropped samples.
        assert stats["dropped_frames"] == 0
        assert hw.registers == {0x05: 0x61}
