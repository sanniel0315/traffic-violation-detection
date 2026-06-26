"""
Driver for ICP DAS tM-PD3R3 — 3-channel dry-contact DI + 3-channel relay output,
RS-485 Modbus RTU.

Hardware reference: docs at icpdas.com/tw/product/tM-PD3R3
Register map per "Quick Start Guide for tM-P3R3" v1.0.1, section 8.

Wiring on AFE-R750 COM1 (DB9, RS-485 mode, COM1_SW1 = 1011 ON/ON/ON/OFF):
    DB9 pin 1 (DATA-) ── tM-PD3R3 DATA-
    DB9 pin 2 (DATA+) ── tM-PD3R3 DATA+
    DB9 pin 5 (GND)   ── tM-PD3R3 GND
    +Vs (10~30 VDC)   ── external supply
"""
from __future__ import annotations

import struct
import time
from contextlib import contextmanager

import serial


NUM_CHANNELS = 3  # 3 DI + 3 DO

# Modbus 5-digit register numbers from the manual.
# Wire offset = number - 1 (within its function-code address space).
_DO_COIL_BASE = 1       # 00001..00003 → offsets 0..2  (FC 01/05/0F)
_DI_INPUT_BASE = 10033  # 10033..10035 → offsets 32..34 (FC 02)
_COUNTER_REG_BASE = 30001  # 30001..30003 → offsets 0..2 (FC 04)
_REG_MODULE_ADDR = 40485   # offset 484 (FC 03/06)
_REG_BAUD_PARITY = 40486   # offset 485 (FC 03/06)
_REG_FW_LOW = 40481
_REG_NAME_LOW = 40483

# Baud-rate code per manual reg 40486 bits 5:0
_BAUD_CODES = {
    1200: 3, 2400: 4, 4800: 5, 9600: 6,
    19200: 7, 38400: 8, 57600: 9, 115200: 10,
}
# Parity code per manual reg 40486 bits 7:6
_PARITY_CODES = {
    "N1": 0b00,  # no parity, 1 stop bit
    "N2": 0b01,  # no parity, 2 stop bits
    "E1": 0b10,  # even parity, 1 stop bit
    "O1": 0b11,  # odd parity, 1 stop bit
}


class ModbusError(IOError):
    """Raised when the device returns a Modbus exception or malformed reply."""


def _crc16(data: bytes) -> bytes:
    crc = 0xFFFF
    for b in data:
        crc ^= b
        for _ in range(8):
            crc = (crc >> 1) ^ 0xA001 if crc & 1 else crc >> 1
    return crc.to_bytes(2, "little")


class PD3R3:
    def __init__(
        self,
        port: str = "/dev/ttyTHS1",
        address: int = 1,
        baudrate: int = 9600,
        parity: str = "N",
        stopbits: int = 1,
        timeout: float = 0.3,
        inter_frame_pause: float = 0.020,
    ):
        if not 1 <= address <= 247:
            raise ValueError(f"address {address} out of Modbus range 1..247")
        self.address = address
        self.inter_frame_pause = inter_frame_pause
        self._ser = serial.Serial(
            port, baudrate,
            bytesize=8, parity=parity, stopbits=stopbits,
            timeout=timeout, write_timeout=1.0,
        )
        # Try to enable RS-485 auto-direction via RTS (works when COM1_SW1 is in
        # "Software flow control" RS-485 mode). Harmless if hardware-auto.
        try:
            from serial.rs485 import RS485Settings
            self._ser.rs485_mode = RS485Settings(
                rts_level_for_tx=True, rts_level_for_rx=False,
                delay_before_tx=0.0, delay_before_rx=0.001,
            )
        except (ImportError, NotImplementedError, OSError, ValueError):
            pass

    # ── lifecycle ────────────────────────────────────────────────────────
    def close(self) -> None:
        self._ser.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # ── low-level Modbus RTU framing ─────────────────────────────────────
    def _txrx(self, fc: int, payload: bytes, expected_len: int, addr: int | None = None) -> bytes:
        a = self.address if addr is None else addr
        frame = bytes([a, fc]) + payload
        frame += _crc16(frame)
        self._ser.reset_input_buffer()
        self._ser.write(frame)
        # 3.5-char silence at 9600 ≈ 4ms; pyserial timeout handles read
        resp = self._ser.read(expected_len)
        if len(resp) < 3:
            raise ModbusError(f"timeout, got {resp!r}")
        if resp[0] != a:
            raise ModbusError(f"wrong slave addr in reply: {resp[0]}")
        if resp[1] & 0x80:
            # exception frame: addr fc|0x80 ex_code crc_lo crc_hi
            tail = resp + self._ser.read(5 - len(resp))
            ex = tail[2] if len(tail) >= 3 else 0
            raise ModbusError(f"Modbus exception fc=0x{fc:02x} code=0x{ex:02x}")
        if len(resp) < expected_len:
            resp += self._ser.read(expected_len - len(resp))
        if _crc16(resp[:-2]) != resp[-2:]:
            raise ModbusError(f"CRC mismatch: {resp.hex()}")
        time.sleep(self.inter_frame_pause)
        return resp[2:-2]  # strip addr/fc and CRC

    def _read_bits(self, fc: int, base_number: int, count: int) -> list[bool]:
        offset = base_number - (1 if fc == 1 else 10001)
        payload = offset.to_bytes(2, "big") + count.to_bytes(2, "big")
        # response: byte_count + ceil(count/8) bytes + CRC
        byte_count = (count + 7) // 8
        body = self._txrx(fc, payload, expected_len=3 + 1 + byte_count + 2)
        # body = [byte_count, b0, b1, ...]
        bits_byte_count = body[0]
        if bits_byte_count != byte_count:
            raise ModbusError(f"unexpected bit byte count {bits_byte_count}")
        bits = []
        for i in range(count):
            bits.append(bool((body[1 + i // 8] >> (i % 8)) & 1))
        return bits

    def _read_regs(self, fc: int, base_number: int, count: int) -> list[int]:
        if fc == 3:
            offset = base_number - 40001
        elif fc == 4:
            offset = base_number - 30001
        else:
            raise ValueError(f"bad fc {fc}")
        payload = offset.to_bytes(2, "big") + count.to_bytes(2, "big")
        body = self._txrx(fc, payload, expected_len=3 + 1 + 2 * count + 2)
        nbytes = body[0]
        if nbytes != 2 * count:
            raise ModbusError(f"unexpected reg byte count {nbytes}")
        return list(struct.unpack(">" + "H" * count, body[1:1 + nbytes]))

    def read_holding_at(self, slave_addr: int, start: int, count: int) -> list[int]:
        """FC03 讀「指定從機位址」的 holding register，start 為 raw 寄存器位址（不做 40001 偏移）。
        供同一條 RS-485 上的其他 Modbus 設備使用（如 E-1507 電子鎖，狀態寄存器 0x0020 起）。
        共用本連線的 serial port，呼叫端需自行用 IOModule._lock 序列化避免 bus 並發。"""
        payload = start.to_bytes(2, "big") + count.to_bytes(2, "big")
        body = self._txrx(3, payload, expected_len=3 + 1 + 2 * count + 2, addr=slave_addr)
        nbytes = body[0]
        if nbytes != 2 * count:
            raise ModbusError(f"unexpected reg byte count {nbytes}")
        return list(struct.unpack(">" + "H" * count, body[1:1 + nbytes]))

    def write_holding_at(self, slave_addr: int, reg: int, value: int) -> None:
        """FC06 寫單個 holding register 到「指定從機位址」(raw 位址)。
        給 E-1507 電子鎖設定用(如改 485 位址 reg 0x2000、485 開鎖 reg 0x2004)。
        正常應答 echo 整個請求(addr fc reg_hi reg_lo val_hi val_lo crc)。"""
        payload = reg.to_bytes(2, "big") + value.to_bytes(2, "big")
        # echo 應答 = 1addr+1fc+2reg+2val+2crc = 8 bytes
        self._txrx(6, payload, expected_len=8, addr=slave_addr)

    def _write_lenient(self, fc: int, payload: bytes) -> None:
        """Modbus write 對手上這顆 PD3R3 的 echo 第一個 byte 常被讀成 0
        (RS-485 line floating / converter timing)，但實際 hardware action 已生效。
        改成寫完後 drain echo + 短 wait，不要 raise wrong-addr，避免 caller error。"""
        frame = bytes([self.address, fc]) + payload
        frame += _crc16(frame)
        self._ser.reset_input_buffer()
        self._ser.write(frame)
        # 等 echo 完全傳完（worst case 8 bytes @ 9600 ≈ 9 ms）+ inter-frame
        time.sleep(0.020)
        # drain echo bytes, don't validate
        try:
            self._ser.read(16)
        except Exception:
            pass
        time.sleep(self.inter_frame_pause)

    def _write_coil(self, number: int, value: bool) -> None:
        offset = number - 1
        payload = offset.to_bytes(2, "big") + (b"\xff\x00" if value else b"\x00\x00")
        self._write_lenient(0x05, payload)

    def _write_coils(self, number: int, values: list[bool]) -> None:
        offset = number - 1
        n = len(values)
        byte_count = (n + 7) // 8
        bits = bytearray(byte_count)
        for i, v in enumerate(values):
            if v:
                bits[i // 8] |= 1 << (i % 8)
        payload = (
            offset.to_bytes(2, "big")
            + n.to_bytes(2, "big")
            + bytes([byte_count])
            + bytes(bits)
        )
        self._write_lenient(0x0F, payload)

    def _write_register(self, number: int, value: int) -> None:
        offset = number - 40001
        payload = offset.to_bytes(2, "big") + value.to_bytes(2, "big")
        self._write_lenient(0x06, payload)

    # ── high-level API ──────────────────────────────────────────────────
    def read_inputs(self) -> list[bool]:
        """Return state of DI0..DI2 (True = active / closed contact)."""
        return self._read_bits(0x02, _DI_INPUT_BASE, NUM_CHANNELS)

    def read_outputs(self) -> list[bool]:
        """Return current relay state for ch0..ch2 (True = energised / closed)."""
        return self._read_bits(0x01, _DO_COIL_BASE, NUM_CHANNELS)

    def set_relay(self, channel: int, on: bool) -> None:
        """Switch a single relay. channel is 0..2."""
        if not 0 <= channel < NUM_CHANNELS:
            raise ValueError(f"channel {channel} out of range 0..{NUM_CHANNELS-1}")
        self._write_coil(_DO_COIL_BASE + channel, on)

    def set_relays(self, states: list[bool]) -> None:
        """Set all 3 relays in one frame. Pass a list of 3 booleans."""
        if len(states) != NUM_CHANNELS:
            raise ValueError(f"need {NUM_CHANNELS} states, got {len(states)}")
        self._write_coils(_DO_COIL_BASE, states)

    def all_off(self) -> None:
        self.set_relays([False] * NUM_CHANNELS)

    def all_on(self) -> None:
        self.set_relays([True] * NUM_CHANNELS)

    def read_counter(self, channel: int) -> int:
        """Read the 16-bit pulse counter on DI channel 0..2."""
        if not 0 <= channel < NUM_CHANNELS:
            raise ValueError(f"channel {channel} out of range")
        return self._read_regs(0x04, _COUNTER_REG_BASE + channel, 1)[0]

    def read_all_counters(self) -> list[int]:
        return self._read_regs(0x04, _COUNTER_REG_BASE, NUM_CHANNELS)

    def clear_counter(self, channel: int) -> None:
        """Reset a DI channel's pulse counter to 0."""
        if not 0 <= channel < NUM_CHANNELS:
            raise ValueError(f"channel {channel} out of range")
        # coil 00513..00515 = clear ch0..ch2 (write 1)
        self._write_coil(513 + channel, True)

    # ── identity / config ───────────────────────────────────────────────
    def get_module_address(self) -> int:
        return self._read_regs(0x03, _REG_MODULE_ADDR, 1)[0]

    def get_firmware_version(self) -> str:
        lo, hi = self._read_regs(0x03, _REG_FW_LOW, 2)
        # version stored as packed BCD/ASCII; show as hex for now
        return f"{hi:04X}{lo:04X}"

    def get_module_name(self) -> str:
        lo, hi = self._read_regs(0x03, _REG_NAME_LOW, 2)
        # name is ASCII packed: e.g. 'tMPD3R3' → returned as 4 bytes
        raw = struct.pack(">HH", hi, lo)
        return raw.rstrip(b"\x00").decode("ascii", errors="replace")

    def set_module_address(self, new_addr: int) -> None:
        """Change module slave address (1..247). Takes effect immediately."""
        if not 1 <= new_addr <= 247:
            raise ValueError("address must be 1..247")
        self._write_register(_REG_MODULE_ADDR, new_addr)
        self.address = new_addr

    def set_baud_and_parity(self, baudrate: int, parity_stop: str = "N1") -> None:
        """Reprogram baud + parity. parity_stop in {'N1','N2','E1','O1'}.
        After this, you must close() and reopen with the new baudrate."""
        if baudrate not in _BAUD_CODES:
            raise ValueError(f"baud {baudrate} not in {sorted(_BAUD_CODES)}")
        if parity_stop not in _PARITY_CODES:
            raise ValueError(f"parity_stop must be one of {list(_PARITY_CODES)}")
        value = (_PARITY_CODES[parity_stop] << 6) | _BAUD_CODES[baudrate]
        self._write_register(_REG_BAUD_PARITY, value)


# ── tiny demo / smoke test ────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse, sys
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="/dev/ttyTHS1")
    ap.add_argument("--addr", type=int, default=1)
    ap.add_argument("--baud", type=int, default=9600)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("info")
    sub.add_parser("status")
    p_on = sub.add_parser("on");  p_on.add_argument("ch", type=int)
    p_of = sub.add_parser("off"); p_of.add_argument("ch", type=int)
    sub.add_parser("all-on")
    sub.add_parser("all-off")
    p_blink = sub.add_parser("blink")
    p_blink.add_argument("ch", type=int)
    p_blink.add_argument("--count", type=int, default=5)
    p_blink.add_argument("--period", type=float, default=0.5)

    args = ap.parse_args()

    with PD3R3(args.port, args.addr, args.baud) as dev:
        if args.cmd == "info":
            print(f"address  = {dev.get_module_address()}")
            print(f"firmware = {dev.get_firmware_version()}")
            print(f"name     = {dev.get_module_name()}")
        elif args.cmd == "status":
            print(f"DI = {dev.read_inputs()}")
            print(f"DO = {dev.read_outputs()}")
            print(f"counters = {dev.read_all_counters()}")
        elif args.cmd == "on":
            dev.set_relay(args.ch, True)
        elif args.cmd == "off":
            dev.set_relay(args.ch, False)
        elif args.cmd == "all-on":
            dev.all_on()
        elif args.cmd == "all-off":
            dev.all_off()
        elif args.cmd == "blink":
            for _ in range(args.count):
                dev.set_relay(args.ch, True)
                time.sleep(args.period)
                dev.set_relay(args.ch, False)
                time.sleep(args.period)
