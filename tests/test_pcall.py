"""Focused source-level regression checks for the C166 PCALL instruction."""

from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[1]


def function_body(source: str, qualified_name: str, return_type: str = "bool") -> str:
    """Return a C++ function body using brace matching."""
    match = re.search(
        rf"{re.escape(return_type)}\s+{re.escape(qualified_name)}\s*\([^{{]+\)\s*{{",
        source,
    )
    if match is None:
        raise AssertionError(f"could not find {qualified_name}")

    opening_brace = match.end() - 1
    depth = 0
    for index in range(opening_brace, len(source)):
        if source[index] == "{":
            depth += 1
        elif source[index] == "}":
            depth -= 1
            if depth == 0:
                return source[opening_brace + 1 : index]

    raise AssertionError(f"unterminated function body for {qualified_name}")


def target(address: int, caddr: int) -> int:
    return (address & 0xFF0000) | caddr


def translate_register(short_address: int, extr: bool = False) -> int:
    if short_address <= 0xEF:
        return (0xF000 if extr else 0xFE00) + 2 * short_address
    return short_address & 0xF


class PcallRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.architecture_source = (ROOT / "src" / "architecture.cpp").read_text()
        cls.instructions_source = (ROOT / "src" / "instructions.h").read_text()
        cls.info_source = (ROOT / "src" / "info.cpp").read_text()
        cls.lift_source = (ROOT / "src" / "lift.cpp").read_text()
        cls.text_source = (ROOT / "src" / "text.cpp").read_text()
        cls.info = function_body(cls.info_source, "Pcall::Info")
        cls.lift = function_body(cls.lift_source, "Pcall::Lift")
        cls.text = function_body(cls.text_source, "Pcall::Text")
        cls.get_target = function_body(
            (ROOT / "src" / "util.cpp").read_text(),
            "Pcall::GetTarget",
            "uint32_t",
        )

    def test_fixture_decodes_full_register_byte_and_little_endian_target(self) -> None:
        fixture = bytes.fromhex("e2 f5 34 12")
        self.assertEqual(fixture[0], 0xE2)
        self.assertEqual(fixture[1], 0xF5)
        self.assertEqual(translate_register(fixture[1]), 5)
        self.assertEqual(int.from_bytes(fixture[2:4], "little"), 0x1234)
        self.assertEqual(target(0x560100, 0x1234), 0x561234)

        self.assertIn("GetRegShortAddr(data, length)", self.lift)
        self.assertIn("GetRegShortAddr(data, length)", self.text)
        self.assertNotIn("GetData4", self.lift)
        self.assertNotIn("GetData4", self.text)

    def test_direct_register_translation_covers_gpr_sfr_and_esfr(self) -> None:
        self.assertEqual(translate_register(0xFA), 10)
        self.assertEqual(translate_register(0x80), 0xFF00)
        self.assertEqual(translate_register(0x80, extr=True), 0xF100)

        for body in (self.lift, self.text):
            self.assertIn("Instruction::TranslateReg(", body)
            self.assertIn("reg <= 0xF", body)

    def test_target_preserves_code_segment_and_handles_boundaries(self) -> None:
        fixtures = (
            (0x120100, 0x0000, 0x120000),
            (0x560100, 0x1234, 0x561234),
            (0xABFEDC, 0xFFFF, 0xABFFFF),
        )
        for address, caddr, expected in fixtures:
            with self.subTest(address=address, caddr=caddr):
                self.assertEqual(target(address, caddr), expected)

        compact = re.sub(r"\s+", "", self.get_target)
        self.assertIn("(addr&0xFF0000u)|Instruction::GetOpCaddr(data,len)", compact)

    def test_info_has_one_direct_call_and_rejects_truncated_input(self) -> None:
        self.assertRegex(self.info, r"if\s*\(maxLen\s*<\s*length\)\s*return false;")
        self.assertEqual(self.info.count("result.AddBranch("), 1)
        self.assertIn("CallDestination", self.info)
        self.assertIn("GetTarget(data, addr, length)", self.info)
        self.assertRegex(self.info, r"result\.length\s*=\s*length")

    def test_text_emits_register_then_resolved_address(self) -> None:
        self.assertIn('ITEXT("pcall")', self.text)
        register = self.text.find("RegisterToken, buf, reg")
        separator = self.text.find('OperandSeparatorToken, ", "')
        call_target = self.text.find("PossibleAddressToken, buf, target, 3")
        self.assertGreaterEqual(register, 0)
        self.assertGreater(separator, register)
        self.assertGreater(call_target, separator)
        self.assertIn("PossibleAddressToken, buf, reg", self.text)
        self.assertRegex(self.text, r"len\s*=\s*length")

    def test_lift_pushes_saved_value_then_calls_target(self) -> None:
        push = self.lift.find("il.Push(2, value, flags)")
        call = self.lift.find(
            "il.Call(il.ConstPointer(3, GetTarget(data, addr, length)))"
        )
        self.assertGreaterEqual(push, 0)
        self.assertGreater(call, push)
        self.assertIn("il.Register(2, reg)", self.lift)
        self.assertIn("il.Load(2, il.Const(3, reg))", self.lift)
        self.assertEqual(self.lift.count("il.Push("), 1)
        self.assertEqual(self.lift.count("UpdateExtSequence(addr, len)"), 1)
        pcall_declaration = re.search(
            r"class Pcall\s*{(?P<body>.*?)\n};", self.instructions_source, re.DOTALL
        )
        if pcall_declaration is None:
            self.fail("could not find Pcall declaration")
        self.assertIn("Flags::WRITE_EZN", pcall_declaration.group("body"))

    def test_related_call_push_and_return_lifts_remain_distinct(self) -> None:
        calla = function_body(self.lift_source, "Calla::Lift")
        push = function_body(self.lift_source, "Push::Lift")
        retp = function_body(self.lift_source, "Retp::Lift")

        self.assertIn("il.Call(", calla)
        self.assertNotIn("il.Push(", calla)
        self.assertIn("il.Push(", push)
        self.assertNotIn("il.Call(", push)
        self.assertIn("STACK_RETURN(length)", retp)

    def test_all_pcall_dispatch_paths_are_implemented(self) -> None:
        dispatches = re.findall(
            r"case Opcodes::PCALL:\s*return (Pcall::(?:Info|Lift|Text))\(",
            self.architecture_source,
        )
        self.assertCountEqual(dispatches, ("Pcall::Info", "Pcall::Lift", "Pcall::Text"))


if __name__ == "__main__":
    unittest.main()
