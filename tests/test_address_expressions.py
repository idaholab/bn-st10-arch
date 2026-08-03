"""Source-level regression checks for C166 indirect address expressions."""

from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[1]


def function_body(source: str, qualified_name: str) -> str:
    """Return a C++ function body using brace matching."""
    match = re.search(
        rf"(?:bool|BN::ExprId)\s+{re.escape(qualified_name)}\s*\([^{{]+\)\s*{{",
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


def extp_address(page: int, register: int, displacement: int = 0) -> int:
    long_address = (register + displacement) & 0xFFFF
    return (page << 14) | (long_address & 0x3FFF)


class AddressExpressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.util_source = (ROOT / "src" / "util.cpp").read_text()
        cls.lift_source = (ROOT / "src" / "lift.cpp").read_text()
        cls.extp = function_body(
            cls.util_source, "Instruction::GetIndAddrExpr_Extp_Rw"
        )
        cls.extp_displacement = function_body(
            cls.util_source, "Instruction::GetIndAddrExpr_Extp_Rw_data16"
        )
        cls.exts_displacement = function_body(
            cls.util_source, "Instruction::GetIndAddrExpr_Exts_Rw_data16"
        )
        cls.dpp_displacement = function_body(
            cls.util_source, "Instruction::GetIndAddrExpr_Rw_data16"
        )
        cls.d4_lift = function_body(cls.lift_source, "Mov::LiftxD4")

    def test_extp_displacement_boundary_values(self) -> None:
        fixtures = (
            (0x012, 0x0000, 0x1234, 0x1234, 0x1234),
            (0x012, 0x3FF0, 0x0030, 0x4020, 0x0020),
            (0x012, 0x7FF0, 0x0030, 0x8020, 0x0020),
            (0x012, 0xFFF0, 0x0030, 0x0020, 0x0020),
        )

        for page, register, displacement, long_address, offset in fixtures:
            with self.subTest(register=register):
                self.assertEqual((register + displacement) & 0xFFFF, long_address)
                physical = extp_address(page, register, displacement)
                self.assertEqual(physical, (page << 14) | offset)
                self.assertEqual(physical >> 14, page)
                self.assertEqual(physical & 0x3FFF, offset)

    def test_extp_displacement_masks_wrapped_long_address_to_page(self) -> None:
        compact = re.sub(r"\s+", "", self.extp_displacement)
        self.assertIn(
            "LongAddr=il.And(2,il.Add(2,il.Register(2,Rw),"
            "il.Const(2,data16)),il.Const(2,0xFFFF))",
            compact,
        )
        self.assertIn(
            "PageOffset=il.And(2,LongAddr,il.Const(2,0x3FFF))", compact
        )
        self.assertIn("il.Or(3,IndAddrPag10,PageOffset)", compact)

    def test_exts_retains_full_segment_offset(self) -> None:
        compact = re.sub(r"\s+", "", self.exts_displacement)
        self.assertIn("il.Const(2,0xFFFF)", compact)
        self.assertNotIn("0x3FFF", compact)
        self.assertEqual(((0x7FF0 + 0x0030) & 0xFFFF), 0x8020)

    def test_dpp_uses_long_address_for_selection_and_14_bit_offset(self) -> None:
        compact = re.sub(r"\s+", "", self.dpp_displacement)
        self.assertIn("il.Const(2,0xC000),Ind", compact)
        self.assertIn("il.Const(2,14)", compact)
        self.assertIn("il.And(2,Ind,il.Const(2,0x3FFF))", compact)

        long_address = (0x7FF0 + 0x0030) & 0xFFFF
        self.assertEqual(long_address >> 14, 2)
        self.assertEqual(long_address & 0x3FFF, 0x0020)

    def test_extp_helpers_share_the_same_14_bit_offset_rule(self) -> None:
        compact = re.sub(r"\s+", "", self.extp)
        self.assertIn(
            "il.And(2,il.Register(2,Rw),il.Const(2,0x3FFF))", compact
        )
        self.assertEqual(
            extp_address(0x012, 0x8020),
            extp_address(0x012, 0x7FF0, 0x0030),
        )

    def test_d4_routes_extp_displacement_through_corrected_helper(self) -> None:
        self.assertIn("GetIndAddrExpr_Extp_Rw_data16", self.d4_lift)


if __name__ == "__main__":
    unittest.main()
