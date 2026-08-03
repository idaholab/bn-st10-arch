"""Focused regression checks for the indexed word MOV encodings.

These tests intentionally avoid importing Binary Ninja so they can run in
environments without a licensed headless core. The source assertions provide a
repeatable fallback when architecture-level checks are unavailable.
"""

from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[1]


def function_body(source: str, qualified_name: str) -> str:
    """Return a C++ function body using brace matching."""
    match = re.search(rf"bool\s+{re.escape(qualified_name)}\s*\([^{{]+\)\s*{{", source)
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


class MovD4RegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.lift_source = (ROOT / "src" / "lift.cpp").read_text()
        cls.text_source = (ROOT / "src" / "text.cpp").read_text()
        cls.d4_lift = function_body(cls.lift_source, "Mov::LiftxD4")
        cls.c4_lift = function_body(cls.lift_source, "Mov::LiftxC4")
        cls.d4_text = function_body(cls.text_source, "Mov::TextxD4")
        cls.c4_text = function_body(cls.text_source, "Mov::TextxC4")

    def test_fixtures_decode_destination_base_displacement_and_length(self) -> None:
        fixtures = (
            ("d4 21 34 12", 2, 1, 0x1234),
            ("d4 f0 00 80", 15, 0, 0x8000),
        )

        for encoded, destination, base, displacement in fixtures:
            with self.subTest(encoded=encoded):
                fixture = bytes.fromhex(encoded)
                self.assertEqual(fixture[0], 0xD4)
                self.assertEqual(fixture[1] >> 4, destination)
                self.assertEqual(fixture[1] & 0xF, base)
                self.assertEqual(
                    int.from_bytes(fixture[2:4], "little"), displacement
                )
                self.assertEqual(len(fixture), 4)

        self.assertIn("GetData4High(data, 2)", self.d4_lift)
        self.assertIn("GetData4Low(data, 2)", self.d4_lift)
        self.assertIn("GetData16(data, 4)", self.d4_lift)
        self.assertRegex(self.d4_lift, r"\blen\s*=\s*4\s*;")

    def test_d4_lifts_to_word_load_assigned_to_destination(self) -> None:
        expected = re.compile(
            r"il\.SetRegister\(\s*2\s*,\s*rwn\s*,\s*"
            r"il\.Load\(\s*2\s*,\s*SrcIndAddr\s*\)\s*,\s*flags\s*\)"
        )

        self.assertRegex(
            self.d4_lift,
            expected,
            "D4 must SET_REG.w(rwn, LOAD.w(source address))",
        )
        self.assertNotIn(
            "il.Store(", self.d4_lift, "D4 register destination must not emit STORE"
        )

    def test_c4_remains_a_word_store(self) -> None:
        self.assertRegex(
            self.c4_lift,
            r"il\.Store\(\s*2\s*,\s*DstIndAddr\s*,\s*"
            r"il\.Register\(\s*2\s*,\s*rwn\s*\)\s*,\s*flags\s*\)",
        )

    def test_d4_text_places_destination_before_memory_source(self) -> None:
        destination = self.d4_text.find("RegToStr(rwn)")
        source = self.d4_text.find("RegToStr(rwm)")
        self.assertGreaterEqual(destination, 0)
        self.assertGreater(source, destination)

    def test_indexed_mov_displacements_are_integer_tokens(self) -> None:
        for name, body in (("D4", self.d4_text), ("C4", self.c4_text)):
            with self.subTest(encoding=name):
                displacement = body.find('"0x%x", data16')
                self.assertGreaterEqual(displacement, 0)
                following_tokens = body[displacement:]
                self.assertRegex(
                    following_tokens,
                    r"emplace_back\(\s*IntegerToken\s*,\s*buf\s*,\s*data16\s*\)",
                )


if __name__ == "__main__":
    unittest.main()
