import unittest

from src.phase1.crop import normalized_to_pixel_box, parse_crop_box


class CropParsingTests(unittest.TestCase):
    def test_parses_official_box_format(self):
        result = parse_crop_box("<box>(100,200),(900,800)</box>")
        self.assertEqual(result.status, "valid")
        self.assertEqual(result.normalized_box, (100.0, 200.0, 900.0, 800.0))

    def test_accepts_spacing_and_decimals(self):
        result = parse_crop_box("crop: ( 10.5, 20 ), (900, 999.5)")
        self.assertEqual(result.status, "valid")

    def test_preserves_parse_failures(self):
        self.assertEqual(parse_crop_box("no box").status, "no_coordinates")
        self.assertEqual(parse_crop_box("(1,2),(3,4),(5,6)").status, "wrong_coordinate_count")
        self.assertEqual(parse_crop_box("(-1,2),(3,4)").status, "out_of_range")
        self.assertEqual(parse_crop_box("(500,500),(400,600)").status, "non_positive_area")

    def test_scales_normalized_box_for_rendering(self):
        self.assertEqual(
            normalized_to_pixel_box((100.0, 200.0, 900.0, 800.0), 2000, 1000),
            (200, 200, 1800, 800),
        )


if __name__ == "__main__":
    unittest.main()
