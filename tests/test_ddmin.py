import unittest

from reproreduce.reduce.ddmin import ddmin


class DdminTests(unittest.TestCase):
    def test_removes_irrelevant_items(self):
        result = ddmin(list("abcdefgh"), lambda items: "d" in items and "g" in items)
        self.assertEqual(set(result), {"d", "g"})

    def test_required_single_item(self):
        self.assertEqual(ddmin(["only"], lambda items: items == ["only"]), ["only"])


if __name__ == "__main__":
    unittest.main()
