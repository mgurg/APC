import unittest
from apc import apc

# Usage:
# python -m unittest test.test_apc


class TestAPC(unittest.TestCase):
    def test_load_video_frames(self):
        with self.assertRaises(FileNotFoundError):
            apc.load_video('noexisting.file')

if __name__ == '__main__':
    unittest.main()