"""Kiểm tra tự động cho các hàm lõi của source.ipynb và các file nhãn tay."""
import io
import json
import contextlib
import unittest
from pathlib import Path

import pandas as pd

SRC = Path(__file__).resolve().parents[1]
DATA = SRC / "data"


def load_notebook_functions():
    """Chạy các ô code của notebook cho tới trước ô nạp dữ liệu (ô có COLUMN_ALIASES)."""
    cells = json.load(open(SRC / "source.ipynb", encoding="utf8"))["cells"]
    ns = {"__name__": "nb_test"}
    with contextlib.redirect_stdout(io.StringIO()):
        for c in cells:
            if c["cell_type"] != "code":
                continue
            code = "".join(c["source"])
            if "COLUMN_ALIASES = {" in code:
                break
            exec(code, ns)
    return ns


NB = load_notebook_functions()


class TestNormalize(unittest.TestCase):
    def test_slang_tone_and_stretched_letters(self):
        f = NB["normalize_for_matching"]
        self.assertEqual(f("Anh k yêuuuu em, hoà bình thuý khoẻ, quý hoàng"), "anh không yêu em hòa bình thúy khỏe quý hoàng")
        self.assertEqual(f("dzô nhìu wá ny"), "vô nhiều quá người yêu")

    def test_punctuation_and_single_letters_removed(self):
        self.assertEqual(NB["normalize_for_matching"]("Mẹ ơi!!! (điệp khúc) ... à"), "mẹ ơi điệp khúc")

    def test_keywords_already_canonical(self):
        canon = NB["canon_text"]
        bad = [kw for kws in NB["TOPIC_RULES"].values() for kw in kws if canon(kw) != kw]
        self.assertEqual(bad, [], "từ khóa chưa ở dạng chuẩn sẽ không bao giờ khớp với lời đã chuẩn hóa")

    def test_canon_is_idempotent(self):
        canon = NB["canon_text"]
        for text in ("Hoà bình", "thuỷ chung", "yêuuuuu", "Quý khách"):
            self.assertEqual(canon(canon(text)), canon(text))


class TestCleaning(unittest.TestCase):
    def test_genre_repair_longest_key_first(self):
        clean = NB["clean_single_col"]
        self.assertEqual(clean("Nh?c Tr? Tình"), "Nhạc Trữ Tình")
        self.assertEqual(clean("Nh?c Vi?t Nam"), "Nhạc Việt Nam")

    def test_unpack_list_string(self):
        self.assertEqual(NB["unpack_val"]("['Sơn Tùng', 'Đen']"), "Sơn Tùng, Đen")
        self.assertEqual(NB["unpack_val"](None), "")

    def test_lyrics_lines_cleaned_and_capitalised(self):
        out = NB["clean_lyrics_col"]("em đi rồi &amp; anh ở lại\\n\\n  ngày mai  trời lại sáng ")
        self.assertEqual(out.split("\n"), ["Em đi rồi & anh ở lại", "Ngày mai trời lại sáng"])

    def test_empty_url(self):
        self.assertEqual(NB["clean_url_col"]("khongcolink"), "Không có đường dẫn")
        self.assertEqual(NB["clean_url_col"]("['https://a.vn/x', 'https://b.vn']"), "https://a.vn/x")


class TestLabelFiles(unittest.TestCase):
    """Bẫy đã gặp: file trong data/ có cả cột tên bài lẫn lời sẽ bị bộ tự tìm dữ liệu nhận nhầm là bảng bài hát."""

    def check(self, name, column, allowed):
        path = DATA / name
        if not path.exists():
            self.skipTest(f"chưa có {name}")
        df = pd.read_csv(path)
        self.assertIn("title", df.columns)
        self.assertNotIn("lyrics", df.columns)
        self.assertFalse(df["title"].duplicated().any(), "notebook ghép nhãn theo tên bài, tên trùng sẽ ghép sai")
        self.assertTrue(df[column].isin(allowed).all(), sorted(set(df[column]) - set(allowed)))

    def test_topic_labels(self):
        for name in ("labels_gold.csv", "labels_bo_sung.csv", "labels_ngau_nhien.csv"):
            with self.subTest(name=name):
                self.check(name, "topic_meaning", NB["TOPIC_LIST"])

    def test_emotion_labels(self):
        self.check("labels_cam_xuc.csv", "emotion", ["Vui tươi", "Bình yên", "Buồn", "Mạnh mẽ"])


if __name__ == "__main__":
    unittest.main()
