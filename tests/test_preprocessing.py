"""
Unit tests for text preprocessing functions and TextPreprocessor class.
"""

import unittest
from src.preprocessing import TextPreprocessor


class TestTextPreprocessing(unittest.TestCase):

    def setUp(self):
        self.preprocessor = TextPreprocessor(use_lemmatization=False, remove_stopwords=False)

    def test_clean_html_entities(self):
        raw = "Wall Street&#39;s rally &amp; Tech &lt;b&gt;boom&lt;/b&gt;"
        cleaned = self.preprocessor.clean_basic(raw)
        self.assertNotIn("&#39;", cleaned)
        self.assertNotIn("&amp;", cleaned)
        self.assertNotIn("<b>", cleaned)
        self.assertNotIn("</b>", cleaned)
        self.assertIn("wall street", cleaned)

    def test_strip_news_prefixes(self):
        raw = "Reuters - Government announces new healthcare reform initiative."
        cleaned = self.preprocessor.clean_basic(raw)
        self.assertFalse(cleaned.startswith("reuters"))
        self.assertIn("government announces", cleaned)

    def test_remove_urls(self):
        raw = "Read full coverage at https://news.example.com/breaking for details."
        cleaned = self.preprocessor.clean_basic(raw)
        self.assertNotIn("https", cleaned)
        self.assertNotIn("news.example.com", cleaned)

    def test_empty_and_none_input(self):
        self.assertEqual(self.preprocessor.clean_basic(""), "")
        self.assertEqual(self.preprocessor.clean_basic(None), "")

    def test_combine_title_description(self):
        combined = self.preprocessor.combine_title_description("Title", "Description")
        self.assertEqual(combined, "Title. Description")


if __name__ == "__main__":
    unittest.main()
