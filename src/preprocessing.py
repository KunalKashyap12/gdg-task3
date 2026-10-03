"""
Preprocessing module for News Classification System.
Includes both basic text normalization and advanced NLP preprocessing
(HTML stripping, contraction expansion, lemmatization, stopword removal).
"""

import os
import re
import html
from typing import List, Union, Optional
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer

from src.config import CONTRACTIONS, NEWS_AGENCY_PATTERNS

# Ensure writable temporary directory is in nltk path for serverless runtimes (Vercel, AWS Lambda)
for _dir in ["/tmp/nltk_data", os.path.expanduser("~/nltk_data")]:
    if _dir not in nltk.data.path:
        nltk.data.path.append(_dir)


class TextPreprocessor:
    """
    Modular text preprocessor providing both basic and advanced text cleaning.
    """

    def __init__(
        self,
        use_lemmatization: bool = True,
        remove_stopwords: bool = True,
        expand_contractions: bool = True,
        title_weight: int = 1,
        custom_stopwords: Optional[List[str]] = None,
    ):
        self.use_lemmatization = use_lemmatization
        self.remove_stopwords = remove_stopwords
        self.expand_contractions = expand_contractions
        self.title_weight = title_weight

        # Compile regex patterns for performance
        self._html_tag_re = re.compile(r"<[^>]+>")
        self._url_re = re.compile(r"https?://\S+|www\.\S+")
        self._email_re = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b")
        self._non_ascii_re = re.compile(r"[^\x00-\x7F]+")
        self._special_char_re = re.compile(r"[^a-zA-Z0-9\s]")
        self._whitespace_re = re.compile(r"\s+")
        self._agency_res = [re.compile(p, re.IGNORECASE) for p in NEWS_AGENCY_PATTERNS]

        # Tokenizer: alphanumeric words of length >= 2
        self.tokenizer = RegexpTokenizer(r"\b[a-zA-Z]{2,}\b")

        # Determine writable download directory if on serverless
        _download_dir = "/tmp/nltk_data" if os.path.isdir("/tmp") and os.access("/tmp", os.W_OK) else None

        # Initialize Lemmatizer and memoization cache
        self.lemmatizer = None
        self._lemma_cache = {}
        if use_lemmatization:
            try:
                self.lemmatizer = WordNetLemmatizer()
                self.lemmatizer.lemmatize("testing")
            except (LookupError, Exception):
                try:
                    nltk.download("wordnet", download_dir=_download_dir, quiet=True)
                    nltk.download("omw-1.4", download_dir=_download_dir, quiet=True)
                    self.lemmatizer = WordNetLemmatizer()
                except Exception:
                    self.lemmatizer = None

        # Build Stopwords Set
        self._stopwords = set()
        if remove_stopwords:
            try:
                base_stops = set(stopwords.words("english"))
            except (LookupError, Exception):
                try:
                    nltk.download("stopwords", download_dir=_download_dir, quiet=True)
                    base_stops = set(stopwords.words("english"))
                except Exception:
                    base_stops = {
                        "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
                        "any", "are", "as", "at", "be", "because", "been", "before", "being", "below",
                        "between", "both", "but", "by", "can", "cannot", "could", "did", "do", "does",
                        "doing", "down", "during", "each", "few", "for", "from", "further", "had",
                        "has", "have", "having", "he", "her", "here", "hers", "herself", "him", "himself",
                        "his", "how", "if", "in", "into", "is", "it", "its", "itself", "just", "me",
                        "more", "most", "my", "myself", "no", "nor", "not", "now", "of", "off", "on",
                        "once", "only", "or", "other", "our", "ours", "ourselves", "out", "over", "own",
                        "same", "she", "should", "so", "some", "such", "than", "that", "the", "their",
                        "theirs", "them", "themselves", "then", "there", "these", "they", "this",
                        "those", "through", "to", "too", "under", "until", "up", "very", "was", "we",
                        "were", "what", "when", "where", "which", "while", "who", "whom", "why", "with",
                        "would", "you", "your", "yours", "yourself", "yourselves"
                    }

            # Domain specific news stopwords (boilerplates, agency tags)
            domain_stops = {
                "reuters", "ap", "afp", "bloomberg", "said", "say", "says",
                "told", "year", "month", "day", "week", "today", "yesterday",
                "inc", "corp", "co", "ltd", "com", "href", "fullquote", "news"
            }
            self._stopwords = base_stops | domain_stops
            if custom_stopwords:
                self._stopwords.update([w.lower() for w in custom_stopwords])

    def clean_basic(self, text: str) -> str:
        """
        Perform basic text cleaning:
        - HTML entity unescaping and malformed tag cleanup
        - Removing HTML tags, URLs, emails
        - Backslash escape cleanup
        - News agency boilerplate removal
        - Contraction expansion
        - Lowercasing and whitespace normalization
        """
        if not isinstance(text, str):
            return ""

        # 1. Unescape HTML entities & clean malformed entities like #39; -> '
        text = text.replace("#39;", "'").replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
        text = html.unescape(text)

        # 2. Fix backslash artifacts (e.g. "early\and" -> "early and")
        text = text.replace("\\", " ")

        # 3. Strip HTML tags
        text = self._html_tag_re.sub(" ", text)

        # 4. Strip URLs and Emails
        text = self._url_re.sub(" ", text)
        text = self._email_re.sub(" ", text)

        # 5. Remove news wire agency prefixes / suffixes
        for agency_re in self._agency_res:
            text = agency_re.sub(" ", text)

        # 6. Lowercase
        text = text.lower()

        # 7. Contraction expansion
        if self.expand_contractions:
            for cont, expansion in CONTRACTIONS.items():
                text = text.replace(cont, expansion)

        # 8. Normalize special characters & punctuation to space
        text = self._special_char_re.sub(" ", text)

        # 9. Collapse multiple spaces
        text = self._whitespace_re.sub(" ", text).strip()

        return text

    def clean_advanced(self, text: str) -> str:
        """
        Perform advanced text preprocessing:
        - Basic cleaning
        - Tokenization
        - Stopwords removal
        - WordNet lemmatization
        """
        # Run basic cleaning first
        cleaned = self.clean_basic(text)

        if not self.use_lemmatization and not self.remove_stopwords:
            return cleaned

        tokens = self.tokenizer.tokenize(cleaned)

        processed_tokens = []
        for token in tokens:
            if self.remove_stopwords and token in self._stopwords:
                continue
            if self.use_lemmatization and self.lemmatizer:
                if token not in self._lemma_cache:
                    lemma = self.lemmatizer.lemmatize(token, pos="n")
                    lemma = self.lemmatizer.lemmatize(lemma, pos="v")
                    self._lemma_cache[token] = lemma
                token = self._lemma_cache[token]
            if len(token) > 1:
                processed_tokens.append(token)

        return " ".join(processed_tokens)

    def combine_title_description(
        self, title: Union[str, pd.Series], description: Union[str, pd.Series]
    ) -> Union[str, pd.Series]:
        """
        Intelligently combine Title and Description.
        Title often carries dense categorical signals, so title_weight controls
        how many times the title is emphasized.
        """
        if isinstance(title, str) and isinstance(description, str):
            weighted_title = " ".join([title] * self.title_weight)
            return f"{weighted_title}. {description}".strip()

        # Pandas Series handling
        weighted_title = title.astype(str)
        for _ in range(self.title_weight - 1):
            weighted_title = weighted_title + " " + title.astype(str)
        return weighted_title + ". " + description.astype(str)

    def process_dataframe(
        self, df: pd.DataFrame, advanced: bool = True
    ) -> pd.Series:
        """
        Process a pandas DataFrame with 'Title' and 'Description' columns.
        Returns a Series of preprocessed text.
        """
        combined = self.combine_title_description(df["Title"], df["Description"])
        clean_fn = self.clean_advanced if advanced else self.clean_basic
        return combined.apply(clean_fn)
