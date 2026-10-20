"""Regenerate src/nlp/data/common_words.txt (needs: pip install wordfreq).

Keeps 2-5 letter alphabetic words from the 30,000 most frequent English words.
These are the tickers that `drop_ambiguous` treats as possibly just words.
"""
from pathlib import Path

from wordfreq import top_n_list

OUT = Path(__file__).parent / "data" / "common_words.txt"


def main(n: int = 30_000):
    words = sorted({w.upper() for w in top_n_list("en", n)
                    if w.isascii() and w.isalpha() and 2 <= len(w) <= 5})
    OUT.write_text(
        f"# 2-5 letter words from the top {n:,} English words (wordfreq 'en', top_n_list).\n"
        "# Regenerate with: python -m src.nlp.build_word_list\n" + "\n".join(words) + "\n"
    )
    print(f"{len(words):,} words -> {OUT}")


if __name__ == "__main__":
    main()
