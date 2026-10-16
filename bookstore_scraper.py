"""
Assignment 2 - The Bookstore Intelligence Mission
Business Data and Text Mining | Web Scraping

Scrapes https://books.toscrape.com/ (sandbox site), builds a clean DataFrame,
saves competitor_books.csv, answers the 7 analysis questions, solves the
CEO GBP 100 challenge, and covers Bonus 1-4 + Boss Level.

Run:
    pip install requests beautifulsoup4 pandas matplotlib numpy
    python bookstore_scraper.py              # 5 pages (Bonus 1)
    python bookstore_scraper.py --pages 3    # minimum requirement
    python bookstore_scraper.py --all        # Boss Level: every catalogue page
"""

import argparse
import re
import time
from urllib.parse import urljoin

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_URL = "https://books.toscrape.com/"
PAGE_URL = BASE_URL + "catalogue/page-{}.html"   # page-1 .. page-50
HEADERS = {"User-Agent": "Mozilla/5.0 (student web-scraping assignment)"}

# Mission 2 - rating challenge: words -> numbers
RATING_MAP = {"One": 1, "Two": 2, "Three": 3, "Four": 4, "Five": 5}


# --------------------------------------------------------------------------
# MISSION 1 - RECONNAISSANCE (what I found with Inspect Element)
# --------------------------------------------------------------------------
# Every book is one <article class="product_pod"> ... </article>
#   Title        -> <h3><a title="FULL TITLE">  (the visible text is cut off
#                   with "...", so read the `title` ATTRIBUTE, not .text)
#   Price        -> <p class="price_color">£51.77</p>
#   Rating       -> <p class="star-rating Three">  (rating word is the 2nd class)
#   Availability -> <p class="instock availability"> In stock </p>
#   Product URL  -> <h3><a href="...">  (relative link, join with page URL)
#   Next page    -> <li class="next"><a href="page-2.html">


# --------------------------------------------------------------------------
# MISSION 2 - CAPTURE ONE PAGE
# --------------------------------------------------------------------------
def get_soup(url):
    """Download a page and return a BeautifulSoup object."""
    response = requests.get(url, headers=HEADERS, timeout=20)
    response.raise_for_status()
    response.encoding = "utf-8"          # avoids 'Â£' garbage in the price
    return BeautifulSoup(response.text, "html.parser")


def parse_books(soup, page_url):
    """Extract every book card on one page into a list of dicts."""
    records = []
    for card in soup.select("article.product_pod"):
        link = card.h3.a
        title = link["title"]                                   # full title
        url = urljoin(page_url, link["href"])                   # Bonus 2

        price_text = card.select_one("p.price_color").get_text(strip=True)
        price = float(re.sub(r"[^\d.]", "", price_text))        # '£51.77' -> 51.77

        rating_word = card.select_one("p.star-rating")["class"][1]
        rating = RATING_MAP[rating_word]                        # 'Three' -> 3

        availability = card.select_one("p.availability").get_text(strip=True)

        records.append(
            {
                "Title": title,
                "Price": price,
                "Rating": rating,
                "Availability": availability,
                "URL": url,
            }
        )
    return records


# --------------------------------------------------------------------------
# MISSION 3 - SCALE THE OPERATION (one reusable loop, no copy-paste)
# --------------------------------------------------------------------------
def scrape_catalogue(max_pages=3, delay=0.5):
    """Scrape pages 1..max_pages (or until there is no 'next' button)."""
    all_records = []
    page = 1
    while max_pages is None or page <= max_pages:
        url = PAGE_URL.format(page)
        print(f"Scraping page {page}: {url}")
        soup = get_soup(url)
        all_records.extend(parse_books(soup, url))

        if soup.select_one("li.next") is None:      # last page reached
            break
        page += 1
        time.sleep(delay)                           # be polite to the server
    return pd.DataFrame(all_records)


# --------------------------------------------------------------------------
# MISSION 5 - CEO CHALLENGE: maximise total stars with budget <= GBP 100
# (0/1 knapsack, solved exactly with dynamic programming, in pence)
# --------------------------------------------------------------------------
def ceo_challenge(df, budget=100.0):
    items = df[df["Availability"].str.contains("In stock", case=False)].reset_index(drop=True)
    cost = (items["Price"] * 100).round().astype(int).to_numpy()
    value = items["Rating"].to_numpy()
    cap = int(round(budget * 100))

    dp = np.zeros(cap + 1, dtype=int)                   # dp[w] = best stars with budget <= w
    keep = np.zeros((len(items), cap + 1), dtype=bool)
    for i in range(len(items)):
        c, v = cost[i], value[i]
        if c > cap:
            continue
        candidate = dp[: cap + 1 - c] + v
        better = candidate > dp[c:]
        keep[i, c:] = better
        dp[c:] = np.where(better, candidate, dp[c:])

    best = dp[cap]
    w = int(np.argmax(dp == best))                      # cheapest budget reaching the best score
    chosen = []
    for i in range(len(items) - 1, -1, -1):
        if keep[i, w]:
            chosen.append(i)
            w -= cost[i]
    picked = items.loc[sorted(chosen), ["Title", "Price", "Rating"]].reset_index(drop=True)
    return picked


# --------------------------------------------------------------------------
# MAIN
# --------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pages", type=int, default=5, help="pages to scrape (default 5)")
    parser.add_argument("--all", action="store_true", help="Boss level: scrape all pages")
    args = parser.parse_args()

    df = scrape_catalogue(None if args.all else args.pages)

    # ---- Clean + check ----
    df = df.drop_duplicates(subset="URL").reset_index(drop=True)
    df["Title"] = df["Title"].str.strip()
    print("\nFirst rows:\n", df.head())
    print("\nColumns / dtypes:\n", df.dtypes)
    print("\nShape:", df.shape)

    # ---- Save CSV (assignment needs Title, Price, Rating, Availability; URL = Bonus 2) ----
    df.to_csv("competitor_books.csv", index=False, encoding="utf-8")
    print("\nSaved competitor_books.csv")

    # ---- MISSION 4: Intelligence analysis ----
    most_exp = df.loc[df["Price"].idxmax()]
    cheapest = df.loc[df["Price"].idxmin()]
    avg_price = df["Price"].mean()
    five_star = df[df["Rating"] == 5]
    in_stock_pct = df["Availability"].str.contains("In stock", case=False).mean() * 100
    top5 = df.sort_values(["Rating", "Price"], ascending=[False, True]).head(5)
    cheapest_5 = five_star.loc[five_star["Price"].idxmin()]

    print("\n================ ANALYSIS ================")
    print(f"1. Most expensive : {most_exp.Title} (£{most_exp.Price:.2f})")
    print(f"2. Cheapest       : {cheapest.Title} (£{cheapest.Price:.2f})")
    print(f"3. Average price  : £{avg_price:.2f}")
    print(f"4. 5-star books   : {len(five_star)}")
    print(f"5. In stock       : {in_stock_pct:.1f}%")
    print("6. Top 5 highest-rated (ties broken by lower price):")
    print(top5[["Title", "Price", "Rating"]].to_string(index=False))
    print(f"7. Cheapest 5-star: {cheapest_5.Title} (£{cheapest_5.Price:.2f})")

    # ---- CEO challenge ----
    picked = ceo_challenge(df)
    total_cost, total_stars = picked["Price"].sum(), picked["Rating"].sum()
    print("\n============ CEO £100 CHALLENGE ============")
    print(picked.to_string(index=False))
    print(f"Books: {len(picked)} | Total cost: £{total_cost:.2f} | Total rating score: {total_stars}")

    # ---- Bonus 3 + 4: rating counts + bar chart ----
    rating_counts = df["Rating"].value_counts().sort_index()
    print("\nBooks per star level:\n", rating_counts)
    ax = rating_counts.plot(kind="bar", color="#2a6f97", edgecolor="black")
    ax.set_title("Number of Books by Star Rating")
    ax.set_xlabel("Star rating")
    ax.set_ylabel("Number of books")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig("rating_counts.png", dpi=150)
    print("Saved rating_counts.png")

    # ---- Business decision: evidence for the recommendation ----
    avg_5 = five_star["Price"].mean()
    share_5 = len(five_star) / len(df) * 100
    cheap_5_count = (five_star["Price"] < avg_price).sum()
    print("\n=========== RECOMMENDATION EVIDENCE ===========")
    print(f"Evidence 1: {share_5:.1f}% of books are 5-star; their average price is "
          f"£{avg_5:.2f} vs £{avg_price:.2f} overall.")
    print(f"Evidence 2: {cheap_5_count} of {len(five_star)} five-star books cost less than the "
          f"average price (cheapest £{cheapest_5.Price:.2f}).")
    print(f"Evidence 3: {in_stock_pct:.0f}% of books are in stock, so they can actually be sold.")

    # Save everything to a text file for the submission
    with open("answers.txt", "w", encoding="utf-8") as f:
        f.write(f"Pages scraped / books: {len(df)}\n\n")
        f.write(f"1. Most expensive: {most_exp.Title} (£{most_exp.Price:.2f})\n")
        f.write(f"2. Cheapest: {cheapest.Title} (£{cheapest.Price:.2f})\n")
        f.write(f"3. Average price: £{avg_price:.2f}\n")
        f.write(f"4. 5-star books: {len(five_star)}\n")
        f.write(f"5. In stock: {in_stock_pct:.1f}%\n")
        f.write("6. Top 5 highest-rated:\n" + top5[["Title", "Price", "Rating"]].to_string(index=False) + "\n")
        f.write(f"7. Cheapest 5-star: {cheapest_5.Title} (£{cheapest_5.Price:.2f})\n\n")
        f.write("CEO £100 challenge:\n" + picked.to_string(index=False) + "\n")
        f.write(f"Total cost: £{total_cost:.2f} | Total rating score: {total_stars}\n\n")
        f.write("Books per star level:\n" + rating_counts.to_string() + "\n\n")
        f.write(f"Evidence 1: {share_5:.1f}% are 5-star, avg £{avg_5:.2f} vs £{avg_price:.2f} overall.\n")
        f.write(f"Evidence 2: {cheap_5_count} of {len(five_star)} five-star books are below average price.\n")
    print("Saved answers.txt")


if __name__ == "__main__":
    main()
