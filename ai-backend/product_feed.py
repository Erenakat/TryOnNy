import csv
from pathlib import Path


DEFAULT_FEED_CANDIDATES = ("product_feed.csv", "product_feed.example.csv")


def _to_int(value: str | None) -> int | None:
    if value is None:
        return None
    s = str(value).strip()
    if not s:
        return None
    try:
        return int(float(s))
    except Exception:
        return None


def load_products(base_dir: Path, limit: int = 30) -> list[dict]:
    """
    Reads a local CSV file and returns a list of product dicts.
    CSV columns supported:
      - title, brand, category, color_name, color_hex, price_nok, url, image_url
    """
    limit = max(1, min(int(limit or 30), 100))

    feed_path = None
    for name in DEFAULT_FEED_CANDIDATES:
        candidate = (base_dir / name).resolve()
        if candidate.is_file():
            feed_path = candidate
            break

    if not feed_path:
        return []

    items: list[dict] = []
    with feed_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not row:
                continue
            title = (row.get("title") or "").strip()
            url = (row.get("url") or "").strip()
            if not title or not url:
                continue
            items.append(
                {
                    "title": title,
                    "brand": (row.get("brand") or "").strip() or None,
                    "category": (row.get("category") or "").strip() or None,
                    "color_name": (row.get("color_name") or "").strip() or None,
                    "color_hex": (row.get("color_hex") or "").strip() or None,
                    "price_nok": _to_int(row.get("price_nok")),
                    "url": url,
                    "image_url": (row.get("image_url") or "").strip() or None,
                }
            )
            if len(items) >= limit:
                break

    return items

