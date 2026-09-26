"""
Group San Diego grower sites into map-ready crop categories.

Run after (or instead of) explore_ag.py, from the folder that holds data/:
    python categorize_ag.py

Outputs (in output/):
    sites_categorized.geojson   one row per site, lat/long, with a `category` field
    category_summary.csv        acres and site counts per category
    unmatched_commodities.csv   commodity names that fell into "Other crops" - review these
    map_categories.png          countywide map colored by category
    bar_category_acres.png      acres by category
"""

import re
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Patch

DATA = Path("data")
OUT = Path("output")
OUT.mkdir(exist_ok=True)
SRC = DATA / "Agricultural_Commodity_2020.geojson"

# ---------------------------------------------------------------------------
# Category rules. Checked top to bottom; the first match wins, so citrus comes
# before grapes (GRAPEFRUIT contains "GRAPE"). Edit freely after reviewing
# output/unmatched_commodities.csv.
# ---------------------------------------------------------------------------
NON_CROP = "Non-crop"
OTHER = "Other crops"

RULES = [
    (NON_CROP, r"FARM/AG BUILDING|RECREATION|LANDSCAPE|UNCULTIVATED|RIGHTS? OF WAY|"
               r"STRUCTURAL|RESIDENTIAL|INDUSTRIAL|REGULATORY|ROAD|DITCH|FALLOW|"
               r"GOLF|CEMETERY|SCHOOL|PARK\b|NON-?AG"),
    ("Avocado", r"\bAVOCADO"),
    ("Citrus", r"LEMON|\bLIME\b|ORANGE|GRAPEFRUIT|TANGERINE|TANGELO|MANDARIN|"
               r"KUMQUAT|POMELO|CITRUS|CALAMONDIN"),
    ("Nursery, flowers & turf", r"^N-|FLOWER|FLWR|TURF|SOD\b|ORNAMENTAL|BEDDING"),
    ("Grapes & wine", r"\bGRAPE|\bWINE"),
    ("Vegetables", r"TOMATO|POTATO|BRUSSEL|LETTUCE|PEPPER|SQUASH|\bBEAN|CUCUMBER|"
                   r"CABBAGE|BROCCOLI|CAULIFLOWER|CELERY|CARROT|ONION|GARLIC|"
                   r"SPINACH|KALE|CHARD|HERB|BASIL|CILANTRO|PEA\b|PEAS\b|CORN|"
                   r"MELON|PUMPKIN|EGGPLANT|RADISH|BEET|LEEK|ARTICHOKE|"
                   r"ASPARAGUS|VEGETABLE|GREENS|SPROUT|MUSHROOM"),
    ("Other fruit & nuts", r"PERSIMMON|POMEGRANATE|GUAVA|APPLE|TROPICAL|FIG\b|"
                           r"PEACH|PLUM|APRICOT|NECTARINE|CHERIMOYA|PASSION|"
                           r"DRAGON|PITAHAYA|MACADAMIA|\bNUT|PECAN|WALNUT|ALMOND|"
                           r"OLIVE|BERR|SAPOTE|LOQUAT|MANGO|PEAR\b|PEARS\b|DATE\b|"
                           r"DATES\b|KIWI|CHERRY|BANANA|PAPAYA|LYCHEE|LONGAN|"
                           r"JUJUBE|FRUIT"),
    ("Grain & forage", r"\bOAT|WHEAT|BARLEY|ALFALFA|RYEGRAS|FOR/FOD|HAY\b|"
                       r"PASTURE|SUDAN|RANGELAND|GRAIN|TRITICALE|SORGHUM"),
]
COMPILED = [(cat, re.compile(pat)) for cat, pat in RULES]

# Fixed display order = fixed color assignment. A category keeps its color
# no matter how the data are filtered.
CATEGORY_ORDER = [
    "Avocado", "Citrus", "Nursery, flowers & turf", "Grapes & wine",
    "Other fruit & nuts", "Vegetables", "Grain & forage", OTHER,
]
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
           "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
COLORS = dict(zip(CATEGORY_ORDER, PALETTE))
COLORS[NON_CROP] = "#b8b8b3"


def categorize(item: str) -> str:
    for cat, rx in COMPILED:
        if rx.search(item):
            return cat
    return OTHER


def load() -> gpd.GeoDataFrame:
    gdf = gpd.read_file(SRC)
    if gdf.crs is None or gdf.crs.to_epsg() != 2230:
        gdf = gdf.set_crs("EPSG:2230", allow_override=True)
    # Your run found 8 invalid polygons; repair them so overlays/joins don't fail.
    bad = ~gdf.is_valid
    if bad.any():
        gdf.loc[bad, "geometry"] = gdf.loc[bad, "geometry"].make_valid()
        print(f"Repaired {bad.sum()} invalid geometries")
    return gdf


def build_items(gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    items = gdf[["objectid", "commodity"]].copy()
    items["item"] = items["commodity"].fillna("").str.split(",")
    items = items.explode("item")
    items["item"] = items["item"].fillna("").str.strip().str.upper()
    # Sites with a blank commodity keep one row so they aren't silently dropped.
    items.loc[items["item"] == "", "item"] = "UNKNOWN"
    items = items.drop_duplicates(["objectid", "item"])
    items["category"] = items["item"].map(categorize)
    return items


def primary_category(group: pd.DataFrame) -> pd.Series:
    crops = group[group["category"] != NON_CROP]
    if crops.empty:
        return pd.Series({"category": NON_CROP, "n_categories": 0,
                          "all_categories": NON_CROP})
    counts = crops["category"].value_counts()
    top = counts[counts == counts.max()].index
    # Ties go to the category that comes first in CATEGORY_ORDER.
    winner = min(top, key=CATEGORY_ORDER.index)
    cats = sorted(counts.index, key=CATEGORY_ORDER.index)
    return pd.Series({"category": winner, "n_categories": len(cats),
                      "all_categories": "; ".join(cats)})


def main() -> None:
    gdf = load()
    items = build_items(gdf)

    # ---- review list: what ended up in "Other crops"
    unmatched = (items[items["category"] == OTHER]
                 .groupby("item").agg(sites=("objectid", "nunique"))
                 .sort_values("sites", ascending=False))
    unmatched.to_csv(OUT / "unmatched_commodities.csv")
    print(f"\n{len(unmatched)} commodity names fell into '{OTHER}'. Top 25:")
    print(unmatched.head(25).to_string())

    print("\nNon-crop names excluded from categories:")
    print(sorted(items.loc[items["category"] == NON_CROP, "item"].unique()))

    # ---- one category per site
    per_site = items.groupby("objectid").apply(primary_category, include_groups=False)
    sites = gdf.merge(per_site, left_on="objectid", right_index=True, how="left")

    # ---- acres per category, two ways (no double counting in either)
    items = items.merge(gdf[["objectid", "calc_acres"]], on="objectid")
    crop_items = items[items["category"] != NON_CROP]
    n_crop_items = crop_items.groupby("objectid")["item"].transform("count")
    crop_items = crop_items.assign(acres_share=crop_items["calc_acres"] / n_crop_items)

    summary = pd.DataFrame({
        "sites": sites.groupby("category")["objectid"].count(),
        "acres_primary": sites.groupby("category")["calc_acres"].sum(),
        "acres_split": crop_items.groupby("category")["acres_share"].sum(),
        "mixed_sites": sites[sites["n_categories"] > 1].groupby("category")["objectid"].count(),
    }).reindex(CATEGORY_ORDER + [NON_CROP]).fillna(0)
    summary["pct_primary"] = 100 * summary["acres_primary"] / summary["acres_primary"].sum()
    summary = summary.round(1)
    summary.to_csv(OUT / "category_summary.csv")
    print("\nacres_primary = whole site assigned to its main category")
    print("acres_split   = site acres divided evenly across its listed crops")
    print(summary.to_string())

    # ---- export site layer for mapping elsewhere (QGIS, kepler.gl, ArcGIS Online)
    keep = ["objectid", "site_id", "site_name", "loc_narr", "commodity", "calc_acres",
            "category", "all_categories", "n_categories", "geometry"]
    sites_ll = sites[keep].to_crs(4326)
    sites_ll.to_file(OUT / "sites_categorized.geojson", driver="GeoJSON")

    # ---- map
    fig, ax = plt.subplots(figsize=(10, 10))
    fig.patch.set_facecolor("#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    present = [c for c in CATEGORY_ORDER + [NON_CROP] if (sites_ll["category"] == c).any()]
    # Draw big-acreage categories first so small ones stay visible on top.
    for cat in sorted(present, key=lambda c: -summary.loc[c, "acres_primary"]):
        sites_ll[sites_ll["category"] == cat].plot(
            ax=ax, color=COLORS[cat], edgecolor="#fcfcfb", linewidth=0.2)
    handles = [Patch(facecolor=COLORS[c], edgecolor="none",
                     label=f"{c}  ({summary.loc[c, 'acres_primary']:,.0f} ac)")
               for c in present]
    ax.legend(handles=handles, loc="lower left", frameon=False, fontsize=9,
              title="Primary crop per site", title_fontsize=10)
    ax.set_title("San Diego County commercial grower sites, 2020",
                 loc="left", fontsize=14, color="#1a1a19")
    ax.text(0, -0.02, "Source: County of San Diego Dept. of Agriculture, Weights & Measures via SANDAG/SanGIS",
            transform=ax.transAxes, fontsize=8, color="#6b6a64")
    ax.set_axis_off()
    fig.savefig(OUT / "map_categories.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    # ---- bar chart: acres by category (crop categories only)
    crop = summary.loc[[c for c in CATEGORY_ORDER if summary.loc[c, "sites"] > 0]]
    crop = crop.sort_values("acres_primary")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    fig.patch.set_facecolor("#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    bars = ax.barh(crop.index, crop["acres_primary"], height=0.7,
                   color=[COLORS[c] for c in crop.index])
    for bar, v in zip(bars, crop["acres_primary"]):
        ax.text(bar.get_width(), bar.get_y() + bar.get_height() / 2, f"  {v:,.0f}",
                va="center", fontsize=9, color="#1a1a19")
    ax.set_title("Grower acres by primary crop category", loc="left", fontsize=12)
    ax.set_xlabel("Acres", color="#6b6a64")
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#d4d3cc")
    ax.tick_params(colors="#6b6a64", length=0)
    ax.xaxis.grid(True, color="#ecebe6", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.margins(x=0.12)
    fig.savefig(OUT / "bar_category_acres.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    print(f"\nSaved to {OUT}/: sites_categorized.geojson, category_summary.csv, "
          "unmatched_commodities.csv, map_categories.png, bar_category_acres.png")


if __name__ == "__main__":
    main()
