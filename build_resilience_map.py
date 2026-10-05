"""
Balanced-day food access map for San Diego County.

For any point on the map, the page answers two questions:

  1. TRIP DISTANCE (main score): how far would one person have to travel to
     gather one balanced day of food under the 2025-2030 Dietary Guidelines
     ("the New Pyramid"): protein foods, dairy, vegetables, fruits, healthy
     fats and whole grains, each from a nearby source that actually produces it?
     It finds the shortest single errand loop (home -> sources -> home) that
     covers every food group, and also the total of separate round trips.

  2. SUPPLY IN RADIUS: what food, and how much, is produced within an adjustable
     radius, and how many people could that feed per food group.

Run from the folder that holds data/ :
    python build_resilience_map.py

Inputs
    data/Agricultural_Commodity_2020.geojson   County AWM grower sites (SANDAG/SanGIS)
    data/AWM_Certified_Producers.geojson       County certified (farmers' market) producers
    ref/sd_county.geojson                      county outline (from plotly/datasets, US Census)
    ref/sd_coast.geojson                       coastline (Natural Earth 10m)

Outputs
    docs/index.html                     the interactive map (GitHub Pages serves this)
    output/resilience_map.html          same page, for local use
    output/site_allocation.csv          acres assigned to each crop on each site
    output/sources.csv                  every food source, its foods and yearly kg
    output/allocation_check.csv         allocated acres vs Crop Report targets
    output/unmodeled_commodities.csv    commodity names with no food rule

Everything below marked ASSUMPTION is an editable estimate. The model is built
to be transparent and adjustable, not authoritative.
"""

import json
import math
import re
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point, shape
from shapely.prepared import prep

DATA, REF, OUT, DOCS = Path("data"), Path("ref"), Path("output"), Path("docs")
OUT.mkdir(exist_ok=True)
DOCS.mkdir(exist_ok=True)
LB = 0.453592  # kg per lb

# ===========================================================================
# 1. FOOD GROUPS (2025-2030 Dietary Guidelines for Americans, realfood.gov)
#    Daily need for one adult, ASSUMPTION where the guidelines give no size:
#      protein foods  1.2-1.6 g protein / kg body weight -> 100 g protein (70 kg adult)
#      dairy          3 servings  -> 735 g milk-equivalent (3 cups)
#      vegetables     3 servings  -> 240 g  (80 g / serving)
#      fruits         2 servings  -> 240 g  (120 g / serving)
#      whole grains   2-4 servings -> 85 g dry (3 x 28 g)
#      healthy fats   no count given -> 30 g fat from fat-source foods
# ===========================================================================
GROUPS = ["Protein", "Dairy", "Vegetables", "Fruits", "Healthy fats", "Whole grains"]
P, D, V, F, H, G = (1 << i for i in range(6))

# Daily need, and which nutrient of the food fills it ("mass" = the food itself)
NEED = {  # group bit: (kg per day, measure)
    P: (0.100, "protein"), D: (0.735, "mass"), V: (0.240, "mass"),
    F: (0.240, "mass"), H: (0.030, "fat"), G: (0.085, "mass"),
}

# ===========================================================================
# 2. FOODS: name, groups, protein/carb/fat g per 100 g edible
# ===========================================================================
FOODS = {}


def food(name, groups, p, c, f):
    FOODS[name] = dict(groups=groups, p=p, c=c, f=f)


# fruit & nut crops
food("Avocado", F | H, 2.0, 8.5, 14.7)
food("Lemon", F, 1.1, 9.3, 0.3)
food("Orange", F, 0.9, 11.8, 0.1)
food("Grapefruit", F, 0.8, 10.7, 0.1)
food("Mandarin", F, 0.8, 13.3, 0.3)
food("Lime", F, 0.7, 10.5, 0.2)
food("Kumquat", F, 1.9, 15.9, 0.9)
food("Citrus (mixed)", F, 0.9, 11.0, 0.2)
food("Strawberry", F, 0.7, 7.7, 0.3)
food("Berries", F, 0.7, 14.5, 0.3)
food("Table grape", F, 0.7, 18.1, 0.2)
food("Persimmon", F, 0.6, 18.6, 0.2)
food("Pomegranate", F, 1.7, 18.7, 1.2)
food("Guava", F, 2.6, 14.3, 1.0)
food("Apple / pear", F, 0.3, 13.8, 0.2)
food("Stone fruit", F, 0.9, 11.0, 0.3)
food("Fig / date", F, 1.5, 45.0, 0.4)
food("Subtropical fruit", F, 1.3, 16.0, 0.5)
food("Melon", F, 0.7, 7.8, 0.2)
food("Olive", H, 0.8, 6.3, 10.7)
food("Tree nuts", P | H, 15.0, 15.0, 65.0)
# vegetables
food("Tomato", V, 0.9, 3.9, 0.2)
food("Peppers", V, 0.9, 6.0, 0.2)
food("Squash & cucumber", V, 0.9, 4.5, 0.2)
food("Leafy greens", V, 1.8, 3.5, 0.3)
food("Brassicas", V, 3.0, 7.0, 0.3)
food("Root veg & onions", V, 1.2, 9.5, 0.1)
food("Potato", V, 2.0, 17.5, 0.1)
food("Sweet corn", V, 3.3, 19.0, 1.4)
food("Green beans & peas", V, 2.5, 8.0, 0.3)
food("Mushrooms", V, 3.1, 3.3, 0.3)
food("Other vegetables", V, 2.0, 6.0, 0.2)
food("Herbs & spices", 0, 3.0, 7.0, 0.6)  # seasoning, not a vegetable serving
# grains & legumes
food("Wheat / grain", G, 12.5, 72.0, 1.8)
food("Oats", 0, 16.9, 66.3, 6.9)  # grain only if the "oats as grain" toggle is on
food("Dry beans", P, 21.0, 62.0, 1.2)
# animal foods
food("Beef", P | H, 18.6, 0.0, 15.0)
food("Lamb", P | H, 16.6, 0.0, 21.0)
food("Pork", P | H, 16.9, 0.0, 21.0)
food("Chicken", P | H, 17.5, 0.0, 15.0)
food("Eggs", P | H, 12.6, 0.7, 9.5)
food("Milk & dairy", D | H, 3.2, 4.8, 3.3)
food("Fish", P | H, 20.0, 0.0, 4.0)
food("Shellfish", P, 11.9, 3.7, 2.2)
# unknown-crop farmers' market growers
food("Mixed produce", V | F, 1.0, 8.0, 0.2)

FOOD_NAMES = list(FOODS)
FI = {n: i for i, n in enumerate(FOOD_NAMES)}

# ===========================================================================
# 3. COMMODITY -> FOOD rules, yields (lb per acre per year) and edible share.
#    Yields for the big crops come from the San Diego County 2017 Crop
#    Statistics & Annual Report (production / harvested acres); others are
#    ASSUMPTION regional defaults.
# ===========================================================================
RULES = [  # (regex on the commodity item, food, lb/acre/yr, edible fraction)
    (r"^AVOCADO", "Avocado", 4600, 0.70),
    (r"^LEMON", "Lemon", 30000, 0.70),
    (r"^ORANGE$", "Orange", 19000, 0.73),
    (r"^GRAPEFRUIT|^POMELO", "Grapefruit", 31600, 0.70),
    (r"^TANGERINE|^TANGELO", "Mandarin", 26000, 0.74),
    (r"^LIME$", "Lime", 9600, 0.70),
    (r"^KUMQUAT", "Kumquat", 4000, 0.93),
    (r"^CITRUS", "Citrus (mixed)", 20000, 0.72),
    (r"^STRAWBERRY", "Strawberry", 50000, 0.94),
    (r"BERRY$|^BERRY", "Berries", 16000, 0.95),
    (r"^GRAPE$", "Table grape", 16000, 0.95),
    (r"^PERSIMMON", "Persimmon", 7000, 0.85),
    (r"^POMEGRANATE", "Pomegranate", 12000, 0.56),
    (r"^GUAVA", "Guava", 10000, 0.90),
    (r"^APPLE|PEAR$|^POME FRUIT|^QUINCE", "Apple / pear", 4000, 0.90),
    (r"^STONE FRUIT|^PEACH|^PLUM|^PLUOT|^NECTARINE|^APRICOT|^CHERRY", "Stone fruit", 10000, 0.90),
    (r"^FIG$|^DATE$", "Fig / date", 8000, 0.95),
    (r"^TROPICAL|^CHERIMOYA|^MANGO|^BANANA|^KIWI|^JUJUBE|^FRUIT$", "Subtropical fruit", 10000, 0.75),
    (r"MELON", "Melon", 30000, 0.55),
    (r"^OLIVE", "Olive", 6000, 0.85),
    (r"NUT$|^NUTS$|^ALMOND|^WALNUT|^PECAN", "Tree nuts", 1500, 1.00),
    (r"^TOMATO$|^TOMATILLO", "Tomato", 30000, 0.95),
    (r"^PEPPER", "Peppers", 32000, 0.85),
    (r"^SQUASH|^ZUCCHINI|^PUMPKIN|^CUCUMBER|^EGGPLANT|^OKRA", "Squash & cucumber", 18000, 0.80),
    (r"^LETTUCE|^KALE|^SWISS CHARD|^SPINACH|^COLLARD|^MUSTARD|^ARUGULA|^ENDIVE|^BOK CHOY|"
     r"^CHINESE GREEN|^VEGETABLE LEAF|^EDIBLE LEAVED|^CELERY|^CEL$|^CARDOON|^RHUBARB|^FENNEL", "Leafy greens", 20000, 0.85),
    (r"^BROCCOLI|^CABBAGE|^CAULIFLOWER|^BRUSSEL|^COLE CROP", "Brassicas", 16000, 0.80),
    (r"^BEET|^CARROT|^RADISH|^TURNIP|^RUTABAGA|^ONION|^GARLIC|^VEGETABLE ROOT", "Root veg & onions", 30000, 0.90),
    (r"^POTATO", "Potato", 40000, 0.85),
    (r"^CORN", "Sweet corn", 16000, 0.45),
    (r"^BEAN SUCCULENT|^BEAN UNSPECIFD|^PEAS", "Green beans & peas", 10000, 0.90),
    (r"^MUSHROOM$", "Mushrooms", 500000, 1.00),
    (r"^VEGETABLE|^ARTICHOKE", "Other vegetables", 15000, 0.75),
    (r"^HERB|^SPICE|BASIL|^PARSLEY|^CILANTRO|^MINT|^SAGE|^ROSEMARY|^TARRAGON|^THYME|^DILL|"
     r"^CHERVIL|^ANISE|^GINGER", "Herbs & spices", 15000, 0.90),
    (r"^WHEAT$|^GRAIN$|^BARLEY$", "Wheat / grain", 3000, 1.00),
    (r"^OAT$", "Oats", 3000, 1.00),
    (r"^BEAN DRIED", "Dry beans", 2000, 1.00),
]
RULES = [(re.compile(rx), f, y, e) for rx, f, y, e in RULES]

FORAGE = re.compile(r"FOR/FOD|^ALFALFA|^SUDANGRASS|^CLOVER|^PASTURELAND|^OAT$")
IRRIGATED_FORAGE = re.compile(r"FOR/FOD|^ALFALFA|^SUDANGRASS|^CLOVER|^PASTURELAND")
NON_PRODUCTIVE = re.compile(  # buildings, landscaping, fumigation, etc.
    r"FARM/AG BUILDING|RECREATION|LANDSCAPE|INDUSTRIAL|FUMIG|^SOIL|ORCHARD FLOOR|UNCUL NON-AG")
NOT_FOOD = re.compile(  # real production, but not food for this model
    r"^N-|^O[PFT]-|^G[PF]-|TURF|CHRISTMAS|WINE|HOPS|SUGARCANE|BEEHIVE|EDIBLE FLOWERS|SEED|"
    r"CHRYSANTHEMUM|MUSHROOM HOUSE|^EGG$")

# ASSUMPTION: relative starting weight of an item before calibration
START_WEIGHT = {"nonprod": 0.1, "uncultivated": 0.5}

# Crop Report 2017 harvested acres used to calibrate mixed sites (raking)
TARGETS = {
    "avocado": (r"^AVOCADO", 15003), "lemon": (r"^LEMON", 3312), "orange": (r"^ORANGE$", 6086),
    "grapefruit": (r"^GRAPEFRUIT|^POMELO", 1473), "mandarin": (r"^TANGERINE|^TANGELO", 1052),
    "kumquat": (r"^KUMQUAT", 69), "lime": (r"^LIME$", 218), "winegrape": (r"^GRAPE, WINE", 1210),
    "persimmon": (r"^PERSIMMON", 196), "strawberry": (r"^STRAWBERRY", 290),
    "apple": (r"^APPLE", 244), "berries": (r"BERRY$|^BERRY", 292),
    "cucumber": (r"^CUCUMBER", 99), "peppers": (r"^PEPPER", 169),
    "squash": (r"^SQUASH|^ZUCCHINI|^PUMPKIN", 411), "tomato": (r"^TOMATO$", 1136),
    "herbs": (r"^HERB|BASIL|^PARSLEY|^CILANTRO|^MINT|^SAGE|^ROSEMARY|^TARRAGON|^THYME|^DILL|^CHERVIL", 135),
    "oathay": (r"^OAT$|^OAT FOR/FOD", 2133), "pasture": (r"^PASTURELAND", 823),
    "miscveg": (r"^LETTUCE|^KALE|^SWISS CHARD|^SPINACH|^COLLARD|^MUSTARD|^ARUGULA|^BROCCOLI|^CABBAGE|"
                r"^CAULIFLOWER|^BRUSSEL|^COLE|^BEET|^CARROT|^RADISH|^TURNIP|^RUTABAGA|^ONION|^GARLIC|"
                r"^CORN|^BEAN SUCC|^PEAS|^EGGPLANT|^VEGETABLE|^CELERY|^ARTICHOKE|^CHINESE|^BOK", 1314),
}
TARGETS = {k: (re.compile(rx), t) for k, (rx, t) in TARGETS.items()}

# ===========================================================================
# 4. LIVESTOCK: county totals (2017 Crop Report) spread over sites that show
#    signs of animals. ASSUMPTION: live-weight -> edible share.
# ===========================================================================
LIVESTOCK = [  # (food, county total kg edible / yr, where it goes)
    ("Beef", 10800 * 100 * LB * 0.40, "forage"),      # 10,800 cwt cattle & calves
    ("Lamb", 968 * 100 * LB * 0.45, "forage"),        # 968 cwt lambs & sheep
    ("Pork", 1302 * 100 * LB * 0.55, "forage"),       # 1,302 cwt hogs & pigs
    ("Milk & dairy", 425363 * 100 * LB, "dairy"),     # 425,363 cwt market milk
    ("Eggs", 26427177 * 0.60, "eggs"),                # 26.4 M dozen x 0.6 kg
    ("Chicken", 63280 * 100 * LB * 0.55, "eggs"),     # chickens (mostly spent hens)
]

# ===========================================================================
# 5. FISHING POTENTIAL. ASSUMPTION: rough sustainable edible catch, kg / yr.
#    Coordinates are approximate public access points; edit freely.
# ===========================================================================
FISHING = [  # name, kind, lat, lon, edible kg/yr, food
    ("Point Loma sportfishing landings", "Sportfishing landing", 32.7230, -117.2280, 400000, "Fish"),
    ("Seaforth Landing, Mission Bay", "Sportfishing landing", 32.7637, -117.2370, 60000, "Fish"),
    ("Oceanside Harbor landing", "Sportfishing landing", 33.2069, -117.3946, 60000, "Fish"),
    ("Tuna Harbor dockside market", "Commercial fish landing", 32.7135, -117.1735, 300000, "Fish"),
    ("Carlsbad shellfish farm (Agua Hedionda)", "Shellfish farm", 33.1450, -117.3270, 100000, "Shellfish"),
    ("Oceanside Pier", "Pier", 33.1934, -117.3866, 5000, "Fish"),
    ("Crystal Pier", "Pier", 32.7963, -117.2577, 5000, "Fish"),
    ("Ocean Beach Pier", "Pier", 32.7495, -117.2556, 5000, "Fish"),
    ("Imperial Beach Pier", "Pier", 32.5794, -117.1351, 5000, "Fish"),
    ("Shelter Island Pier", "Pier", 32.7155, -117.2290, 5000, "Fish"),
    ("Embarcadero fishing pier", "Pier", 32.7058, -117.1680, 3000, "Fish"),
    ("Chula Vista Bayfront pier", "Pier", 32.6260, -117.1060, 3000, "Fish"),
    ("Pepper Park pier, National City", "Pier", 32.6440, -117.1130, 2000, "Fish"),
    ("Lake Jennings", "Lake (stocked)", 32.8590, -116.8870, 3000, "Fish"),
    ("Lake Poway", "Lake (stocked)", 33.0070, -117.0080, 3000, "Fish"),
    ("Dixon Lake", "Lake (stocked)", 33.1625, -117.0490, 3000, "Fish"),
    ("Lake Wohlford", "Lake (stocked)", 33.1700, -117.0000, 3000, "Fish"),
    ("Lake Hodges", "Lake (stocked)", 33.0560, -117.1250, 8000, "Fish"),
    ("Santee Lakes", "Lake (stocked)", 32.8607, -116.9926, 3000, "Fish"),
    ("Lake Murray", "Lake (stocked)", 32.7830, -117.0420, 3000, "Fish"),
    ("Lake Miramar", "Lake (stocked)", 32.9115, -117.0960, 3000, "Fish"),
    ("San Vicente Reservoir", "Lake (stocked)", 32.9150, -116.9250, 8000, "Fish"),
    ("El Capitan Reservoir", "Lake (stocked)", 32.8930, -116.8130, 8000, "Fish"),
    ("Lower Otay Lake", "Lake (stocked)", 32.6150, -116.9250, 8000, "Fish"),
    ("Lake Cuyamaca", "Lake (stocked)", 32.9870, -116.5800, 8000, "Fish"),
    ("Lake Morena", "Lake (stocked)", 32.6860, -116.5230, 5000, "Fish"),
    ("Lake Henshaw", "Lake (stocked)", 33.2360, -116.7630, 8000, "Fish"),
    ("Barrett Lake", "Lake (stocked)", 32.6800, -116.6700, 3000, "Fish"),
    ("Lake Sutherland", "Lake (stocked)", 33.1150, -116.7850, 3000, "Fish"),
    ("Guajome Lake", "Lake (stocked)", 33.2490, -117.2690, 1000, "Fish"),
    ("Lindo Lake", "Lake (stocked)", 32.8590, -116.9230, 1000, "Fish"),
]
SURF_KG_PER_SAMPLE = 400       # ASSUMPTION: shore fishing per 2 km of open coast
SURF_SPACING_KM = 2.0
NO_ACCESS_LAT = (33.215, 33.36)  # Camp Pendleton coast: no public fishing

# Certified-producer business types read from the business name (no names are published)
CERT_EGGS = re.compile(r"EGG|HENS|DUCK|POULTRY")
CERT_FISH = re.compile(r"AQUAPONIC|FISH")
AQUAPONICS_KG = 2000
MARKET_GROWER_LB_PER_ACRE = 12000  # ASSUMPTION for unknown-crop market growers

QUALIFIERS = {"HUMAN CON", "SPICE", "SUMMER", "WINTER", "WINE"}  # glue onto the item before
TRUNCATED = {"L", "CEL"}  # cut-off fragments at the 254-character field limit


# ===========================================================================
def parse_items(commodity: str) -> list:
    parts = [p.strip().upper() for p in str(commodity or "").split(",")]
    items = []
    for p in parts:
        if not p:
            continue
        if p in QUALIFIERS and items:
            items[-1] = f"{items[-1]}, {p}"
        elif p in TRUNCATED:
            continue
        else:
            items.append(p)
    return list(dict.fromkeys(items))


def item_kind(item: str) -> str:
    if NON_PRODUCTIVE.search(item):
        return "nonprod"
    if item.startswith("UNCULTIVATED"):
        return "uncultivated"
    if FORAGE.search(item):
        return "forage"
    if NOT_FOOD.search(item):
        return "notfood"
    for rx, *_ in RULES:
        if rx.search(item):
            return "food"
    return "unmodeled"


def food_rule(item: str):
    for rx, name, y, e in RULES:
        if rx.search(item):
            return name, y, e
    return None


def nutrient_person_days(food_name: str, kg: float) -> np.ndarray:
    """Person-days per year of each food group that `kg` of a food provides."""
    fd = FOODS[food_name]
    out = np.zeros(len(GROUPS))
    for i, bit in enumerate([P, D, V, F, H, G]):
        if not fd["groups"] & bit:
            continue
        need, measure = NEED[bit]
        amount = kg * (fd["p"] / 100 if measure == "protein" else fd["f"] / 100 if measure == "fat" else 1)
        out[i] = amount / need
    return out


# ===========================================================================
def load_sites() -> gpd.GeoDataFrame:
    gdf = gpd.read_file(DATA / "Agricultural_Commodity_2020.geojson")
    if gdf.crs is None or gdf.crs.to_epsg() != 2230:
        gdf = gdf.set_crs("EPSG:2230", allow_override=True)
    before = len(gdf)
    gdf = gdf[~gdf.geometry.to_wkb().duplicated()].copy()
    print(f"Removed {before - len(gdf)} exact duplicate polygons ({len(gdf)} sites remain)")
    bad = ~gdf.is_valid
    gdf.loc[bad, "geometry"] = gdf.loc[bad, "geometry"].make_valid()
    pts = gdf.geometry.representative_point().to_crs(4326)
    gdf["lat"], gdf["lon"] = pts.y, pts.x
    gdf["items"] = gdf["commodity"].map(parse_items)
    return gdf


def rake(gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    """Split each site's acres among its items, calibrated to Crop Report totals."""
    rows = []
    for _, s in gdf.iterrows():
        for it in s["items"] or ["UNKNOWN"]:
            kind = item_kind(it)
            rows.append(dict(objectid=s["objectid"], item=it, kind=kind,
                             site_acres=s["calc_acres"], w=START_WEIGHT.get(kind, 1.0)))
    t = pd.DataFrame(rows)
    t["target"] = None
    for key, (rx, _) in TARGETS.items():
        t.loc[t["item"].str.contains(rx) & t["target"].isna(), "target"] = key

    for _ in range(60):
        t["acres"] = t["site_acres"] * t["w"] / t.groupby("objectid")["w"].transform("sum")
        for key, (_, target) in TARGETS.items():
            m = t["target"] == key
            total = t.loc[m, "acres"].sum()
            if total > 0:
                t.loc[m, "w"] *= np.clip(target / total, 0.5, 2.0)
    t["acres"] = t["site_acres"] * t["w"] / t.groupby("objectid")["w"].transform("sum")

    check = []
    for key, (rx, target) in TARGETS.items():
        m = t["item"].str.contains(rx)
        even = (t.loc[m, "site_acres"] / t.loc[m].groupby("objectid")["item"].transform("size")).sum()
        check.append(dict(group=key, crop_report_acres=target,
                          even_split_acres=round(even), calibrated_acres=round(t.loc[m, "acres"].sum())))
    check = pd.DataFrame(check)
    check.to_csv(OUT / "allocation_check.csv", index=False)
    print("\nMixed-farm calibration (acres):")
    print(check.to_string(index=False))
    return t


def build_sources(gdf, alloc):
    """Every food source as a point with its foods (kg/yr)."""
    sites = gdf.set_index("objectid")
    sources = []   # dicts: lat, lon, kind, label, foods{name: kg}
    site_src = {}  # objectid -> index in sources

    # crops
    for oid, grp in alloc.groupby("objectid"):
        s = sites.loc[oid]
        foods = {}
        for _, r in grp.iterrows():
            if r["kind"] != "food":
                continue
            name, y, e = food_rule(r["item"])
            foods[name] = foods.get(name, 0) + r["acres"] * y * LB * e
        site_src[oid] = len(sources)
        sources.append(dict(lat=s["lat"], lon=s["lon"], kind="Farm", foods=foods,
                            acres=float(s["calc_acres"]), items=grp["item"].tolist(),
                            alloc=dict(zip(grp["item"], grp["acres"].round(2)))))

    # livestock: weights per site
    w_forage = alloc[alloc["kind"] == "forage"].groupby("objectid")["acres"].sum()
    w_unc = alloc[alloc["kind"] == "uncultivated"].groupby("objectid")["acres"].sum() * 0.5
    w_forage = w_forage.add(w_unc, fill_value=0)
    w_dairy = alloc[alloc["item"].str.contains(IRRIGATED_FORAGE)].groupby("objectid")["acres"].sum()
    w_eggs = alloc[alloc["item"].eq("EGG")].groupby("objectid")["acres"].sum()

    # certified producers: egg & aquaponics by business type; others optional
    cert = gpd.read_file(DATA / "AWM_Certified_Producers.geojson").to_crs(2230)
    cert_pts = cert.geometry.representative_point()
    cert_ll = gpd.GeoSeries(cert_pts, crs=2230).to_crs(4326)
    in_layer = gpd.sjoin(gpd.GeoDataFrame(geometry=cert.geometry, crs=2230),
                         gdf[["geometry"]], how="left", predicate="intersects")
    overlaps = in_layer["index_right"].notna().groupby(level=0).any()
    egg_src = []
    for i, c in cert.iterrows():
        name = str(c["Business_N"]).upper()
        lat, lon = cert_ll.iloc[i].y, cert_ll.iloc[i].x
        area = max(float(c["Area"] or 0), 1.0)
        if CERT_EGGS.search(name):
            egg_src.append((len(sources), area))
            sources.append(dict(lat=lat, lon=lon, kind="Egg & poultry producer", foods={}, acres=area))
        elif CERT_FISH.search(name):
            sources.append(dict(lat=lat, lon=lon, kind="Aquaponics farm", foods={"Fish": AQUAPONICS_KG},
                                acres=area))
        elif not overlaps.iloc[i]:
            kg = area * MARKET_GROWER_LB_PER_ACRE * LB
            sources.append(dict(lat=lat, lon=lon, kind="Market grower (crops unknown)",
                                foods={"Mixed produce": kg}, acres=area))

    def spread(food_name, total_kg, weights_by_index):
        tot = sum(weights_by_index.values())
        for idx, w in weights_by_index.items():
            sources[idx]["foods"][food_name] = sources[idx]["foods"].get(food_name, 0) + total_kg * w / tot

    weights = {
        "forage": {site_src[o]: w for o, w in w_forage.items() if w > 0},
        "dairy": {site_src[o]: w for o, w in w_dairy.items() if w > 0},
        "eggs": {**{site_src[o]: w for o, w in w_eggs.items() if w > 0}, **dict(egg_src)},
    }
    print("\nLivestock placed on:", {k: len(v) for k, v in weights.items()}, "sources")
    for food_name, kg, where in LIVESTOCK:
        spread(food_name, kg, weights[where])

    # fishing
    for name, kind, lat, lon, kg, fd in FISHING:
        sources.append(dict(lat=lat, lon=lon, kind=kind, name=name, foods={fd: kg}))
    coast = shape(json.load(open(REF / "sd_coast.geojson"))["geometry"])
    county = shape(json.load(open(REF / "sd_county.geojson"))["geometry"])
    step = SURF_SPACING_KM / 100.0  # ~degrees; resampled below in km
    n_surf = 0
    for line in getattr(coast, "geoms", [coast]):
        L = line.length
        d = 0.0
        while d <= L:
            p = line.interpolate(d)
            if 32.535 < p.y < 33.505 and not (NO_ACCESS_LAT[0] < p.y < NO_ACCESS_LAT[1]) \
                    and county.buffer(0.02).contains(p):
                sources.append(dict(lat=p.y, lon=p.x, kind="Shore fishing", name="Open-coast shore fishing",
                                    foods={"Fish": SURF_KG_PER_SAMPLE}))
                n_surf += 1
            d += step
    print(f"Fishing: {len(FISHING)} named spots + {n_surf} shore points")
    return sources


def main():
    gdf = load_sites()
    alloc = rake(gdf)
    alloc.to_csv(OUT / "site_allocation.csv", index=False)

    um = alloc[alloc["kind"] == "unmodeled"].groupby("item")["acres"].agg(["size", "sum"]).round(1)
    um.columns = ["sites", "acres"]
    um.sort_values("acres", ascending=False).to_csv(OUT / "unmodeled_commodities.csv")

    sources = build_sources(gdf, alloc)

    # ---- per-source person-days by group, and compact rows for the page
    kinds = sorted({s["kind"] for s in sources}, key=lambda k: (k != "Farm", k))
    rows, csv_rows = [], []
    oat_i = FI["Oats"]
    for s in sources:
        foods = {k: v for k, v in s["foods"].items() if v >= 1}
        if not foods:
            continue
        pdays = np.zeros(len(GROUPS))
        for name, kg in foods.items():
            pdays += nutrient_person_days(name, kg)
        oat_grain_days = foods.get("Oats", 0) / NEED[G][0]
        label = s.get("name") or ""
        if s["kind"] == "Farm":
            top = sorted(foods.items(), key=lambda kv: -kv[1])[:3]
            label = ", ".join(n for n, _ in top)
        rows.append([round(s["lat"], 5), round(s["lon"], 5), kinds.index(s["kind"]), label,
                     [round(x) for x in pdays], round(oat_grain_days),
                     [[FI[n], round(kg)] for n, kg in sorted(foods.items(), key=lambda kv: -kv[1])]])
        csv_rows.append(dict(kind=s["kind"], label=label, lat=round(s["lat"], 5), lon=round(s["lon"], 5),
                             **{f"{n} kg/yr": round(kg) for n, kg in foods.items()},
                             **{f"{g} person-days/yr": round(x) for g, x in zip(GROUPS, pdays)}))
    pd.DataFrame(csv_rows).to_csv(OUT / "sources.csv", index=False)

    tot = np.array([r[4] for r in rows]).sum(axis=0) / 365
    print("\nCounty supply, people fed for a year per food group:")
    for g, x in zip(GROUPS, tot):
        print(f"  {g:13s} {x:>12,.0f}")
    print(f"Sources: {len(rows):,}  ({', '.join(f'{k}: {sum(1 for r in rows if kinds[r[2]] == k)}' for k in kinds)})")

    # ---- land mask grid over the county
    county = shape(json.load(open(REF / "sd_county.geojson"))["geometry"])
    pc = prep(county)
    cell = 0.01
    s_, w_, n_, e_ = 32.53, -117.62, 33.51, -116.08
    nr, nc = math.ceil((n_ - s_) / cell), math.ceil((e_ - w_) / cell)
    mask = "".join("1" if pc.contains(Point(w_ + (c + .5) * cell, n_ - (r + .5) * cell)) else "0"
                   for r in range(nr) for c in range(nc))

    # ---- parcels (simplified) colored by dominant pyramid group
    dom = []
    for _, s in gdf.iterrows():
        grp = alloc[alloc["objectid"] == s["objectid"]]
        best, best_a = "Not food", 0
        for _, r in grp.iterrows():
            if r["kind"] == "food":
                gbits = FOODS[food_rule(r["item"])[0]]["groups"]
                gname = next((GROUPS[i] for i in range(6) if gbits & (1 << i)), "Not food")
            elif r["kind"] in ("forage", "uncultivated"):
                gname = "Livestock land"
            else:
                gname = "Not food"
            if r["acres"] > best_a:
                best, best_a = gname, r["acres"]
        dom.append(best)
    parcels = gdf[["geometry"]].copy()
    parcels["g"] = dom
    parcels["c"] = gdf["commodity"].str.slice(0, 120)
    parcels["a"] = gdf["calc_acres"].round(1)
    parcels["geometry"] = parcels.geometry.simplify(20)
    parcels = parcels.to_crs(4326)
    pj = json.loads(parcels.to_json(drop_id=True))
    for ft in pj["features"]:
        ft["geometry"]["coordinates"] = _round(ft["geometry"]["coordinates"])

    payload = dict(
        groups=GROUPS, foods=FOOD_NAMES,
        foodGroups=[FOODS[n]["groups"] for n in FOOD_NAMES],
        kcal100=[round(4 * FOODS[n]["p"] + 4 * FOODS[n]["c"] + 9 * FOODS[n]["f"]) for n in FOOD_NAMES],
        need=[NEED[1 << i][0] * 1000 for i in range(6)],
        kinds=kinds, fishKinds=[k for k in kinds if k in
                                {"Sportfishing landing", "Commercial fish landing", "Shellfish farm", "Pier",
                                 "Lake (stocked)", "Shore fishing", "Aquaponics farm"}],
        sources=rows, oatFood=oat_i,
        grid=dict(s=s_, w=w_, n=n_, e=e_, cell=cell, nr=nr, nc=nc, mask=mask),
        county=json.load(open(REF / "sd_county.geojson"))["geometry"],
    )
    html = (TEMPLATE.replace("__DATA__", json.dumps(payload, separators=(",", ":")))
            .replace("__PARCELS__", json.dumps(pj, separators=(",", ":"))))
    (OUT / "resilience_map.html").write_text(html, encoding="utf-8")
    (DOCS / "index.html").write_text(html, encoding="utf-8")
    print(f"\nWrote docs/index.html and output/resilience_map.html ({len(html) / 1e6:.1f} MB)")


def _round(c):
    if isinstance(c[0], (int, float)):
        return [round(c[0], 5), round(c[1], 5)]
    return [_round(x) for x in c]


TEMPLATE = (Path(__file__).parent / "map_template.html").read_text(encoding="utf-8")

if __name__ == "__main__":
    main()
