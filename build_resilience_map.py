"""
Micro food-resilience map for San Diego County.

For any point on the map, scores how well the farmland within walking/biking
distance could supply a *balanced* and *diverse* diet: enough protein, carbs
AND fat, from many different foods, as close as possible.

Run from the folder that holds data/ (and categorize_ag.py):
    python build_resilience_map.py

Writes:
    output/resilience_map.html          open in any browser (needs internet for the basemap)
    output/site_nutrients.csv           per-site macro estimates, for checking
    output/unmodeled_commodities.csv    food-ish commodities with no nutrient rule yet

------------------------------------------------------------------------------
THE MODEL (all parameters below are editable)

1. Each site's acres (calc_acres) are split evenly among its listed crops.
   Non-crop items (buildings, recreation, etc.) get no share.
2. Each crop share -> annual edible harvest = acres x yield x edible fraction.
3. Harvest -> kg protein, carbs, fat (USDA-style composition per 100 g).
4. Kg -> "person-years": how many people that macro could fully cover for a
   year at FDA Daily Values (50 g protein, 275 g carbs, 78 g fat per day).
5. In the browser, for a point, every site within the radius contributes with
   a Gaussian distance weight (closer counts more).
6. Score (0-100) = 100 x sufficiency x diversity factor, where
     sufficiency = 1 - exp(-min(P, C, F) / target people)
         -> the *limiting* macro sets the score, so balance is required
     diversity   = (1 - w) + w x (1 - 1/D)
         -> D = effective number of foods (exp of Shannon entropy of
            calorie shares); w = "diversity advantage" slider.

The yields are rough California/San Diego defaults. For real results,
calibrate them against the San Diego County Crop Statistics & Annual Report
(production / harvested acres per commodity).
------------------------------------------------------------------------------
"""

import json
import re
from pathlib import Path

import geopandas as gpd
import pandas as pd

from categorize_ag import COLORS, NON_CROP, CATEGORY_ORDER, categorize

DATA = Path("data")
OUT = Path("output")
OUT.mkdir(exist_ok=True)
SRC = DATA / "Agricultural_Commodity_2020.geojson"

LB_TO_KG = 0.453592
# FDA Daily Values (2,000 kcal reference diet), kg per person per year
DV_KG_YEAR = {"p": 0.050 * 365, "c": 0.275 * 365, "f": 0.078 * 365}

# ---------------------------------------------------------------------------
# Nutrient rules: (food name, regex, yield lb/acre/yr, edible fraction,
#                  protein, carbs, fat in g per 100 g edible)
# First match wins -> keep specific patterns above generic ones.
# ---------------------------------------------------------------------------
FOODS = [
    ("Avocado",        r"\bAVOCADO",                 6000, 0.70, 2.0,  8.5, 14.7),
    ("Grapefruit",     r"GRAPEFRUIT|POMELO",        30000, 0.70, 0.8, 10.7,  0.1),
    ("Lemon",          r"LEMON",                    30000, 0.70, 1.1,  9.3,  0.3),
    ("Lime",           r"\bLIME\b",                 20000, 0.70, 0.7, 10.5,  0.2),
    ("Orange",         r"ORANGE",                   25000, 0.73, 0.9, 11.8,  0.1),
    ("Mandarin",       r"TANGERINE|TANGELO|MANDARIN|CLEMENTINE", 25000, 0.74, 0.8, 13.3, 0.3),
    ("Kumquat",        r"KUMQUAT|CALAMONDIN",       10000, 0.93, 1.9, 15.9,  0.9),
    ("Citrus (other)", r"CITRUS",                   25000, 0.72, 0.9, 11.0,  0.2),
    ("Strawberry",     r"STRAWBERR",                50000, 0.94, 0.7,  7.7,  0.3),
    ("Berries",        r"BERR",                     10000, 0.95, 0.7, 14.5,  0.3),
    ("Grape",          r"\bGRAPE",                   8000, 0.95, 0.7, 18.1,  0.2),
    ("Macadamia",      r"MACADAMIA",                 1000, 1.00, 7.9, 13.8, 75.8),
    ("Almond",         r"ALMOND",                    2000, 1.00, 21.2, 21.6, 49.9),
    ("Pecan",          r"PECAN",                     1200, 1.00, 9.2, 13.9, 72.0),
    ("Walnut / nuts",  r"WALNUT|\bNUT",              2000, 1.00, 15.2, 13.7, 65.2),
    ("Olive",          r"OLIVE",                     8000, 0.85, 0.8,  6.3, 10.7),
    ("Persimmon",      r"PERSIMMON",                15000, 0.85, 0.6, 18.6,  0.2),
    ("Pomegranate",    r"POMEGRANATE",              15000, 0.56, 1.7, 18.7,  1.2),
    ("Guava",          r"GUAVA",                    10000, 0.90, 2.6, 14.3,  1.0),
    ("Apple / pear",   r"APPLE|\bPEARS?\b",         20000, 0.90, 0.3, 13.8,  0.2),
    ("Stone fruit",    r"PEACH|PLUM|APRICOT|NECTARINE|CHERRY|CHERRIES|PRUNE", 15000, 0.90, 0.9, 11.0, 0.3),
    ("Fig",            r"\bFIGS?\b",                 8000, 1.00, 0.8, 19.2,  0.3),
    ("Date",           r"\bDATES?\b",                8000, 0.90, 2.5, 75.0,  0.4),
    ("Banana",         r"BANANA",                   15000, 0.64, 1.1, 22.8,  0.3),
    ("Mango",          r"MANGO",                    10000, 0.70, 0.8, 15.0,  0.4),
    ("Papaya",         r"PAPAYA",                   20000, 0.67, 0.5, 10.8,  0.3),
    ("Dragon fruit",   r"DRAGON|PITAHAYA|PITAYA",   15000, 0.70, 1.2, 13.0,  0.4),
    ("Passion fruit",  r"PASSION",                  10000, 0.50, 2.2, 23.4,  0.7),
    ("Subtropical fruit", r"CHERIMOYA|SAPOTE|LOQUAT|LYCHEE|LONGAN|KIWI|JUJUBE|"
                          r"FEIJOA|STARFRUIT|CARAMBOLA|TROPICAL|FRUIT",
                                                10000, 0.75, 1.3, 16.0,  0.5),
    ("Dry beans",      r"DRY BEAN|GARBANZO|LENTIL",  2000, 1.00, 21.0, 62.0,  1.2),
    ("Green beans / peas", r"\bBEANS?\b|\bPEAS?\b", 10000, 0.90, 2.5,  8.0,  0.3),
    ("Potato",         r"POTATO",                   40000, 0.85, 2.0, 17.5,  0.1),
    ("Sweet corn",     r"CORN",                     16000, 0.45, 3.3, 19.0,  1.4),
    ("Tomato",         r"TOMATO",                   60000, 0.95, 0.9,  3.9,  0.2),
    ("Brassicas",      r"BRUSSEL|BROCCOLI|CAULIFLOWER|CABBAGE|KOHLRABI|SPROUT",
                                                16000, 0.80, 3.0,  7.0,  0.3),
    ("Leafy greens & herbs", r"LETTUCE|SPINACH|KALE|CHARD|GREENS|HERB|BASIL|"
                             r"CILANTRO|PARSLEY|ARUGULA|CELERY",
                                                20000, 0.85, 1.5,  3.5,  0.3),
    ("Peppers",        r"PEPPER|CHILI",             30000, 0.85, 0.9,  6.0,  0.2),
    ("Root veg & alliums", r"CARROT|BEET|RADISH|TURNIP|ONION|LEEK|SHALLOT|GARLIC",
                                                35000, 0.90, 1.2,  9.5,  0.1),
    ("Squash & melons", r"SQUASH|ZUCCHINI|PUMPKIN|MELON|CUCUMBER|CUCURBIT|EGGPLANT",
                                                25000, 0.70, 0.9,  6.0,  0.2),
    ("Grain (oat)",    r"\bOATS?\b",                 2500, 1.00, 16.9, 66.3,  6.9),
    ("Grain (wheat, barley)", r"WHEAT|BARLEY|TRITICALE|SORGHUM|GRAIN",
                                                 3000, 1.00, 12.5, 72.0,  1.8),
    ("Mushrooms",      r"MUSHROOM",                200000, 1.00, 3.1,  3.3,  0.3),
    ("Vegetables (other)", r"VEGETABLE|ARTICHOKE|ASPARAGUS|OKRA",
                                                15000, 0.75, 2.0,  6.0,  0.2),
]
# Known non-food or not-for-direct-eating: counted as 0 food on purpose.
NOT_FOOD = re.compile(
    r"^N-|FLOWER|FLWR|TURF|\bSOD\b|ORNAMENTAL|BEDDING|\bWINE|ALFALFA|RYEGRAS|"
    r"FOR/FOD|\bHAY\b|PASTURE|SUDAN|RANGELAND|FARM/AG BUILDING|RECREATION|"
    r"LANDSCAPE|UNCULTIVATED|RIGHTS? OF WAY|STRUCTURAL|RESIDENTIAL|INDUSTRIAL|"
    r"REGULATORY|FALLOW|UNKNOWN|CHRISTMAS TREE|SEED")
FOOD_RX = [(name, re.compile(rx), y, e, p, c, f) for name, rx, y, e, p, c, f in FOODS]
FOOD_NAMES = [f[0] for f in FOODS]


def match_food(item: str):
    if NOT_FOOD.search(item):
        return "not_food", None
    for name, rx, y, e, p, c, f in FOOD_RX:
        if rx.search(item):
            return "food", (name, y, e, p, c, f)
    return "unmodeled", None


def load() -> gpd.GeoDataFrame:
    gdf = gpd.read_file(SRC)
    if gdf.crs is None or gdf.crs.to_epsg() != 2230:
        gdf = gdf.set_crs("EPSG:2230", allow_override=True)
    bad = ~gdf.is_valid
    if bad.any():
        gdf.loc[bad, "geometry"] = gdf.loc[bad, "geometry"].make_valid()
    return gdf


def main() -> None:
    gdf = load()
    pts = gdf.geometry.representative_point().to_crs(4326)
    gdf["lat"], gdf["lon"] = pts.y.round(5), pts.x.round(5)

    site_rows, detail_rows, unmodeled = [], [], []
    for _, s in gdf.iterrows():
        items = [i.strip().upper() for i in str(s["commodity"] or "").split(",") if i.strip()]
        items = list(dict.fromkeys(items)) or ["UNKNOWN"]
        crop_items = [i for i in items if categorize(i) != NON_CROP]
        share = s["calc_acres"] / len(crop_items) if crop_items else 0.0

        foods = {}  # food index -> [P, C, F person-years, Mkcal]
        for item in crop_items:
            kind, rule = match_food(item)
            if kind == "unmodeled":
                unmodeled.append((item, s["objectid"], share))
                continue
            if kind == "not_food":
                continue
            name, y, e, p, c, f = rule
            kg = share * y * LB_TO_KG * e
            pk, ck, fk = kg * p / 100, kg * c / 100, kg * f / 100
            idx = FOOD_NAMES.index(name)
            acc = foods.setdefault(idx, [0.0, 0.0, 0.0, 0.0])
            acc[0] += pk / DV_KG_YEAR["p"]
            acc[1] += ck / DV_KG_YEAR["c"]
            acc[2] += fk / DV_KG_YEAR["f"]
            acc[3] += (4 * pk + 4 * ck + 9 * fk) / 1000  # Mkcal
            detail_rows.append({"objectid": s["objectid"], "item": item, "food": name,
                                "acres": round(share, 2), "protein_kg": round(pk, 1),
                                "carb_kg": round(ck, 1), "fat_kg": round(fk, 1)})
        if foods:
            site_rows.append([float(s["lat"]), float(s["lon"]),
                              [[i] + [round(v, 3) for v in vals] for i, vals in foods.items()]])

    pd.DataFrame(detail_rows).to_csv(OUT / "site_nutrients.csv", index=False)
    um = pd.DataFrame(unmodeled, columns=["item", "objectid", "acres"])
    um = (um.groupby("item").agg(sites=("objectid", "nunique"), acres=("acres", "sum"))
            .sort_values("acres", ascending=False).round(1))
    um.to_csv(OUT / "unmodeled_commodities.csv")

    det = pd.DataFrame(detail_rows)
    tot = det[["protein_kg", "carb_kg", "fat_kg"]].sum()
    print(f"Food-producing sites: {len(site_rows):,} of {len(gdf):,}")
    print("County total, person-years supplied: "
          f"protein {tot.protein_kg / DV_KG_YEAR['p']:,.0f} | "
          f"carbs {tot.carb_kg / DV_KG_YEAR['c']:,.0f} | "
          f"fat {tot.fat_kg / DV_KG_YEAR['f']:,.0f}")
    print("\nTop foods by acres:")
    print(det.groupby("food")["acres"].sum().sort_values(ascending=False).head(12).round(0).to_string())
    print(f"\n{len(um)} commodity names have no nutrient rule (see unmodeled_commodities.csv):")
    print(um.head(15).to_string())

    # ---- parcels layer (simplified) colored by category
    items_cat = gdf["commodity"].fillna("").map(
        lambda c: [categorize(i.strip().upper()) for i in c.split(",") if i.strip()])

    def main_cat(cats):
        crops = [c for c in cats if c != NON_CROP]
        if not crops:
            return NON_CROP
        counts = pd.Series(crops).value_counts()
        return min(counts[counts == counts.max()].index, key=CATEGORY_ORDER.index)

    parcels = gdf[["commodity", "calc_acres", "geometry"]].copy()
    parcels["cat"] = items_cat.map(main_cat)
    parcels["geometry"] = parcels.geometry.simplify(15)
    parcels = parcels.to_crs(4326)
    parcels["calc_acres"] = parcels["calc_acres"].round(1)
    parcels_json = json.loads(parcels.to_json(drop_id=True))
    for feat in parcels_json["features"]:
        geom = feat["geometry"]
        geom["coordinates"] = _round_coords(geom["coordinates"])

    payload = {"foods": FOOD_NAMES, "sites": site_rows,
               "colors": COLORS, "catOrder": CATEGORY_ORDER + [NON_CROP]}
    html = (TEMPLATE
            .replace("__DATA__", json.dumps(payload, separators=(",", ":")))
            .replace("__PARCELS__", json.dumps(parcels_json, separators=(",", ":"))))
    (OUT / "resilience_map.html").write_text(html, encoding="utf-8")
    print(f"\nWrote {OUT/'resilience_map.html'}  -  open it in a browser.")


def _round_coords(c):
    if isinstance(c[0], (int, float)):
        return [round(c[0], 5), round(c[1], 5)]
    return [_round_coords(x) for x in c]


TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Food Resilience Map</title>
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.min.css">
<script src="https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.min.js"></script>
<style>
:root{
  --surface-0:#f3f3f0; --surface-1:#fcfcfb; --border:#e1e0d9;
  --text-primary:#1a1a19; --text-secondary:#56554f; --text-muted:#8a897f;
  --bar:#56554f; --bar-track:#ecebe6; --accent:#2a78d6;
}
@media (prefers-color-scheme: dark){
  :root{ --surface-0:#111110; --surface-1:#1a1a19; --border:#33332f;
    --text-primary:#ffffff; --text-secondary:#c3c2b7; --text-muted:#8f8e85;
    --bar:#c3c2b7; --bar-track:#2a2a27; --accent:#3987e5; }
}
*{box-sizing:border-box}
html,body{margin:0;height:100%;background:var(--surface-0);color:var(--text-primary);
  font:14px/1.45 system-ui,-apple-system,"Segoe UI",sans-serif}
#app{display:grid;grid-template-columns:360px 1fr;height:100vh}
#panel{background:var(--surface-1);border-right:1px solid var(--border);overflow-y:auto;padding:18px 18px 28px}
#map{height:100%}
h1{font-size:18px;margin:0 0 4px}
.sub{color:var(--text-secondary);margin:0 0 16px;font-size:13px}
.ctl{margin:0 0 14px}
.ctl label{display:flex;justify-content:space-between;font-size:13px;color:var(--text-secondary);margin-bottom:4px}
.ctl label b{color:var(--text-primary);font-weight:600}
input[type=range]{width:100%;accent-color:var(--accent)}
.toggles{display:flex;gap:16px;font-size:13px;color:var(--text-secondary);margin:0 0 16px}
.card{border-top:1px solid var(--border);padding-top:14px;margin-top:6px}
.hero{font-size:44px;font-weight:650;line-height:1;margin:2px 0 4px;font-variant-numeric:tabular-nums}
.hero small{font-size:16px;color:var(--text-muted);font-weight:400}
.muted{color:var(--text-muted);font-size:12px}
.row{display:grid;grid-template-columns:118px 1fr 76px;align-items:center;gap:8px;margin:7px 0;font-size:13px}
.row .val{text-align:right;font-variant-numeric:tabular-nums}
.track{position:relative;height:10px;background:var(--bar-track);border-radius:0 4px 4px 0}
.fill{position:absolute;left:0;top:0;bottom:0;background:var(--bar);border-radius:0 4px 4px 0}
.tick{position:absolute;top:-3px;bottom:-3px;width:2px;background:var(--text-primary)}
.limit{font-size:11px;color:var(--text-primary);border:1px solid var(--text-primary);border-radius:3px;padding:0 4px;margin-left:4px}
table{width:100%;border-collapse:collapse;font-size:13px;margin-top:6px}
td{padding:3px 0;border-bottom:1px solid var(--border)} td.n{text-align:right;font-variant-numeric:tabular-nums;color:var(--text-secondary)}
details{margin-top:18px;font-size:13px;color:var(--text-secondary)} summary{cursor:pointer;color:var(--text-primary)}
.legend{background:var(--surface-1);color:var(--text-primary);padding:8px 10px;border-radius:6px;border:1px solid var(--border);font-size:12px}
.ramp{width:180px;height:10px;border-radius:2px;margin:4px 0 2px}
.ramp-lbl{display:flex;justify-content:space-between;color:var(--text-muted)}
.cat{display:flex;align-items:center;gap:6px;margin:2px 0}.sw{width:12px;height:12px;border-radius:2px}
#hover{background:var(--surface-1);color:var(--text-primary);border:1px solid var(--border);border-radius:6px;padding:4px 8px;font-size:12px;font-variant-numeric:tabular-nums}
.leaflet-container{background:var(--surface-0)}
@media (max-width:760px){#app{grid-template-columns:1fr;grid-template-rows:55vh auto;height:auto}
  #map{height:55vh;order:-1}#panel{border-right:0;padding:16px}}
</style>
</head>
<body>
<div id="app">
 <aside id="panel">
  <h1>Micro food-resilience</h1>
  <p class="sub">How well nearby farms could feed a balanced, diverse diet. Click anywhere on the map.</p>

  <div class="ctl"><label>Radius <b id="rv"></b></label>
   <input id="radius" type="range" min="0.5" max="15" step="0.5" value="3"></div>
  <div class="ctl"><label>People to feed <b id="tv"></b></label>
   <input id="target" type="range" min="1" max="4.3" step="0.05" value="2.7"></div>
  <div class="ctl"><label>Diversity advantage <b id="dv"></b></label>
   <input id="divw" type="range" min="0" max="1" step="0.05" value="0.4"></div>
  <div class="toggles">
   <label><input type="checkbox" id="showHeat" checked> Score surface</label>
   <label><input type="checkbox" id="showParcels"> Crop parcels</label>
  </div>

  <div id="detail" class="card"><p class="muted">Click the map to score a location.</p></div>

  <details>
   <summary>How the score works</summary>
   <p>Each farm's acres are split among its crops and converted to an estimated yearly harvest of protein, carbohydrate and fat, expressed as <b>person-years</b>: the number of people that nutrient could fully cover for a year at FDA Daily Values (50 g protein, 275 g carbs, 78 g fat per day).</p>
   <p>Farms within the radius count toward a point, weighted by distance (a Gaussian falloff: a farm at half the radius counts ~61%, at the edge ~14%).</p>
   <p><b>Sufficiency</b> = 1 − e<sup>−min(P, C, F) / people</sup>. The scarcest macronutrient sets it, so a place rich in fat but short on protein scores low.</p>
   <p><b>Diversity</b> uses the effective number of foods <i>D</i> (exp of Shannon entropy over calorie shares). Factor = (1 − w) + w·(1 − 1/D).</p>
   <p>Score = 100 × sufficiency × diversity. Yields are rough regional defaults; calibrate them against the County Crop Report before drawing conclusions. Livestock, eggs, dairy and fish are not modeled.</p>
  </details>
 </aside>
 <div id="map"></div>
</div>

<script>
const DATA = __DATA__;
const PARCELS = __PARCELS__;
const RAMP = ["#cde2fb","#b7d3f6","#9ec5f4","#86b6ef","#6da7ec","#5598e7","#3987e5","#2a78d6","#256abf","#1c5cab","#184f95","#104281","#0d366b"];
const RGB = RAMP.map(h=>[1,3,5].map(i=>parseInt(h.slice(i,i+2),16)));
const MACROS = [["Protein",0],["Carbs",1],["Fat",2]];
function nice(v){const m=10**Math.floor(Math.log10(v)-1); return Math.round(v/m)*m;}
const state = {radius:3, target:500, divw:0.4};
const NF = DATA.foods.length;

// ---------- spatial bins so each query only checks nearby sites
const KY = 110.57;
let bins, binDeg;
function buildBins(){
  binDeg = state.radius / KY;
  bins = new Map();
  DATA.sites.forEach((s,i)=>{
    const k = Math.floor(s[0]/binDeg)+","+Math.floor(s[1]/binDeg);
    (bins.get(k) || bins.set(k,[]).get(k)).push(i);
  });
}
const foodAcc = new Float64Array(NF*4);

function evaluate(lat, lon, detail){
  const R = state.radius, s2 = 2*(R/2)**2, kx = 111.32*Math.cos(lat*Math.PI/180);
  foodAcc.fill(0);
  let P=0,C=0,F=0,K=0,nearest=Infinity,nSites=0;
  const bi = Math.floor(lat/binDeg), bj = Math.floor(lon/binDeg);
  for(let di=-1;di<=1;di++) for(let dj=-1;dj<=1;dj++){
    const b = bins.get((bi+di)+","+(bj+dj)); if(!b) continue;
    for(const i of b){
      const s = DATA.sites[i];
      const dy = (s[0]-lat)*KY, dx = (s[1]-lon)*kx, d2 = dx*dx+dy*dy;
      if(d2 > R*R) continue;
      const d = Math.sqrt(d2); if(d<nearest) nearest=d;
      const w = Math.exp(-d2/s2); nSites++;
      for(const f of s[2]){
        P+=w*f[1]; C+=w*f[2]; F+=w*f[3]; K+=w*f[4];
        const o=f[0]*4; foodAcc[o]+=w*f[1]; foodAcc[o+1]+=w*f[2]; foodAcc[o+2]+=w*f[3]; foodAcc[o+3]+=w*f[4];
      }
    }
  }
  if(K<=0) return detail ? {score:0,P,C,F,D:0,nFoods:0,nearest,nSites,foods:[]} : 0;
  let H=0,nFoods=0;
  for(let j=0;j<NF;j++){const k=foodAcc[j*4+3]; if(k>0){nFoods++; const p=k/K; H-=p*Math.log(p);}}
  const D = Math.exp(H);
  const suff = 1-Math.exp(-Math.min(P,C,F)/state.target);
  const div = (1-state.divw) + state.divw*(1-1/D);
  const score = 100*suff*div;
  if(!detail) return score;
  const foods=[];
  for(let j=0;j<NF;j++){const k=foodAcc[j*4+3]; if(k>0) foods.push({name:DATA.foods[j],share:k/K,p:foodAcc[j*4],c:foodAcc[j*4+1],f:foodAcc[j*4+2]});}
  foods.sort((a,b)=>b.share-a.share);
  return {score,suff,div,P,C,F,D,nFoods,nearest,nSites,foods};
}

// ---------- map
const map = L.map("map",{zoomControl:true}).setView([33.05,-116.85],10);
// Esri Light Gray Canvas: no API key needed, works when the file is opened from disk.
const ESRI = "https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/";
L.tileLayer(ESRI+"World_Light_Gray_Base/MapServer/tile/{z}/{y}/{x}",{
  maxZoom:18, maxNativeZoom:16,
  attribution:'Basemap &copy; Esri, HERE, Garmin, &copy; OpenStreetMap contributors | Farm data: County of San Diego AWM via SANDAG/SanGIS'}).addTo(map);
// Place-name labels drawn above the score surface so towns stay readable.
map.createPane("labels"); map.getPane("labels").style.zIndex = 650; map.getPane("labels").style.pointerEvents = "none";
L.tileLayer(ESRI+"World_Light_Gray_Reference/MapServer/tile/{z}/{y}/{x}",{
  maxZoom:18, maxNativeZoom:16, pane:"labels"}).addTo(map);

const lats = DATA.sites.map(s=>s[0]), lons = DATA.sites.map(s=>s[1]);
const pad = 0.15;
const B = {s:Math.min(...lats)-pad, n:Math.max(...lats)+pad, w:Math.min(...lons)-pad, e:Math.max(...lons)+pad};
map.fitBounds([[B.s,B.w],[B.n,B.e]]);

const CELL = 0.008; // ~0.9 km
const nr = Math.ceil((B.n-B.s)/CELL), nc = Math.ceil((B.e-B.w)/CELL);
let grid = new Float32Array(nr*nc), heat = null;

function drawGrid(){
  buildBins();
  const cv = document.createElement("canvas"); cv.width=nc; cv.height=nr;
  const ctx = cv.getContext("2d"), img = ctx.createImageData(nc,nr);
  for(let r=0;r<nr;r++){
    const lat = B.n - (r+0.5)*CELL;
    for(let c=0;c<nc;c++){
      const sc = evaluate(lat, B.w+(c+0.5)*CELL, false);
      grid[r*nc+c]=sc;
      if(sc<0.5) continue;
      const k = Math.min(RAMP.length-1, Math.floor(sc/100*RAMP.length)), o=(r*nc+c)*4;
      img.data[o]=RGB[k][0]; img.data[o+1]=RGB[k][1]; img.data[o+2]=RGB[k][2];
      img.data[o+3]=Math.round(140+ 90*Math.min(1,sc/60));
    }
  }
  ctx.putImageData(img,0,0);
  const url = cv.toDataURL();
  if(heat) heat.setUrl(url);
  else heat = L.imageOverlay(url,[[B.s,B.w],[B.n,B.e]],{opacity:0.75,interactive:false}).addTo(map);
}

// parcels
const parcels = L.geoJSON(PARCELS,{
  style:f=>{const c=DATA.colors[f.properties.cat]||"#b8b8b3"; return {color:c,weight:1,fillColor:c,fillOpacity:0.85};},
  onEachFeature:(f,l)=>l.bindTooltip(`<b>${f.properties.cat}</b><br>${f.properties.commodity||""}<br>${f.properties.calc_acres} ac`,{sticky:true})
});

// legend
const legend = L.control({position:"bottomright"});
legend.onAdd = ()=>{
  const d = L.DomUtil.create("div","legend");
  const cats = DATA.catOrder.map(c=>`<div class="cat"><span class="sw" style="background:${DATA.colors[c]}"></span>${c}</div>`).join("");
  d.innerHTML = `<b>Resilience score</b><div class="ramp" style="background:linear-gradient(90deg,${RAMP.join(",")})"></div>
    <div class="ramp-lbl"><span>0</span><span>50</span><span>100</span></div>
    <div id="catLegend" style="display:none;margin-top:8px"><b>Crop parcels</b>${cats}</div>`;
  return d;
};
legend.addTo(map);

// hover readout
const hover = L.control({position:"topright"});
hover.onAdd = ()=>{const d=L.DomUtil.create("div"); d.id="hover"; d.textContent="Hover the map"; return d;};
hover.addTo(map);
map.on("mousemove",e=>{
  const r=Math.floor((B.n-e.latlng.lat)/CELL), c=Math.floor((e.latlng.lng-B.w)/CELL);
  const v=(r>=0&&r<nr&&c>=0&&c<nc)?grid[r*nc+c]:0;
  document.getElementById("hover").textContent = `Score here ≈ ${v.toFixed(0)}  ·  click for detail`;
});

// click detail
let pin=null, ring=null, last=null;
map.on("click",e=>{last=e.latlng; showDetail();});
function fmt(n){return n>=100?Math.round(n).toLocaleString():n>=10?n.toFixed(0):n.toFixed(1);}
function showDetail(){
  if(!last) return;
  const {lat,lng}=last, r=evaluate(lat,lng,true);
  if(pin){pin.setLatLng(last); ring.setLatLng(last).setRadius(state.radius*1000);}
  else{ pin=L.circleMarker(last,{radius:6,color:"#fff",weight:2,fillColor:"#1a1a19",fillOpacity:1}).addTo(map);
        ring=L.circle(last,{radius:state.radius*1000,color:"#1a1a19",weight:1.5,dashArray:"4 4",fill:false,interactive:false}).addTo(map);}
  const el=document.getElementById("detail");
  if(!r.foods.length){
    el.innerHTML=`<div class="hero">0<small> / 100</small></div><p class="muted">No food-producing farms within ${state.radius} km.${isFinite(r.nearest)?"":""}</p>`;
    return;
  }
  const vals=[r.P,r.C,r.F], mn=Math.min(...vals), lim=vals.indexOf(mn);
  const scaleMax=Math.max(state.target*1.5,...vals);
  const rows=MACROS.map(([n,i])=>`<div class="row"><span>${n}${i===lim?'<span class="limit">limiting</span>':""}</span>
    <div class="track"><div class="fill" style="width:${100*vals[i]/scaleMax}%"></div><div class="tick" style="left:${100*state.target/scaleMax}%"></div></div>
    <span class="val">${fmt(vals[i])} ppl</span></div>`).join("");
  const top=r.foods.slice(0,8).map(f=>`<tr><td>${f.name}</td><td class="n">${(100*f.share).toFixed(0)}% kcal</td></tr>`).join("");
  el.innerHTML=`<div class="muted">${lat.toFixed(4)}, ${lng.toFixed(4)} · ${r.nSites} farm sites within ${state.radius} km</div>
   <div class="hero">${r.score.toFixed(0)}<small> / 100</small></div>
   <div class="muted">Sufficiency ${(100*r.suff).toFixed(0)}% × diversity ${(100*r.div).toFixed(0)}%</div>
   <p style="margin:14px 0 2px;font-size:13px"><b>People fully supplied for a year</b> <span class="muted">(tick = ${state.target.toLocaleString()} target)</span></p>
   ${rows}
   <p style="margin:14px 0 0;font-size:13px"><b>${r.nFoods} foods</b> · effective diversity D = ${r.D.toFixed(1)}</p>
   <table>${top}</table>`;
}

// controls
const $=id=>document.getElementById(id);
function syncLabels(){
  $("rv").textContent=state.radius+" km";
  $("tv").textContent=state.target.toLocaleString();
  $("dv").textContent=state.divw.toFixed(2);
}
let t=null;
function recompute(){clearTimeout(t); t=setTimeout(()=>{drawGrid(); showDetail();},60);}
$("radius").addEventListener("input",e=>{state.radius=+e.target.value; syncLabels(); recompute();});
$("target").addEventListener("input",e=>{state.target=nice(10**+e.target.value); syncLabels(); recompute();});
$("divw").addEventListener("input",e=>{state.divw=+e.target.value; syncLabels(); recompute();});
$("showHeat").addEventListener("change",e=>{e.target.checked?heat.addTo(map):heat.remove();});
$("showParcels").addEventListener("change",e=>{
  if(e.target.checked){parcels.addTo(map); if(heat) heat.setOpacity(0.45);}
  else{parcels.remove(); if(heat) heat.setOpacity(0.75);}
  $("catLegend").style.display=e.target.checked?"block":"none";
});
syncLabels(); drawGrid();
</script>
</body>
</html>
"""

if __name__ == "__main__":
    main()
