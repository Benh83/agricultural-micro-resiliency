# San Diego Micro Food-Resilience Map

**[Open the interactive map →](https://benh83.github.io/agricultural-micro-resiliency/)**

For any point in San Diego County, this map estimates how well the farms nearby could supply a
**balanced and diverse diet**: enough protein, carbohydrate *and* fat, drawn from many different foods,
as close as possible.

Click anywhere on the map to see:

- the resilience score (0–100)
- how many people the nearby harvest could fully supply for a year with protein, carbs and fat, with the limiting nutrient flagged
- which foods make up the local supply, and how diverse it is

Sliders adjust the search radius, the number of people to feed, and how much weight crop diversity gets.

## How the score works

1. **Nutrients per farm.** Each grower site's acres are split among its listed crops. Each crop is
   converted to an estimated yearly harvest (yield per acre × edible share) and then to kilograms of
   protein, carbs and fat. That is expressed as **person-years** at FDA Daily Values
   (50 g protein, 275 g carbs, 78 g fat per day).
2. **Distance.** For a point, every farm within the radius counts, with a Gaussian distance weight so that
   closer farms count more.
3. **Balance.** Sufficiency = 1 − e^(−min(P, C, F) / people). The scarcest nutrient sets the score.
4. **Diversity.** *D* = the effective number of foods (exp of Shannon entropy over calorie shares).
   Diversity factor = (1 − w) + w·(1 − 1/D), where *w* is the "diversity advantage" slider.
5. **Score** = 100 × sufficiency × diversity.

### Limitations

- Yields are rough regional defaults, not measured values. Calibrate the `FOODS` table in
  `build_resilience_map.py` against the San Diego County Crop Statistics & Annual Report.
- Livestock, eggs, dairy and fish are not modeled.
- Nursery, flowers, turf, forage and wine grapes count as zero food.
- Each farm is treated as a single point at its center.
- The data are from 2020.

## Data

| File | Source |
|---|---|
| `data/Agricultural_Commodity_2020.geojson` | County of San Diego Dept. of Agriculture, Weights & Measures, via the [SANDAG/SanGIS Regional Data Warehouse](https://sdgis-sandag.opendata.arcgis.com/) |
| `data/AWM_Certified_Producers.geojson` | Same source |

The basemap is Esri World Light Gray Canvas.

## Rebuild the map

```bash
pip install -r requirements.txt
python explore_ag.py            # optional: profile the raw data
python categorize_ag.py         # optional: crop categories, summary charts
python build_resilience_map.py  # writes docs/index.html (the live map)
```

Then commit and push `docs/index.html`. GitHub Pages updates within a minute or two.

## Project layout

```
data/                     raw GeoJSON from SANDAG/SanGIS
docs/index.html           the published map (GitHub Pages serves this folder)
explore_ag.py             data profiling
categorize_ag.py          crop categories and summary charts
build_resilience_map.py   nutrient model and interactive map
output/                   local outputs (CSVs, PNGs), not committed
```
