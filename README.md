# San Diego Agricultural Micro-resiliency Map

**[Open the interactive map →](https://benh83.github.io/agricultural-micro-resiliency/)**

How far would you have to travel to gather one balanced day of local food? Click any point in
San Diego County. The map plans the shortest errand loop from that spot to farms, ranches and fishing
spots that together cover every food group in the 2025–2030 Dietary Guidelines ("the New Pyramid"):
protein foods, dairy, vegetables, fruits, healthy fats and whole grains.

For the same point, it also shows which foods are produced within an adjustable radius, how much, and
how many people that could feed for each food group.

## What the map shows

- **Trip distance (main layer).** The length of the shortest loop home → sources → home that covers
  every food group. One stop can cover several groups; for example, an avocado farm covers fruits and
  healthy fats. Switch to **Separate trips** to see the total of individual round trips to the nearest
  source of each group instead.
- **Route.** Clicking a point draws the route with numbered stops, what each stop provides, and each leg's length.
- **Supply in radius.** Foods and yearly volumes within the radius, and the people fed per food group.
- **Toggles.** Include fishing, require whole grains, count oat fields as grain, include farmers'-market
  growers whose crops are unknown, show food sources, show farm parcels.

## Model

| Step | Method |
|---|---|
| Food groups | 2025–2030 Dietary Guidelines. Daily needs used: 100 g protein from protein foods (1.2–1.6 g/kg), 3 cups dairy (735 g), 3 servings vegetables (240 g), 2 servings fruit (240 g), whole grains 85 g dry, 30 g fat from fat-rich foods. Serving sizes are assumptions. |
| Mixed farms | Each site's acres are split among its listed crops, then calibrated ("raked") so that county totals match the 2017 County Crop Report acreage for each crop. Buildings and landscaping get a small share. |
| Crop output | Acres × yield (Crop Report production per acre where available) × edible share. |
| Meat, milk, eggs | County Crop Report totals spread over sites showing animals: beef, lamb and pork on forage, pasture and oat-hay sites; milk on irrigated forage sites; eggs and chicken on egg sites plus certified egg and poultry producers. **This is a rough placement.** |
| Fishing | Piers, sportfishing landings, Tuna Harbor, stocked lakes, a shellfish farm, aquaponics farms, and open-coast shore points every 2 km (excluding Camp Pendleton), with rough yearly catch potential. |
| Trip | Uses the 6 nearest sources of each group; an exact subset dynamic program finds the shortest covering loop. Distances are straight-line × 1.3 for roads. A source counts for a group if it produces at least a month of that group for one person per year. Trip score = 100 × ½^(km ÷ 50). |
| Supply score | Per group: 1 − e^(−people fed ÷ target), averaged over the groups, then multiplied by a diversity factor from the effective number of foods (by calories). Closer farms are weighted more. |

### What the data says

- **Whole grains** are the bottleneck. Almost no grain is grown in the county; the main sources are
  potato-and-grain farms near Warner Springs and Borrego. Turn off "Require whole grains" to see how much
  shorter trips become.
- **Dairy** comes from a few irrigated-forage sites in the San Pasqual and Ramona valleys, and milk is placed
  there by assumption.
- **404 polygons** in the County layer are exact duplicates; they are removed, which takes away about 4,800 double-counted acres.

### Limitations

- Seasonality, prices, whether a farm sells directly, backyard food and imports are not modeled.
- Livestock and fishing amounts are estimates. Edit `LIVESTOCK` and `FISHING` in `build_resilience_map.py`.
- Crop data are from 2020 and calibration targets from 2017.
- Each farm is treated as a single point.

## Data

| File | Source |
|---|---|
| `data/Agricultural_Commodity_2020.geojson` | County of San Diego Dept. of Agriculture, Weights & Measures, via the [SANDAG/SanGIS Regional Data Warehouse](https://sdgis-sandag.opendata.arcgis.com/) |
| `data/AWM_Certified_Producers.geojson` | Same source |
| `ref/sd_county.geojson` | County outline from [plotly/datasets](https://github.com/plotly/datasets) (US Census cartographic boundary) |
| `ref/sd_coast.geojson` | Coastline from [Natural Earth](https://github.com/nvkelso/natural-earth-vector) 10 m |
| Calibration and livestock totals | [San Diego County 2017 Crop Statistics & Annual Report](https://www.sandiegocounty.gov/content/dam/sdc/awm/docs/AWM_2017_Crop_Report.pdf) |
| Food groups | [Dietary Guidelines for Americans 2025–2030](https://cdn.realfood.gov/DGA.pdf) |

The basemap is Esri World Light Gray Canvas.

## Rebuild the map

```bash
pip install -r requirements.txt
python build_resilience_map.py   # writes docs/index.html (the live map) and CSVs in output/
```

Then commit and push `docs/index.html`. GitHub Pages updates within a minute or two.

The build writes these CSVs to `output/` for checking:

- `site_allocation.csv`: acres per crop per site
- `sources.csv`: every food source with kg/yr and person-days per group
- `allocation_check.csv`: allocated acres vs. Crop Report
- `unmodeled_commodities.csv`: commodity names with no food rule

## Project layout

```
data/                     raw GeoJSON from SANDAG/SanGIS
ref/                      county outline and coastline
docs/index.html           the published map (GitHub Pages serves this folder)
build_resilience_map.py   food model: allocation, livestock, fishing, sources
map_template.html         the map page (trip solver, supply, interface)
explore_ag.py             optional: data profiling
categorize_ag.py          optional: crop categories and summary charts
output/                   local outputs (CSVs), not committed
```
