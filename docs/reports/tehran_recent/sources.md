# منابع و یادداشت‌های استخراج

بازیابی: 2026-09-12؛ خلاصهٔ حقایق، نه آرشیو کامل صفحات.

## dlearn
[D-learn: CBI monthly data, city rows (region=-1)](https://d-learn.ir/wp-content/uploads/2023/01/cbi_houseprice_tehran.csv)

روش دسترسی: `fetch_page`

Selected rows transcribed from returned CSV chunks 3 and 4, not a byte-for-byte full download. Scale changes at 1401/05: values become 100 times smaller. Conversion is an explicit analyst correction supported by cross-checks; the publisher does not document this change.

## dlearn_doc
[D-learn data dictionary](https://d-learn.ir/p/tehran-houseprice-data-cbi/)

روش دسترسی: `fetch_page`

region=-1 denotes Tehran city; raw source attributed to CBI.

## unit_140006
[CBI via IRIB: Shahrivar 1400](https://www.iribnews.ir/fa/news/3233813/%D9%85%D8%AA%D9%88%D8%B3%D8%B7-%D9%82%DB%8C%D9%85%D8%AA-%D9%85%D8%B3%DA%A9%D9%86-%D8%AF%D8%B1-%D8%AA%D9%87%D8%B1%D8%A7%D9%86-%D8%AF%D8%B1-%D8%B4%D9%87%D8%B1%DB%8C%D9%88%D8%B1-%DB%B1%DB%B4%DB%B0%DB%B0)

روش دسترسی: `web_search`

Reported city price: 31,700,000 toman; corroborates pre-change CSV magnitude, not every digit.

## unit_140108
[CBI via Tasnim/Sarpoosh: Aban 1401](https://www.sarpoosh.com/economy/housing-construction/housing-construction1401091169.html)

روش دسترسی: `web_search`

467 million IRR (46.7 million toman); corroborates corrected post-change CSV scale, not every digit.

## sarmayeh_02
[CBI via Sarmayesazan: Ordibehesht 1403](https://sarmayesazan.com/blog/%D8%AA%D8%BA%DB%8C%DB%8C%D8%B1%D8%A7%D8%AA-%D8%A8%D8%A7%D8%B2%D8%A7%D8%B1-%D9%85%D8%B9%D8%A7%D9%85%D9%84%D8%A7%D8%AA-%D9%85%D8%B3%DA%A9%D9%86-%D8%B4%D9%87%D8%B1-%D8%AA%D9%87%D8%B1%D8%A7%D9%86-%D8%AF/)

روش دسترسی: `fetch_page`

Only price table used. Age/transaction tables elsewhere on the page contain inconsistent totals and are excluded.

## sarmayeh_03
[CBI via Sarmayesazan: Khordad 1403](https://sarmayesazan.com/blog/%D9%85%D8%B9%D8%A7%D9%85%D9%84%D8%A7%D8%AA-%D9%85%D8%B3%DA%A9%D9%86-%D8%AA%D9%87%D8%B1%D8%A7%D9%86-%D8%AE%D8%B1%D8%AF%D8%A7%D8%AF-1403/)

روش دسترسی: `fetch_page`

Use numeric price table (859.1), not prose typo 1.859. June 1403 cross-checked with Fardaye Eghtesad.

## sarmayeh_04
[CBI via Sarmayesazan: Tir 1403](https://sarmayesazan.com/blog/%DA%AF%D8%B2%D8%A7%D8%B1%D8%B4-%D9%85%D8%B9%D8%A7%D9%85%D9%84%D8%A7%D8%AA-%D9%85%D8%B3%DA%A9%D9%86-%D8%AA%D9%87%D8%B1%D8%A7%D9%86-%D8%AA%DB%8C%D8%B1-1403-%D8%AA%D8%AD%D9%84%DB%8C%D9%84-%D8%AC%D8%A7/)

روش دسترسی: `web_search`

Price comparison table only.

## sarmayeh_05
[CBI via Sarmayesazan: Mordad 1403](https://sarmayesazan.com/blog/%D9%85%D8%B9%D8%A7%D9%85%D9%84%D8%A7%D8%AA-%D9%85%D8%B3%DA%A9%D9%86-%D8%AA%D9%87%D8%B1%D8%A7%D9%86-%D9%85%D8%B1%D8%AF%D8%A7%D8%AF-1403/)

روش دسترسی: `fetch_page`

Price comparison table only; age table has inconsistent counts and is excluded.

## farda_03
[CBI via Fardaye Eghtesad: Khordad 1403](https://www.fardayeeghtesad.com/news/39676/%D9%85%D8%AA%D9%88%D8%B3%D8%B7-%D9%82%DB%8C%D9%85%D8%AA-%D9%85%D8%B3%DA%A9%D9%86-%D8%AF%D8%B1-%D8%AA%D9%87%D8%B1%D8%A7%D9%86-%D8%A7%D8%B9%D9%84%D8%A7%D9%85-%D8%B4%D8%AF)

روش دسترسی: `fetch_page`

Body explicitly says Khordad 1403: 85,910,000 toman. Lead incorrectly names Ordibehesht; body/date used.

## tejarat_h2
[CBI via Tejaratnews: second half of 1402](https://tejaratnews.com/%D8%A8%D8%AE%D8%B4-%D9%85%D8%B3%DA%A9%D9%86-48/904124-%D9%82%DB%8C%D9%85%D8%AA-%D9%85%D8%B3%DA%A9%D9%86-%D8%AF%D8%B1-%D9%86%DB%8C%D9%85%D9%87-%D8%AF%D9%88%D9%85-%D8%B3%D8%A7%D9%84)

روش دسترسی: `fetch_page`

Use explicit Shahrivar/Mehr/Dey/Esfand values only. Aban/Azar prose is approximate or inconsistent; independent reports used instead. Bahman only states a price band; not converted to an exact observation.

## otagh_08
[CBI via Iran Chamber: Aban 1402](https://otaghiranonline.ir/news/68538/%D8%AA%D9%88%D8%B1%D9%85-%D9%86%D9%82%D8%B7%D9%87-%D8%A8%D9%87-%D9%86%D9%82%D8%B7%D9%87-%D9%85%D8%B3%DA%A9%D9%86-%D8%AA%D9%87%D8%B1%D8%A7%D9%86-%D8%A8%D9%87-62-2-%D8%AF%D8%B1%D8%B5%D8%AF-%D8%B1%D8%B3%DB%8C%D8%AF)

روش دسترسی: `web_search`

Explicit 75,770,000 toman.

## tasnim_09
[CBI via Tasnim: Azar 1402](https://www.tasnimnews.com/fa/news/1402/10/03/3011261/%DA%A9%D8%A7%D9%87%D8%B4-2-2-%D8%AF%D8%B1%D8%B5%D8%AF%DB%8C-%D9%82%DB%8C%D9%85%D8%AA-%D9%85%D8%B3%DA%A9%D9%86-%D8%AF%D8%B1-%D8%A2%D8%B0%D8%B1%D9%85%D8%A7%D9%87-%D9%85%D8%AA%D9%88%D8%B3%D8%B7-%D9%82%DB%8C%D9%85%D8%AA-%D9%85%D8%B3%DA%A9%D9%86-%D8%AF%D8%B1-%D8%AA%D9%87%D8%B1%D8%A7%D9%86-74-%D9%85%DB%8C%D9%84%DB%8C%D9%88%D9%86-%D8%AA%D9%88%D9%85%D8%A7%D9%86-%D8%B4%D8%AF)

روش دسترسی: `web_search`

740.9 million IRR. Rounding differs from Ecoiran 74,088,000 toman by 2,000 toman; do not claim precision beyond source.

## kilid
[Kilid: displayed monthly city price trend](https://kilid.com/house-prices/tehran)

روش دسترسی: `fetch_page`

The current page describes listing-based prices and estimated typical-home price. Trend methodology/history revisions are not fully documented. Treat as published platform indicator, not verified transactions or raw listing observations. Monthly table disagrees with headline MoM (210/195 - 1 = 7.69%, headline 2.2%); only table used. Query period=60 still returned just 12 months.
