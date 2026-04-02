# Adjusted Sheet Total Error Kruskal-Wallis Analysis

- Workbook: master_error.xlsx
- Sheet analyzed: adjusted_position
- Outcome analyzed: total_error
- Rows analyzed: 18
- Factors analyzed: side, collar, arc, operator
- Significance threshold: alpha = 0.05

## Normality check

Overall `total_error` distribution was non-normal by Shapiro-Wilk (W = 0.896, p = 0.0487, n = 18).

### Group-level normality summary

| Factor | Groups evaluated | Groups tested | Non-normal groups | Insufficient n | All tested groups normal |
| --- | ---: | ---: | ---: | ---: | --- |
| arc | 4 | 4 | 0 | 0 | Yes |
| collar | 2 | 2 | 1 | 0 | No |
| operator | 3 | 3 | 0 | 0 | Yes |
| side | 2 | 2 | 0 | 0 | Yes |

## Significant omnibus findings

No factor showed a significant Kruskal-Wallis difference in `total_error` at alpha = 0.05.

## Interpretation

Because overall `total_error` normality was not supported and no grouping factor reached Kruskal-Wallis significance, the current adjusted-position data do not show evidence that `total_error` differs across the tested columns at alpha = 0.05.
