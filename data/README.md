# AG News Classification Dataset

This directory contains the AG News classification dataset partitioned into training and test splits.

## Files

| File | Records | Description | Size |
| :--- | :--- | :--- | :--- |
| `train.csv` | 120,000 | Training set containing balanced articles across 4 classes | ~29.1 MB |
| `test.csv` | 7,600 | Test evaluation set (1,900 articles per class) | ~1.8 MB |

## Schema

Both files follow the standard CSV format with 3 columns:
1. `Class Index` (Integer: 1 to 4)
2. `Title` (String): Headline of the news article
3. `Description` (String): News summary / teaser body

## Category Mapping

- `1`: **World**
- `2`: **Sports**
- `3`: **Business**
- `4`: **Sci/Tech**
