# Complete Analysis Summary - complete_20250925_002121

## Overview
Complete end-to-end analysis run completed on Thu Sep 25 00:26:36 UTC 2025

## Data Generation
- **State shifts**: 8 (gradient-optimized)
- **State+action shifts**: 4 (gradient-optimized) 
- **Policy shifts**: 4 (gradient-optimized)
- **Total splits**: 16

## Model Training
- **Models trained**: 5/5
- ✅ cvae_pid_complete_20250925_002121
- ✅ cvae_repr_complete_20250925_002121
- ✅ cvae_reg_complete_20250925_002121
- ✅ gru_complete_20250925_002121
- ✅ trans_cvae_complete_20250925_002121

## Performance Analysis
- **Detailed plots**: 0/36 expected
- **Three-shifts plots**: 0 expected
- **Task B analysis**: ❌ Failed

## Results Location
- **Data**: `data/complete_analysis_complete_20250925_002121`
- **Models**: `reports/complete_analysis_complete_20250925_002121/models`
- **Evaluation**: `reports/complete_analysis_complete_20250925_002121/evaluation`
- **Analysis**: `reports/complete_analysis_complete_20250925_002121/analysis`
- **Logs**: `logs/*_complete_20250925_002121.log`

## Key Plots
### Detailed Performance Analysis
- Degradation curves: `reports/complete_analysis_complete_20250925_002121/analysis/detailed_performance/degradation_curves/`
- Task comparisons: `reports/complete_analysis_complete_20250925_002121/analysis/detailed_performance/task_comparisons/`
- Robustness ranking: `reports/complete_analysis_complete_20250925_002121/analysis/detailed_performance/robustness_ranking/`

### Three Distribution Shifts Analysis
- Specialized analysis: `reports/complete_analysis_complete_20250925_002121/analysis/three_shifts/`

## Usage
View results by opening the PNG files in the analysis directories or running:
```bash
# View all plots
find reports/complete_analysis_complete_20250925_002121/analysis -name "*.png" | head -10

# View summary data
ls reports/complete_analysis_complete_20250925_002121/evaluation/
```
