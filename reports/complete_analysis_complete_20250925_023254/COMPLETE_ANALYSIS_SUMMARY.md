# Complete Analysis Summary - complete_20250925_023254

## Overview
Complete end-to-end analysis run completed on Thu Sep 25 02:37:46 UTC 2025

## Data Generation
- **State shifts**: 8 (gradient-optimized)
- **State+action shifts**: 4 (gradient-optimized) 
- **Policy shifts**: 0 (gradient-optimized)
- **Total splits**: 12

## Model Training
- **Models trained**: 4/5
- ✅ cvae_pid_complete_20250925_023254
- ✅ cvae_reg_complete_20250925_023254
- ✅ gru_complete_20250925_023254
- ✅ trans_cvae_complete_20250925_023254

## Performance Analysis
- **Detailed plots**: 22/36 expected
- **Three-shifts plots**: 7 expected
- **Task B analysis**: ✅ Completed

## Results Location
- **Data**: `data/complete_analysis_complete_20250925_023254`
- **Models**: `reports/complete_analysis_complete_20250925_023254/models`
- **Evaluation**: `reports/complete_analysis_complete_20250925_023254/evaluation`
- **Analysis**: `reports/complete_analysis_complete_20250925_023254/analysis`
- **Logs**: `logs/*_complete_20250925_023254.log`

## Key Plots
### Detailed Performance Analysis
- Degradation curves: `reports/complete_analysis_complete_20250925_023254/analysis/detailed_performance/degradation_curves/`
- Task comparisons: `reports/complete_analysis_complete_20250925_023254/analysis/detailed_performance/task_comparisons/`
- Robustness ranking: `reports/complete_analysis_complete_20250925_023254/analysis/detailed_performance/robustness_ranking/`

### Three Distribution Shifts Analysis
- Specialized analysis: `reports/complete_analysis_complete_20250925_023254/analysis/three_shifts/`

## Usage
View results by opening the PNG files in the analysis directories or running:
```bash
# View all plots
find reports/complete_analysis_complete_20250925_023254/analysis -name "*.png" | head -10

# View summary data
ls reports/complete_analysis_complete_20250925_023254/evaluation/
```
