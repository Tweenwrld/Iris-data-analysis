# Iris Dataset Analysis Project

## Overview
This project performs comprehensive data analysis on the classic Iris dataset, demonstrating data exploration, statistical analysis, and visualization techniques using Python. The analysis includes data cleaning, statistical summaries, and various visualization methods to understand patterns and relationships within the dataset.

## Dataset Description
The Iris dataset contains measurements from 150 iris flowers representing three species:
- Iris Setosa
- Iris Versicolor
- Iris Virginica

For each flower, four features were measured:
- Sepal length (cm)
- Sepal width (cm)
- Petal length (cm)
- Petal width (cm)

## Tasks Completed

### 1. Data Loading and Exploration
- Loaded the Iris dataset using scikit-learn
- Explored data structure and types
- Handled missing values (simulated for demonstration)
- Examined the distribution of features across species

### 2. Statistical Analysis
- Computed descriptive statistics for all features
- Performed group analysis by species
- Identified key patterns and relationships in the data
- Analyzed correlations between different measurements

### 3. Data Visualization
- **Line Chart**: Shows measurement trends when sorted by petal length
- **Bar Chart**: Compares average measurements across species
- **Histograms**: Displays distribution of measurements by species
- **Scatter Plots**: Reveals relationships between feature pairs
- **Pair Plot**: Shows all pairwise relationships simultaneously
- **Box Plots**: Illustrates statistical distributions of measurements by species

## Key Findings
- Iris setosa is clearly distinguishable from other species based on petal dimensions
- Petal length and width show the strongest correlation
- Sepal width shows more overlap between species than other features
- The three species form distinct clusters in the feature space
- Virginica species has the largest sepal length and petal dimensions
- Setosa species has the smallest petal length and width

## Requirements
- Python 3.x
- Libraries:
  - pandas (data manipulation)
  - numpy (numerical operations)
  - matplotlib (basic plotting)
  - seaborn (enhanced visualization)
  - scikit-learn (dataset loading)

## Installation

```bash
# Create and activate a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install required packages
pip install pandas numpy matplotlib seaborn scikit-learn
```

## Usage

Run the analysis with:
```bash
python iris_analysis.py
```

The script will:
1. Display analysis results in the console
2. Show visualizations in separate windows (close each window to proceed to the next)
3. Save all visualizations as PNG files in the current directory

## Visualization Output Files
- `iris_line_chart.png`: Measurements sorted by petal length
- `iris_bar_chart.png`: Average measurements by species
- `iris_histograms.png`: Distribution of each measurement by species
- `iris_scatter_plots.png`: Relationships between measurement pairs
- `iris_pair_plot.png`: All pairwise relationships between variables
- `iris_box_plots.png`: Statistical distributions by species

## Future Work
- Apply machine learning classification algorithms
- Explore additional visualization techniques
- Perform principal component analysis (PCA)
- Compare Iris dataset with other flower datasets

## License
This project is open source and available under the [MIT License](LICENSE).

## Acknowledgements
- R.A. Fisher for the original Iris dataset
- The scikit-learn team for making the dataset easily accessible