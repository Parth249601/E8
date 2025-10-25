# E8 Text Processing Assignment

## Overview
This project analyzes student session summaries from a data science course. The dataset contains student responses about various topics including machine learning, statistics, and data analysis concepts.

## Dataset
- **File**: `Session-Summary-all-2025-S1.csv`
- **Total Entries**: 2,548 student responses
- **Students**: 198 unique students
- **Topics**: 2,131 unique topics covered
- **Average Response Length**: 305.7 words

## Analysis Performed

### 1. Basic Text Statistics
- Word count analysis (mean, median, min, max)
- Total entries and unique contributors
- Topic distribution

### 2. Topic Analysis
Top 10 Most Discussed Topics:
1. SQL (16 entries)
2. Advantages and limitations of CART, BAGGING (15 entries)
3. Levels of Measurement (14 entries)
4. Logistic Regression (13 entries)
5. Feature Encoding (11 entries)

### 3. Word Frequency Analysis
Most frequent technical terms identified:
- **data** (8,742 occurrences)
- **model** (5,278 occurrences)
- **regression** (2,896 occurrences)
- **values** (2,514 occurrences)
- **error** (2,376 occurrences)

### 4. Temporal Analysis
- Analyzed submission patterns by date and hour
- Peak activity identification

### 5. Student Participation Analysis
- Most active students identified
- Participation distribution:
  - 152 students with >10 entries
  - 18 students with 6-10 entries
  - 19 students with 2-5 entries
  - 9 students with 1 entry

### 6. Keyword Search Functionality
- Search for specific terms (e.g., 'regression', 'classification')
- 1,041 entries mention regression
- 595 entries mention classification

## Visualizations Created

The script generates `text_analysis_results.png` with 4 plots:
1. **Top 15 Topics Bar Chart** - Most discussed topics
2. **Word Count Distribution** - Histogram showing response lengths
3. **Top 20 Words Bar Chart** - Most frequent words
4. **Hourly Activity Plot** - Submission patterns by hour

## Running the Analysis

### Prerequisites
```bash
pip install pandas matplotlib seaborn
```

### Execute
```bash
python main.py
```

## Key Insights

1. **High Engagement**: 152 out of 198 students (76.8%) contributed more than 10 responses
2. **Rich Content**: Average response length of 305.7 words indicates detailed analyses
3. **ML Focus**: Top words reflect machine learning concepts (model, regression, classification, features)
4. **Core Topics**: SQL, CART/Bagging, and regression methods are most discussed
5. **Statistical Depth**: High frequency of terms like variance, mean, distribution shows statistical rigor

## Functions Available

- `load_data()` - Load and preview the CSV dataset
- `basic_text_stats()` - Calculate word count and basic statistics
- `analyze_topics()` - Show most discussed topics
- `word_frequency_analysis()` - Count word frequencies (excluding stop words)
- `analyze_timestamps()` - Temporal pattern analysis
- `student_analysis()` - Participation metrics
- `search_keyword()` - Search for specific terms
- `create_visualizations()` - Generate analysis plots

## Customization

You can modify the analysis by:
- Changing the number of top words displayed: `word_frequency_analysis(df, top_n=50)`
- Searching for different keywords: `search_keyword(df, 'neural network')`
- Adjusting visualization parameters in `create_visualizations()`

## Files Generated

1. `text_analysis_results.png` - Visualization dashboard

## Author
Analysis created for E8 Text Processing assignment

## Date
October 2025
