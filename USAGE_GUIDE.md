# E8 Text Processing - Usage Guide

## Quick Start

### Run Complete Analysis
```bash
python main.py
```
This generates:
- Comprehensive text statistics
- Topic analysis
- Word frequency analysis
- Temporal patterns
- Student participation metrics
- Visualizations saved as `text_analysis_results.png`

### Run Quick Analysis Demo
```bash
python quick_analysis.py
```
This shows example usage of the interactive analyzer.

## Available Scripts

### 1. `main.py` - Complete Analysis
The main comprehensive analysis script that:
- Loads and analyzes the entire dataset
- Generates statistical summaries
- Creates visualizations
- Performs keyword searches
- Exports analysis results

**Output:**
- Console: Detailed statistics and insights
- File: `text_analysis_results.png` (4 visualization plots)

### 2. `quick_analysis.py` - Interactive Analyzer
Easy-to-use class-based analyzer for custom queries.

**Example Usage:**
```python
from quick_analysis import SessionAnalyzer

# Initialize
analyzer = SessionAnalyzer()

# Get student summary
print(analyzer.student_summary(127))

# Search topics
print(analyzer.topic_search('regression'))

# Search content
print(analyzer.content_search('neural network'))

# Top students
print(analyzer.top_students(10))

# Compare keywords
print(analyzer.keyword_compare('regression', 'classification', 'clustering'))

# Export filtered data
analyzer.export_filtered_data('machine learning', 'ml_entries.csv')
```

### 3. `advanced_analysis.py` - Deep Dive Functions
Advanced analytical functions for detailed exploration:

**Available Functions:**
- `analyze_specific_topic(df, topic_keyword)` - Deep dive into specific topics
- `compare_students(df, roll_numbers)` - Compare multiple students
- `extract_code_snippets(df)` - Find entries with code
- `topic_word_association(df)` - Topic-keyword relationships
- `analyze_response_quality(df)` - Quality metrics
- `time_series_analysis(df)` - Temporal trends
- `generate_student_report(df, roll_no)` - Individual student report

**Example Usage:**
```python
from advanced_analysis import *

df = load_data()

# Analyze regression topics
regression_df = analyze_specific_topic(df, 'regression')

# Compare top 3 students
compare_students(df, [127, 138, 104])

# Find code snippets
code_df = extract_code_snippets(df)

# Get detailed student report
report = generate_student_report(df, 127)
```

## Common Analysis Tasks

### Task 1: Find All Entries About a Topic
```python
from quick_analysis import SessionAnalyzer

analyzer = SessionAnalyzer()
print(analyzer.topic_search('decision tree'))
```

### Task 2: Analyze a Specific Student
```python
from quick_analysis import SessionAnalyzer

analyzer = SessionAnalyzer()
print(analyzer.student_summary(YOUR_ROLL_NUMBER))
```

### Task 3: Compare Keyword Frequencies
```python
from quick_analysis import SessionAnalyzer

analyzer = SessionAnalyzer()
print(analyzer.keyword_compare('supervised', 'unsupervised', 'reinforcement'))
```

### Task 4: Export Filtered Dataset
```python
from quick_analysis import SessionAnalyzer

analyzer = SessionAnalyzer()
analyzer.export_filtered_data('SQL', 'sql_entries.csv')
```

### Task 5: Generate Student Report
```python
from advanced_analysis import generate_student_report, load_data

df = load_data()
report = generate_student_report(df, 127)
```

## Custom Analysis

### Create Your Own Analysis Script

```python
import pandas as pd
from collections import Counter
import re

# Load data
df = pd.read_csv('Session-Summary-all-2025-S1.csv')

# Example: Find longest responses
df['word_count'] = df['YourAnalysis'].fillna('').apply(lambda x: len(str(x).split()))
top_longest = df.nlargest(10, 'word_count')[['RollNo', 'Topic', 'word_count']]
print(top_longest)

# Example: Count specific term
sql_count = df['YourAnalysis'].fillna('').str.lower().str.contains('sql').sum()
print(f"SQL mentioned in {sql_count} entries")

# Example: Student activity over time
df['datetime'] = pd.to_datetime(df['Timestamp'], format='%d-%m-%y %H:%M', errors='coerce')
df['date'] = df['datetime'].dt.date
daily_activity = df.groupby('date').size()
print(daily_activity)
```

## Visualizations

The main analysis generates 4 plots:

1. **Top 15 Topics Bar Chart**
   - Shows most discussed topics
   - Helps identify curriculum focus areas

2. **Word Count Distribution Histogram**
   - Shows distribution of response lengths
   - Includes mean marker

3. **Top 20 Words Bar Chart**
   - Most frequent technical terms
   - Filtered for stop words

4. **Hourly Activity Line Plot**
   - Submission patterns by hour
   - Shows peak activity times

## Data Structure

### CSV Columns:
- `Timestamp` - Entry date/time (format: DD-MM-YY HH:MM)
- `RollNo` - Student roll number
- `Topic` - Topic of the session/analysis
- `YourAnalysis` - Student's written analysis (main text content)

### Derived Metrics:
- `word_count` - Number of words in analysis
- `datetime` - Parsed timestamp
- `date` - Date only
- `hour` - Hour of submission

## Tips for Analysis

### 1. Working with Large Text
```python
# Read specific entries
df.iloc[0:10]  # First 10 entries
df.sample(5)   # Random 5 entries

# Filter by criteria
long_entries = df[df['word_count'] > 500]
recent_entries = df[df['date'] > '2025-02-01']
```

### 2. Text Preprocessing
```python
# Lowercase
text = df['YourAnalysis'].str.lower()

# Remove punctuation
text = df['YourAnalysis'].str.replace(r'[^\w\s]', '', regex=True)

# Extract words
words = df['YourAnalysis'].str.findall(r'\b\w+\b')
```

### 3. Frequency Analysis
```python
from collections import Counter

# Get all words
all_text = ' '.join(df['YourAnalysis'].fillna(''))
words = all_text.lower().split()

# Count frequencies
word_counts = Counter(words)
top_20 = word_counts.most_common(20)
```

### 4. Filtering Data
```python
# By student
student_data = df[df['RollNo'] == 127]

# By topic keyword
mask = df['Topic'].str.contains('regression', case=False, na=False)
regression_data = df[mask]

# By content keyword
mask = df['YourAnalysis'].str.contains('neural', case=False, na=False)
neural_data = df[mask]
```

## Troubleshooting

### Issue: Module not found
```bash
pip install pandas matplotlib seaborn
```

### Issue: Date parsing errors
The script handles date parsing errors automatically with `errors='coerce'`

### Issue: Empty results
- Check keyword spelling
- Try case-insensitive search
- Use partial matches

### Issue: Memory concerns
For large datasets:
```python
# Process in chunks
chunks = pd.read_csv('file.csv', chunksize=1000)
for chunk in chunks:
    # Process each chunk
    pass
```

## Additional Resources

- **analysis_summary.md** - Detailed findings and insights
- **README.md** - Project overview and key statistics
- **text_analysis_results.png** - Visual analysis dashboard

## Questions?

Common queries answered:
- How many students participated? → 198
- How many entries total? → 2,548
- Most discussed topic? → SQL (16 entries)
- Average response length? → 305.7 words
- Most active student? → Roll No 127 & 138 (20 entries each)

---

Happy Analyzing! 🎉
