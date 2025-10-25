"""
Interactive Text Analysis Tool for E8 Assignment
Quick analysis utilities with simple function calls
"""

import pandas as pd
import re
from collections import Counter

class SessionAnalyzer:
    """Easy-to-use class for analyzing session summary data"""
    
    def __init__(self, filepath='Session-Summary-all-2025-S1.csv'):
        """Initialize with dataset"""
        self.df = pd.read_csv(filepath)
        self.df['word_count'] = self.df['YourAnalysis'].fillna('').apply(
            lambda x: len(str(x).split())
        )
        print(f"✓ Loaded {len(self.df)} entries from {self.df['RollNo'].nunique()} students")
    
    def student_summary(self, roll_no):
        """Get quick summary for a student"""
        student_data = self.df[self.df['RollNo'] == roll_no]
        if len(student_data) == 0:
            return f"No data found for Roll No {roll_no}"
        
        return f"""
Student Roll No: {roll_no}
Total Entries: {len(student_data)}
Unique Topics: {student_data['Topic'].nunique()}
Total Words: {student_data['word_count'].sum()}
Average Words: {student_data['word_count'].mean():.1f}
Shortest Entry: {student_data['word_count'].min()} words
Longest Entry: {student_data['word_count'].max()} words
"""
    
    def topic_search(self, keyword):
        """Search topics containing keyword"""
        mask = self.df['Topic'].fillna('').str.lower().str.contains(keyword.lower())
        results = self.df[mask]
        
        if len(results) == 0:
            return f"No topics found containing '{keyword}'"
        
        output = f"\nFound {len(results)} entries with '{keyword}' in topic:\n"
        output += "\nTop matching topics:\n"
        for i, topic in enumerate(results['Topic'].value_counts().head(5).index, 1):
            count = len(results[results['Topic'] == topic])
            output += f"{i}. {topic} ({count} entries)\n"
        
        return output
    
    def content_search(self, keyword):
        """Search analysis content containing keyword"""
        mask = self.df['YourAnalysis'].fillna('').str.lower().str.contains(keyword.lower())
        results = self.df[mask]
        
        if len(results) == 0:
            return f"No content found containing '{keyword}'"
        
        output = f"\nFound {len(results)} entries containing '{keyword}':\n"
        output += f"By {results['RollNo'].nunique()} unique students\n"
        output += f"Average word count: {results['word_count'].mean():.1f}\n"
        
        # Show related topics
        output += "\nMost common related topics:\n"
        for i, (topic, count) in enumerate(results['Topic'].value_counts().head(5).items(), 1):
            output += f"{i}. {topic[:60]}... ({count})\n"
        
        return output
    
    def top_students(self, n=10):
        """Get top N most active students"""
        student_counts = self.df['RollNo'].value_counts().head(n)
        
        output = f"\nTop {n} Most Active Students:\n"
        for i, (roll, count) in enumerate(student_counts.items(), 1):
            student_words = self.df[self.df['RollNo'] == roll]['word_count'].sum()
            output += f"{i:2d}. Roll No {roll}: {count} entries, {student_words} total words\n"
        
        return output
    
    def topic_stats(self):
        """Get topic statistics"""
        topic_counts = self.df['Topic'].value_counts()
        
        output = "\nTopic Statistics:\n"
        output += f"Total unique topics: {len(topic_counts)}\n"
        output += f"Topics with 1 entry: {(topic_counts == 1).sum()}\n"
        output += f"Topics with 2-5 entries: {((topic_counts >= 2) & (topic_counts <= 5)).sum()}\n"
        output += f"Topics with >5 entries: {(topic_counts > 5).sum()}\n"
        
        output += "\nTop 10 Most Discussed Topics:\n"
        for i, (topic, count) in enumerate(topic_counts.head(10).items(), 1):
            output += f"{i:2d}. {topic[:55]}... ({count})\n"
        
        return output
    
    def word_cloud_data(self, top_n=50):
        """Get top N words for word cloud generation"""
        all_text = ' '.join(self.df['YourAnalysis'].fillna('').astype(str))
        words = re.findall(r'\b[a-z]{3,}\b', all_text.lower())
        
        # Stop words
        stop_words = {
            'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can',
            'was', 'one', 'our', 'out', 'has', 'him', 'his', 'how', 'its',
            'let', 'she', 'too', 'use', 'with', 'also', 'been', 'from',
            'have', 'into', 'like', 'make', 'many', 'more', 'most', 'only',
            'other', 'over', 'some', 'such', 'than', 'that', 'their', 'them',
            'then', 'there', 'these', 'they', 'this', 'very', 'were', 'what',
            'when', 'where', 'which', 'will', 'would', 'your'
        }
        
        filtered_words = [w for w in words if w not in stop_words]
        word_freq = Counter(filtered_words).most_common(top_n)
        
        return dict(word_freq)
    
    def daily_activity(self):
        """Get daily activity statistics"""
        self.df['datetime'] = pd.to_datetime(
            self.df['Timestamp'], 
            format='%d-%m-%y %H:%M', 
            errors='coerce'
        )
        self.df['date'] = self.df['datetime'].dt.date
        
        daily_counts = self.df['date'].value_counts().sort_index()
        
        output = "\nDaily Activity:\n"
        for date, count in daily_counts.items():
            output += f"{date}: {count} entries\n"
        
        return output
    
    def keyword_compare(self, *keywords):
        """Compare frequency of multiple keywords"""
        results = {}
        
        for keyword in keywords:
            mask = self.df['YourAnalysis'].fillna('').str.lower().str.contains(keyword.lower())
            count = mask.sum()
            results[keyword] = count
        
        output = "\nKeyword Frequency Comparison:\n"
        for keyword, count in sorted(results.items(), key=lambda x: x[1], reverse=True):
            output += f"{keyword:20s}: {count:4d} entries\n"
        
        return output
    
    def export_filtered_data(self, keyword, filename='filtered_data.csv'):
        """Export entries containing keyword to CSV"""
        mask = self.df['YourAnalysis'].fillna('').str.lower().str.contains(keyword.lower())
        filtered_df = self.df[mask]
        
        filtered_df.to_csv(filename, index=False)
        return f"Exported {len(filtered_df)} entries to {filename}"


# Quick usage examples
def demo():
    """Demonstrate the analyzer usage"""
    print("="*60)
    print("SESSION ANALYZER DEMO")
    print("="*60)
    
    # Initialize
    analyzer = SessionAnalyzer()
    
    # Example 1: Student summary
    print("\n1. Student Summary:")
    print(analyzer.student_summary(127))
    
    # Example 2: Topic search
    print("\n2. Topic Search:")
    print(analyzer.topic_search('regression'))
    
    # Example 3: Content search
    print("\n3. Content Search:")
    print(analyzer.content_search('classification'))
    
    # Example 4: Top students
    print("\n4. Top Students:")
    print(analyzer.top_students(5))
    
    # Example 5: Topic stats
    print("\n5. Topic Statistics:")
    print(analyzer.topic_stats())
    
    # Example 6: Keyword comparison
    print("\n6. Keyword Comparison:")
    print(analyzer.keyword_compare('regression', 'classification', 'clustering', 'neural'))
    
    print("\n" + "="*60)
    print("Demo complete! Try creating your own analyzer:")
    print("  analyzer = SessionAnalyzer()")
    print("  print(analyzer.student_summary(YOUR_ROLL_NO))")
    print("="*60)


if __name__ == "__main__":
    demo()
