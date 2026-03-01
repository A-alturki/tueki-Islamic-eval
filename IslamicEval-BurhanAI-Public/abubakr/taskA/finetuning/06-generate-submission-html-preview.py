"""
Submission HTML Preview Generator
===============================

This script generates an HTML preview for Subtask 1A submissions to help debug and review results.
"""

import pandas as pd
import json
import argparse
import os
from typing import Dict, List, Any

class SubmissionHTMLPreview:
    """Generate HTML preview for submission debugging"""
    
    def __init__(self):
        pass
    
    def load_test_data(self, test_file_path: str) -> Dict[str, Dict[str, Any]]:
        """Load test data from JSONL file and index by ID"""
        test_data = {}
        with open(test_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                test_data[data['ID']] = data
        return test_data
    
    def load_submission_data(self, submission_file_path: str) -> pd.DataFrame:
        """Load submission data from TSV file"""
        return pd.read_csv(submission_file_path, sep='\t')
    
    def highlight_spans(self, text: str, spans: List[Dict[str, Any]]) -> str:
        """Highlight detected spans in the text with colored backgrounds"""
        if not spans:
            return text
        
        # Sort spans by start position (descending) to process from end to beginning
        sorted_spans = sorted(spans, key=lambda x: x['Span_Start'], reverse=True)
        
        highlighted_text = text
        colors = {
            'Ayah': 'bg-success text-white',
            'Hadith': 'bg-primary text-white', 
            'NoAnnotation': 'bg-secondary text-white'
        }
        
        for span in sorted_spans:
            if span['Span_Type'] != 'NoAnnotation':
                start = span['Span_Start']
                end = span['Span_End'] + 1  # Convert to exclusive end
                span_type = span['Span_Type']
                color_class = colors.get(span_type, 'bg-warning text-dark')
                
                # Insert highlighting tags
                highlighted_text = (
                    highlighted_text[:start] + 
                    f'<span class="badge {color_class}" title="{span_type}">' +
                    highlighted_text[start:end] + 
                    '</span>' +
                    highlighted_text[end:]
                )
        
        return highlighted_text
    
    def generate_html(self, submission_file: str, test_file: str, output_file: str) -> None:
        """Generate HTML preview file"""
        print(f"Loading submission data from: {submission_file}")
        submission_df = self.load_submission_data(submission_file)
        
        print(f"Loading test data from: {test_file}")
        test_data = self.load_test_data(test_file)
        
        # Group submission data by Question_ID
        submission_groups = submission_df.groupby('Question_ID')
        
        # Generate HTML content
        html_content = self._generate_html_template()
        
        examples_html = ""
        for question_id, group in submission_groups:
            if question_id in test_data:
                example_data = test_data[question_id]
                spans = group.to_dict('records')
                
                examples_html += self._generate_example_html(question_id, example_data, spans)
        
        # Insert examples into template
        html_content = html_content.replace('{{EXAMPLES}}', examples_html)
        
        # Add summary statistics
        summary_html = self._generate_summary_html(submission_df)
        html_content = html_content.replace('{{SUMMARY}}', summary_html)
        
        # Write HTML file
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"HTML preview generated: {output_file}")
        print(f"Total examples: {len(submission_groups)}")
    
    def _generate_html_template(self) -> str:
        """Generate the base HTML template"""
        return """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Subtask 1A Submission Preview</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        .arabic-text { font-family: 'Arial Unicode MS', Tahoma, sans-serif; direction: rtl; text-align: right; }
        .span-highlight { margin: 2px; }
        .example-card { margin-bottom: 20px; }
    </style>
</head>
<body>
    <div class="container-fluid py-4">
        <h1 class="mb-4">Subtask 1A Submission Preview</h1>
        
        <!-- Summary Section -->
        <div class="row mb-4">
            <div class="col-12">
                <div class="card">
                    <div class="card-header">
                        <h3>Summary Statistics</h3>
                    </div>
                    <div class="card-body">
                        {{SUMMARY}}
                    </div>
                </div>
            </div>
        </div>
        
        <!-- Legend -->
        <div class="row mb-4">
            <div class="col-12">
                <div class="card">
                    <div class="card-header">
                        <h4>Legend</h4>
                    </div>
                    <div class="card-body">
                        <span class="badge bg-success text-white me-2">Ayah</span>
                        <span class="badge bg-primary text-white me-2">Hadith</span>
                        <span class="badge bg-secondary text-white me-2">NoAnnotation</span>
                    </div>
                </div>
            </div>
        </div>
        
        <!-- Examples -->
        <div class="row">
            <div class="col-12">
                {{EXAMPLES}}
            </div>
        </div>
    </div>
</body>
</html>"""
    
    def _generate_example_html(self, question_id: str, example_data: Dict[str, Any], spans: List[Dict[str, Any]]) -> str:
        """Generate HTML for a single example"""
        response_text = example_data.get('Response', '')
        question_text = example_data.get('Question', '')
        
        # Highlight spans in response text
        highlighted_response = self.highlight_spans(response_text, spans)
        
        # Generate spans table
        spans_table = self._generate_spans_table(spans)
        
        return f"""
        <div class="card example-card">
            <div class="card-header">
                <h4>Question ID: {question_id}</h4>
            </div>
            <div class="card-body">
                <div class="row">
                    <div class="col-md-6">
                        <h5>Question:</h5>
                        <div class="arabic-text p-3 bg-light rounded">{question_text}</div>
                    </div>
                    <div class="col-md-6">
                        <h5>Response (Highlighted):</h5>
                        <div class="arabic-text p-3 bg-light rounded">{highlighted_response}</div>
                    </div>
                </div>
                <div class="row mt-3">
                    <div class="col-12">
                        <h5>Detected Spans:</h5>
                        {spans_table}
                    </div>
                </div>
            </div>
        </div>
        """
    
    def _generate_spans_table(self, spans: List[Dict[str, Any]]) -> str:
        """Generate HTML table for spans"""
        if not spans or (len(spans) == 1 and spans[0]['Span_Type'] == 'NoAnnotation'):
            return '<p class="text-muted">No spans detected</p>'
        
        table_rows = ""
        for span in spans:
            if span['Span_Type'] != 'NoAnnotation':
                original_span = span.get('Original_Span', 'N/A')
                llm_value = span.get('LLM_Value', 'N/A')
                
                table_rows += f"""
                <tr>
                    <td>{span['Span_Start']}</td>
                    <td>{span['Span_End']}</td>
                    <td><span class="badge bg-{'success' if span['Span_Type'] == 'Ayah' else 'primary'}">{span['Span_Type']}</span></td>
                    <td class="arabic-text">{original_span}</td>
                    <td class="arabic-text">{llm_value}</td>
                </tr>
                """
        
        if not table_rows:
            return '<p class="text-muted">No spans detected</p>'
        
        return f"""
        <table class="table table-striped table-sm">
            <thead>
                <tr>
                    <th>Start</th>
                    <th>End</th>
                    <th>Type</th>
                    <th>Original Text</th>
                    <th>LLM Value</th>
                </tr>
            </thead>
            <tbody>
                {table_rows}
            </tbody>
        </table>
        """
    
    def _generate_summary_html(self, submission_df: pd.DataFrame) -> str:
        """Generate summary statistics HTML"""
        total_examples = submission_df['Question_ID'].nunique()
        span_counts = submission_df['Span_Type'].value_counts()
        
        summary_html = f"""
        <div class="row">
            <div class="col-md-3">
                <div class="card text-center">
                    <div class="card-body">
                        <h4 class="card-title">{total_examples}</h4>
                        <p class="card-text">Total Examples</p>
                    </div>
                </div>
            </div>
        """
        
        for span_type, count in span_counts.items():
            color = 'success' if span_type == 'Ayah' else ('primary' if span_type == 'Hadith' else 'secondary')
            summary_html += f"""
            <div class="col-md-3">
                <div class="card text-center">
                    <div class="card-body">
                        <h4 class="card-title text-{color}">{count}</h4>
                        <p class="card-text">{span_type}</p>
                    </div>
                </div>
            </div>
            """
        
        summary_html += "</div>"
        return summary_html


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Generate HTML preview for Subtask 1A submission")
    parser.add_argument("--submission-file", required=True, help="Path to submission TSV file")
    parser.add_argument("--test-file", required=True, help="Path to test data JSONL file")
    parser.add_argument("--output", default="submission_preview.html", help="Output HTML file path")
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.submission_file):
        print(f"Error: Submission file not found: {args.submission_file}")
        return
    
    if not os.path.exists(args.test_file):
        print(f"Error: Test file not found: {args.test_file}")
        return
    
    try:
        generator = SubmissionHTMLPreview()
        generator.generate_html(args.submission_file, args.test_file, args.output)
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
