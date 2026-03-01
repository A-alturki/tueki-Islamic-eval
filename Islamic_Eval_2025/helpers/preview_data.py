import pandas as pd
import xml.etree.ElementTree as ET
import re

# test data for 1c
XML_file = 'resources/test dataset/Test_Subtask_1C.xml'  # full data with question ..
Annotation = 'resources/test dataset/Test_Subtask_1C_USER.tsv'  # annotation file or the file we will find answers for
add_root = False

XML_file = 'resources/test dataset/Test_Subtask_1B.xml'  # full data with question ..
Annotation = 'resources/test dataset/Test_Subtask_1B_USER.tsv'  # annotation file or the file we will find answers for
add_root = False
#####################################
# XML_file = 'resources/dev_SubtaskB/dev_SubtaskB.xml'  # full data with question ..
# Annotation = 'resources/dev_SubtaskB/dev_SubtaskB.tsv'  # annotation file or the file we will find answers for
# add_root = True
#####################################

class IslamEvalProcessor:
    def __init__(self, tsv_path, xml_path):
        self.tsv_path = tsv_path
        self.xml_path = xml_path
        self.tsv_data = None
        self.questions_dict = None

    def read_tsv(self):
        """Read the TSV annotation file"""
        self.tsv_data = pd.read_csv(self.tsv_path, sep='\t')
        return self.tsv_data

    def read_xml(self):
        """Read the XML file containing the actual text"""
        if self.questions_dict is not None:
            return self.questions_dict

        self.questions_dict = {}

        with open(self.xml_path, 'r', encoding='utf-8') as file:
            content = file.read()

        # Wrap the content with a root element to make it valid XML
        wrapped_content = content
        if add_root:
            wrapped_content = f"<root>{content}</root>"

        try:
            root = ET.fromstring(wrapped_content)

            # Extract all Question elements
            for question in root.findall('Question'):
                question_id_elem = question.find('ID')
                text_elem = question.find('Text')
                response_elem = question.find('Response')

                if question_id_elem is not None:
                    question_id = question_id_elem.text.strip()

                    # Store as tuple (Text, Response)
                    text_content = text_elem.text.strip() if text_elem is not None and text_elem.text else ""
                    response_content = response_elem.text.strip() if response_elem is not None and response_elem.text else ""

                    self.questions_dict[question_id] = (text_content, response_content)

        except ET.ParseError as e:
            print(f"XML Parse Error: {e}")
            return None

        return self.questions_dict

    def extract_text_by_question_id(self, question_id):
        """Extract the full text for a given question ID"""
        if self.questions_dict is None:
            self.read_xml()

        if question_id in self.questions_dict:
            query, response_content = self.questions_dict[question_id]
            # Combine them for span extraction (as spans likely reference the combined text)
            return query.rstrip(), response_content
        return None

    def get_question_parts(self, question_id):
        """Get the separate Text and Response parts for a given question ID"""
        if self.questions_dict is None:
            self.read_xml()

        return self.questions_dict.get(question_id, ("", ""))

    def extract_span_text(self, full_text, start_pos, end_pos):
        """Extract specific span from the full text"""
        if full_text and 0 <= start_pos < len(full_text) and start_pos < end_pos <= len(full_text):
            return full_text[start_pos:end_pos - 1]
        return None

    def process_annotations(self):
        """Process all annotations and extract corresponding spans"""
        if self.tsv_data is None:
            self.read_tsv()

        results = []

        for _, row in self.tsv_data.iterrows():
            question_id = row['Question_ID']
            span_start = int(row['Span_Start']) if pd.notna(row['Span_Start']) else 0
            span_end = int(row['Span_End']) if pd.notna(row['Span_End']) else 0
            label = row.get("Label", "---")

            # Get full text for this question (combined for span extraction)
            query, response = self.extract_text_by_question_id(question_id)

            # Get separate parts
            text_part, response_part = self.get_question_parts(question_id)

            # Extract the specific span
            extracted_span = self.extract_span_text(response, span_start, span_end)

            result = {
                'Question_ID': question_id,
                'Annotation_ID': row.get("Annotation_ID", "---"),
                'Sequence_ID': row.get("Sequence_ID", "---"),
                'Label': label,
                'Span_Start': span_start,
                'Span_End': span_end,
                'Original_Span': row['Original_Span'] if 'Original_Span' in row else '',
                'Extracted_Text': extracted_span,
                'Full_Question_Text': response,
                'Question_Text': text_part,
                'Question_Response': response_part,
                'Text_Length': len(response) if response else 0
            }

            results.append(result)

        return results

    def get_samples_by_label(self, results, label_type=None, num_samples=5):
        """Get sample results, optionally filtered by label type"""
        if label_type:
            filtered_results = [r for r in results if r['Label'] == label_type]
        else:
            filtered_results = results

        return filtered_results[:num_samples]

    def debug_question_parsing(self, num_questions=3):
        """Debug function to show how questions are being parsed"""
        if self.questions_dict is None:
            self.read_xml()

        print("DEBUG: Question parsing results:")
        print(f"Total questions parsed: {len(self.questions_dict)}")

        for i, (qid, (text, response)) in enumerate(list(self.questions_dict.items())[:num_questions]):
            print(f"\nQuestion ID: {qid}")
            print(f"Text part length: {len(text)}")
            print(f"Response part length: {len(response)}")
            print(f"Text part (first 100 chars): {text[:100]}...")
            print(f"Response part (first 100 chars): {response[:100]}...")


# Usage example
def main():
    # Initialize the processor

    processor = IslamEvalProcessor(Annotation,
                                   XML_file)
    # Debug XML parsing first
    print("=" * 60)
    print("DEBUGGING XML PARSING:")
    print("=" * 60)
    processor.debug_question_parsing()

    # Read the TSV file and show sample entries
    print("\n" + "=" * 60)
    print("TSV FILE ANALYSIS:")
    print("=" * 60)

    tsv_data = processor.read_tsv()
    print("Sample TSV entries:")
    print(tsv_data.head())
    print(f"\nTotal annotations: {len(tsv_data)}")
    print(f"Unique questions: {tsv_data['Question_ID'].nunique()}")
    print(f"Label distribution:")
    # print(tsv_data['Label'].value_counts())

    # Process all annotations
    print("\nProcessing annotations...")
    results = processor.process_annotations()

    # Show samples for different label types
    print("\n" + "=" * 80)
    print("SAMPLE EXTRACTED SPANS:")
    print("=" * 80)

    # Get all unique labels
    all_labels = set(r['Label'] for r in results)
    print(f"Available labels: {all_labels}")

    for label in all_labels:
        samples = processor.get_samples_by_label(results, label, 100)
        if samples:
            print(f"\n{label} samples:")
            for i, result in enumerate(samples, 1):
                print(f"\n{i}. Question: {result['Question_ID']}")
                print(f"   Sequence_ID: {result['Sequence_ID']}")
                print(f"   Label: {result['Label']}")
                print(f"   Span: {result['Span_Start']}-{result['Span_End']}")
                print(f"   Original Span: {result['Original_Span']}")
                print(f"   Extracted Text: {result['Extracted_Text']}")
                print(f"   Question Text Length: {len(result['Question_Text'])}")
                print(f"   Response Length: {len(result['Question_Response'])}")

    return results


# Run the main function
if __name__ == "__main__":
    results = main()
