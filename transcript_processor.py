import pdfplumber
import csv
import re
import os
import sys
from datetime import datetime


def extract_text_from_pdf(pdf_path):
    """Extract all text from the PDF transcript."""
    print(f"Extracting text from: {pdf_path}")
    try:
        with pdfplumber.open(pdf_path) as pdf:
            text = ''
            for i, page in enumerate(pdf.pages):
                print(f"Processing page {i + 1}/{len(pdf.pages)}")
                page_text = page.extract_text()
                text += page_text + '\n'
        return text
    except Exception as e:
        print(f"Error opening or reading PDF: {str(e)}")
        return None


def extract_course_data(transcript):
    """Extract course data directly from the PDF content only."""
    course_data = []
    lines = transcript.split('\n')
    current_semester = None

    # Process each line
    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Check for semester information
        semester_match = re.search(r'Semester (\d+) - (\d{4})', line)
        if semester_match:
            sem_num = semester_match.group(1)
            year = semester_match.group(2)
            current_semester = f"{year}-S{sem_num}"
            continue

        # Skip lines that don't contain course information
        if not current_semester or "Program:" in line or "END OF TRANSCRIPT" in line:
            continue

        # Look for department code pattern at the beginning of a line or after whitespace
        course_match = re.search(r'([A-Z]{2,4})\s+(\d{4}[A-Z]?)\b', line)
        if not course_match:
            continue

        dept = course_match.group(1)
        code = course_match.group(2)
        course_code = f"{dept} {code}"

        # Skip "Enrolled" courses
        if "Enrolled" in line:
            continue

        # Special case for COMP 3851A - should not have a grade
        if course_code == "COMP 3851A":
            course_data.append([current_semester, course_code, "NA", "", ""])
            continue

        # Split the line into parts
        parts = re.split(r'\s+', line)

        # Look for grade
        grade = None
        for idx, part in enumerate(parts):
            if part in ['HD', 'D', 'C', 'P', 'F', 'NA', 'UP']:
                grade = part

                # Mark is typically before grade
                mark = ""
                if idx > 0 and parts[idx - 1].isdigit():
                    mark = parts[idx - 1]

                # Units is typically after grade
                units = ""
                if idx < len(parts) - 1:
                    if parts[idx + 1].isdigit():
                        units = parts[idx + 1]
                    elif parts[idx + 1] in ['--', '-']:
                        units = ""

                # Add the course
                course_data.append([current_semester, course_code, grade, mark, units])
                break

    return course_data


def write_to_csv(data, filename):
    """Write course data to CSV."""
    with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['semester', 'course_code', 'grade', 'mark', 'units'])
        writer.writerows(data)


def main():
    # Check if PDF file is provided as a command-line argument
    if len(sys.argv) > 1:
        pdf_filename = sys.argv[1]
        if not os.path.isfile(pdf_filename):
            print(f"Error: File '{pdf_filename}' not found.")
            sys.exit(1)
    else:
        # Prompt for PDF filename
        pdf_filename = input("Enter the path to your transcript PDF: ")
        if not os.path.isfile(pdf_filename):
            print(f"Error: File '{pdf_filename}' not found.")
            sys.exit(1)

    # Extract text from PDF
    transcript_content = extract_text_from_pdf(pdf_filename)

    if transcript_content:
        # Extract course data
        course_data = extract_course_data(transcript_content)

        print(f"Number of courses extracted: {len(course_data)}")

        if course_data:
            # Generate CSV filename based on input PDF name
            base_name = os.path.splitext(os.path.basename(pdf_filename))[0]
            timestamp = datetime.now().strftime("%Y%m%d")
            csv_filename = f"{base_name}_grades_{timestamp}.csv"
            write_to_csv(course_data, csv_filename)
            print(f"CSV file '{csv_filename}' has been created successfully.")

            print("\nPreview of extracted data:")
            print(f"{'Semester':<10} {'Course':<12} {'Grade':<5} {'Mark':<5} {'Units':<5}")
            print("-" * 50)
            for row in course_data[:5]:  # Show first 5 courses
                sem, code, grade, mark, units = row
                print(f"{sem:<10} {code:<12} {grade:<5} {mark:<5} {units:<5}")

            if len(course_data) > 5:
                print(f"... and {len(course_data) - 5} more courses")
        else:
            print("No course data was extracted. The PDF content might not match the expected format.")
    else:
        print("Failed to extract text from the PDF.")


if __name__ == "__main__":
    main()