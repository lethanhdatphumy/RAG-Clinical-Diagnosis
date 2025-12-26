from os.path import exists
from pathlib import Path
import fitz
import json


class ClinicalPDFExtractor:

    def __init__(self, output_dir="data/processed/extracted"):
        """
        :param output_dir: specify where to store the extracted output files.
        We can override it.
        """
        self.output_dir = Path(output_dir)  # Convert the path(str) into a Path object.
        self.output_dir.mkdir(parents=True,  # Ensure that the dir actually existed, else create it.
                              exist_ok=True)  # Void raising error if the folder already created.

    def extract_case_report(self, pdf_path):
        """
        :param pdf_path: the direction of the PDF file (input).
        :return: a dictionary with text content and image paths.
        """
        pdf_name = Path(
            pdf_path).stem  # Extracts just the file name without extension name. Ex: document.pdf -> document.
        case_dir = self.output_dir / pdf_name
        case_dir.mkdir(exist_ok=True)  # Create the directory if it does not exist.
        print(f"the data path is :{pdf_path}")
        print(f"the type of data path: {type(pdf_path)}")
        doc = fitz.open(pdf_path)

        # Store the case data.
        case_data = {
            'pdf_name': pdf_name,
            'pdf_path': str(pdf_path),
            'pages': []
        }

        for page_num, page in enumerate(doc):
            # Store the page information
            page_data = {
                'page_number': page_num + 1,
                'text': '',
                'image': []
            }

            # Get text from the page.
            text = page.get_text()
            page_data['text'] = text.strip()

            # Get image from the page.
            image_list = page.get_images()
            for img_index, img in enumerate(image_list):
                x_ref = img[0]  # The image's reference ID
                base_image = doc.extract_image(x_ref)  # Extract the image bytes and metadata.
                image_bytes = base_image["image"]  # Actual binary image data
                image_ext = base_image["ext"]  # file extension: jpg or png

                # Save the image
                image_filename = f"page{page_num + 1}_image {img_index + 1}.{image_ext}"
                image_path = case_dir / image_filename

                with open(image_path, "wb") as img_file:
                    img_file.write(image_bytes)

                page_data['image'].append({
                    'filename': image_filename,
                    'path': str(image_path),
                    'page': page_num + 1
                })
            case_data['pages'].append(page_data)  # Add case_data into page_data.

        doc.close()

        # Save the metadata.
        metadata_path = case_dir / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(case_data, f, indent=2)
            # json.dump() write an object case_data into a file as JSON text
            # indent=2 makes it formatted and readable (2- indentation)
        return case_data

    def extract_all_report(self, reports_dir="data/raw/case_reports"):
        """
        :param reports_dir: directories of case reports.
        :return: executes all extract_case_report for all cases (PDF files).
        """
        reports_path = Path(reports_dir)
        pdf_files = list(
            reports_path.glob("*.pdf"))  # Search a folder for all PDF files inside it and store their path in a list

        print(f"Found {len(pdf_files)} PDF files")

        all_cases = []
        for pdf_file in pdf_files:
            print(f"\nProcessing: {pdf_file.name}")
            case_data = self.extract_case_report(pdf_file)
            print(case_data)

            # Summary
            total_images = sum(len(page['image']) for page in case_data['pages'])
            print(f"  - Extracted {len(case_data['pages'])} pages")
            print(f"  - Found {total_images} images")

            all_cases.append(case_data)

        return all_cases


if __name__ == "__main__":
    extractor = ClinicalPDFExtractor()
    extractor.extract_all_report()
