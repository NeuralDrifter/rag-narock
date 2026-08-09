import sys
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Convert an image-only PDF into a Searchable PDF using Chrome's Screen AI OCR.")
    parser.add_argument("input_pdf", help="Path to the input PDF file")
    parser.add_argument("output_pdf", nargs="?", help="Path to save the searchable PDF (optional, defaults to <input>_searchable.pdf)")
    
    args = parser.parse_args()
    input_path = Path(args.input_pdf)
    
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        sys.exit(1)
        
    if args.output_pdf:
        output_path = Path(args.output_pdf)
    else:
        output_path = input_path.with_name(f"{input_path.stem}_searchable{input_path.suffix}")

    print("Initializing Chrome Screen AI OCR Engine...")
    
    try:
        from ingestion.locro.ocr import ScreenAI
    except ImportError as e:
        print(f"Error importing locro: {e}")
        print("Make sure you run this script from the root of rag-narock.")
        sys.exit(1)

    try:
        ai = ScreenAI()
    except Exception as e:
        print(f"Failed to initialize Chrome OCR Engine: {e}")
        sys.exit(1)
        
    print(f"Engine Ready! Processing {input_path.name}...")
    
    try:
        # ocr_to_searchable_pdf automatically uses PyMuPDF to place invisible text overlays
        # matching the exact bounding boxes of the detected text.
        ai.ocr_to_searchable_pdf(input_path, output_path)
        print(f"\nSuccess! Searchable PDF saved to: {output_path}")
    except Exception as e:
        print(f"Error during OCR processing: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
