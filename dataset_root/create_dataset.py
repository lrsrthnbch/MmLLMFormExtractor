import os
import json
import shutil
from pathlib import Path
from pdf2image import convert_from_path
from PIL import Image, ImageEnhance, ImageOps, ImageFilter
import cv2
import numpy as np
import pytesseract
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import ImageTk
from typing import Dict, List, Optional

class DatasetCreationTool:
    def __init__(self, base_path: str, num_sections: int = 6):
        self.base_path = Path(base_path)
        self.num_sections = num_sections
        self.current_file = None
        self.setup_directories()
        self.load_templates()
        
        # Tesseract setup
        self.tesseract_path = r'C:\Users\x131763\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'
        pytesseract.pytesseract.tesseract_cmd = self.tesseract_path
        
        # PDF conversion parameters
        self.pdf_params = {
            'dpi': 300,
            'max_dim': 2000,
            'sharpness_factor': 2,
            'contrast_factor': 1.2,
            'brightness_factor': 1.1,
            'denoise_factor': 0.2,
            'grayscale': True,
            'poppler_path': r"C:\Users\x131763\OneDrive - SH-Gruppe Prod\Dokumente\Dev\vision\poppler-24.08.0\Library\bin"
        }
        
        # Preprocessing parameters
        self.preprocess_params = {
            'scale_factor': 2.0,
            'contrast_factor': 1.2,
            'brightness_factor': 1.1,
            'bilateral_d': 10,
            'bilateral_sigma_color': 15,
            'bilateral_sigma_space': 15,
            'adaptive_block_size': 35,
            'adaptive_C': 15,
            'final_scale_factor': 1.5
        }
        
        # Initialize dataset
        self.dataset_file = self.base_path / 'dataset.json'
        self.dataset = self.load_dataset()

    def setup_directories(self):
        """Create necessary directory structure"""
        directories = ['input', 'images', 'prompts', 'responses', 'temp']
        for dir_name in directories:
            (self.base_path / dir_name).mkdir(parents=True, exist_ok=True)

    def load_templates(self):
        """Load prompt and response templates"""
        self.prompts = {}
        self.responses = {}
        for i in range(self.num_sections):
            prompt_path = self.base_path / 'prompts' / f'prompt_{i}.txt'
            response_path = self.base_path / 'responses' / f'response_{i}.txt'
            
            if prompt_path.exists():
                self.prompts[i] = prompt_path.read_text(encoding='utf-8')
            if response_path.exists():
                self.responses[i] = response_path.read_text(encoding='utf-8')

    def load_dataset(self) -> List[Dict]:
        """Load or create the dataset file"""
        if self.dataset_file.exists():
            return json.loads(self.dataset_file.read_text(encoding='utf-8'))
        return []

    def save_dataset(self):
        """Save the dataset to file"""
        self.dataset_file.write_text(json.dumps(self.dataset, indent=2), encoding='utf-8')

    def process_image(self, img: Image.Image) -> Image.Image:
        """Process the image using the provided preprocessing code"""
        width, height = img.size
        scaling_factor = min(self.pdf_params['max_dim'] / width, 
                           self.pdf_params['max_dim'] / height, 1)
        img = img.resize((int(width * scaling_factor), 
                         int(height * scaling_factor)), Image.LANCZOS)
        
        if self.pdf_params['denoise_factor'] > 0:
            img = img.filter(ImageFilter.GaussianBlur(self.pdf_params['denoise_factor']))

        if self.pdf_params['sharpness_factor'] != 1.0:
            enhancer = ImageEnhance.Sharpness(img)
            img = enhancer.enhance(self.pdf_params['sharpness_factor'])

        if self.pdf_params['contrast_factor'] != 1.0:
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(self.pdf_params['contrast_factor'])
        
        if self.pdf_params['brightness_factor'] != 1.0:
            enhancer = ImageEnhance.Brightness(img)
            img = enhancer.enhance(self.pdf_params['brightness_factor'])

        if self.pdf_params['grayscale']:
            img = ImageOps.grayscale(img)

        return img

    def enhance_document(self, input_path: Path, output_path: Path):
        """Enhanced document preprocessing with improved text quality"""
        img = cv2.imread(str(input_path))
        if img is None:
            raise FileNotFoundError(f"Could not load image at {input_path}")
        
        # Get original dimensions
        original_height, original_width = img.shape[:2]
        
        # Scale up the image for better processing
        scaled_width = int(original_width * self.preprocess_params['scale_factor'])
        scaled_height = int(original_height * self.preprocess_params['scale_factor'])
        img_scaled = cv2.resize(img, (scaled_width, scaled_height), 
                              interpolation=cv2.INTER_LANCZOS4)
        
        # Convert to grayscale
        gray = cv2.cvtColor(img_scaled, cv2.COLOR_BGR2GRAY)
        
        # Apply bilateral filtering
        denoised = cv2.bilateralFilter(
            gray, 
            self.preprocess_params['bilateral_d'],
            self.preprocess_params['bilateral_sigma_color'],
            self.preprocess_params['bilateral_sigma_space']
        )
        
        # Convert to PIL Image for enhancement
        pil_img = Image.fromarray(denoised)
        
        # Enhance contrast
        enhancer = ImageEnhance.Contrast(pil_img)
        contrast_img = enhancer.enhance(self.preprocess_params['contrast_factor'])
        
        # Enhance brightness
        enhancer = ImageEnhance.Brightness(contrast_img)
        bright_img = enhancer.enhance(self.preprocess_params['brightness_factor'])
        
        # Convert back to OpenCV format
        enhanced = np.array(bright_img)
        
        # Adjust adaptive block size for scaled image
        scaled_block_size = int(self.preprocess_params['adaptive_block_size'] * 
                              self.preprocess_params['scale_factor'])
        if scaled_block_size % 2 == 0:
            scaled_block_size += 1
        
        # Apply adaptive thresholding
        binary = cv2.adaptiveThreshold(
            enhanced,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            scaled_block_size,
            self.preprocess_params['adaptive_C']
        )
        
        # Apply slight Gaussian blur to smooth edges
        smoothed = cv2.GaussianBlur(binary, (3, 3), 0)
        
        # Threshold again to ensure binary image
        _, binary = cv2.threshold(smoothed, 127, 255, cv2.THRESH_BINARY)
        
        # Scale back to desired size
        final_width = int(original_width * self.preprocess_params['final_scale_factor'])
        final_height = int(original_height * self.preprocess_params['final_scale_factor'])
        final_image = cv2.resize(binary, (final_width, final_height), 
                               interpolation=cv2.INTER_LANCZOS4)
        
        # Save the processed image
        cv2.imwrite(str(output_path), final_image)

    def find_section_boundaries(self, image_path: str) -> List[int]:
        """Find section boundaries using the provided section detection code"""
        # Read the image
        image = cv2.imread(image_path)
        height, width = image.shape[:2]
        
        # Perform OCR with bounding boxes
        custom_config = r'--oem 3 --psm 1 -l deu'
        boxes = pytesseract.image_to_data(image, config=custom_config, 
                                        output_type=pytesseract.Output.DICT)
        
        # Define sections with their search terms and expected order
        sections = [
            ("Der Bausparer", "Der Bausparer wünscht", 1),
            ("ändern", "Adresse/Namen", 2),
            ("Vertragsänderung", "Vertragsänderung beantragen", 3),
            ("Umbuchung", "Umbuchung beantragen", 4),
            ("Staatliche", "Staatliche Förderung", 5),
            ("Betreuung", "Betreuung ändern", 6)
        ]
        
        # Dictionary to store all potential matches for each section
        section_candidates = {full_name: [] for _, full_name, _ in sections}
        
        # First pass: collect all potential matches for each section
        for idx, text in enumerate(boxes['text']):
            text = text.strip()
            if text and int(boxes['conf'][idx]) > 0:
                y = boxes['top'][idx]
                
                for search_term, full_name, _ in sections:
                    if search_term.lower() in text.lower():
                        section_candidates[full_name].append({
                            'y': y,
                            'text': text,
                            'confidence': boxes['conf'][idx]
                        })
        
        # Initialize result dictionary
        found_sections = {'top': 0}  # Always include top of page
        section_coords = [0]  # Start with top of page
        
        # Second pass: select the best candidate for each section based on position and order
        last_y = 0
        min_section_distance = 50
        
        for _, full_name, expected_order in sections:
            candidates = section_candidates[full_name]
            if candidates:
                # Sort candidates by vertical position
                candidates.sort(key=lambda x: x['y'])
                
                # Find the first candidate that's far enough from the last section
                # and follows the expected order
                for candidate in candidates:
                    y = candidate['y']
                    if (y > last_y + min_section_distance and 
                        y < height - min_section_distance and  
                        all(y > found_sections.get(prev_name, 0) 
                            for prev_name in [s[1] for s in sections if s[2] < expected_order])):
                        
                        found_sections[full_name] = y
                        section_coords.append(y)
                        last_y = y
                        break
        
        # Sort coordinates
        section_coords.sort()
        
        # Add bottom of image
        section_coords.append(height)
        
        return section_coords
    
    def visualize_sections(self, image_path: Path, section_coords: List[int]) -> Path:
        """Visualize detected sections with red lines and labels"""
        # Read the image
        image = cv2.imread(str(image_path))
        debug_image = image.copy()
        
        # Draw horizontal lines at each section boundary
        for i, y in enumerate(section_coords):
            # Draw red lines for section boundaries
            cv2.line(debug_image, (0, y), (image.shape[1], y), (0, 0, 255), 2)
            
            # Add section number label
            if i < len(section_coords) - 1:
                cv2.putText(debug_image, f"Section {i}", 
                           (10, y + 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 
                           1, (0, 0, 255), 2)
        
        # Save debug image
        debug_dir = self.base_path / 'debug'
        debug_dir.mkdir(exist_ok=True)
        debug_path = debug_dir / f"{image_path.stem}_annotated.jpg"
        cv2.imwrite(str(debug_path), debug_image)
        
        return debug_path

    def verify_sections(self, image_path: Path, section_coords: List[int]) -> bool:
        """Show verification dialog with annotated image"""
        root = tk.Tk()
        root.title("Verify Section Detection")
        
        # Create main container with padding
        main_frame = ttk.Frame(root)
        main_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Add instruction label
        ttk.Label(main_frame, 
                 text="Please verify if the sections were detected correctly:",
                 font=('Arial', 12)).pack(pady=5)
        
        # Create frame for image and scrollbar
        image_container = ttk.Frame(main_frame)
        image_container.pack(fill='both', expand=True, pady=5)
        
        # Create annotated image and display it
        debug_path = self.visualize_sections(image_path, section_coords)
        img = Image.open(debug_path)
        
        # Calculate new dimensions maintaining aspect ratio
        MAX_WIDTH = 600
        MAX_HEIGHT = 800
        
        width_ratio = MAX_WIDTH / img.width
        height_ratio = MAX_HEIGHT / img.height
        scale_factor = min(width_ratio, height_ratio)
        
        new_width = int(img.width * scale_factor)
        new_height = int(img.height * scale_factor)
        
        img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # Create scrollable frame for image
        canvas = tk.Canvas(image_container, width=new_width, height=min(new_height, MAX_HEIGHT))
        scrollbar = ttk.Scrollbar(image_container, orient="vertical", command=canvas.yview)
        
        photo = ImageTk.PhotoImage(img)
        image_label = ttk.Label(canvas, image=photo)
        image_label.image = photo
        
        canvas.create_window((0, 0), window=image_label, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Pack scrollable components
        canvas.pack(side="left", fill="both", expand=True)
        if new_height > MAX_HEIGHT:
            scrollbar.pack(side="right", fill="y")
        
        # Create a separate frame for buttons at the bottom
        button_container = ttk.Frame(main_frame)
        button_container.pack(fill='x', pady=10)
        
        # Create centered frame for buttons
        button_frame = ttk.Frame(button_container)
        button_frame.pack(anchor='center')
        
        # Result variable
        result = {'verified': False}
        
        def confirm():
            result['verified'] = True
            root.quit()
            root.destroy()
        
        def reject():
            result['verified'] = False
            root.quit()
            root.destroy()
        
        ttk.Button(button_frame, 
                  text="Sections Detected Correctly", 
                  command=confirm).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, 
                  text="Detection Incorrect - Cancel", 
                  command=reject).pack(side=tk.LEFT, padx=5)
        
        # Configure canvas scrolling
        canvas.bind('<Configure>', lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        canvas.bind_all("<MouseWheel>", _on_mousewheel)
        
        # Set window size and position
        root.update_idletasks()
        
        # Add extra padding for buttons and make sure they're visible
        window_width = new_width + 50
        window_height = min(new_height, MAX_HEIGHT) + 120  # Increased padding for buttons
        
        # Center the window
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        x = (screen_width - window_width) // 2
        y = (screen_height - window_height) // 2
        
        # Set minimum size to ensure buttons are always visible
        root.minsize(window_width, window_height)
        root.geometry(f"{window_width}x{window_height}+{x}+{y}")
        
        root.mainloop()
        
        return result['verified']

    def convert_pdf_to_image(self, pdf_path: Path) -> Path:
        """Convert PDF to image using the provided PDF conversion code"""
        images = convert_from_path(str(pdf_path), dpi=self.pdf_params['dpi'], 
                                 poppler_path=self.pdf_params['poppler_path'])
        if not images:
            raise ValueError("No images extracted from PDF")
            
        img = images[0]  # We only process the first page
        processed_img = self.process_image(img)
        output_path = self.base_path / 'temp' / f'{pdf_path.stem}.jpg'
        processed_img.save(str(output_path), format='JPEG')
        return output_path

    def split_document(self, image_path: Path) -> List[Path]:
        """Split the document into sections"""
        section_coords = self.find_section_boundaries(str(image_path))
        return self.save_sections(image_path, section_coords)

    def save_sections(self, image_path: Path, section_coords: List[int]) -> List[Path]:
        """Save individual sections as separate images"""
        image = cv2.imread(str(image_path))
        section_paths = []
        
        for i in range(len(section_coords) - 1):
            y1 = section_coords[i]
            y2 = section_coords[i + 1]
            section = image[y1:y2, :]
            
            output_path = self.base_path / 'images' / f'{image_path.stem}_section_{i}.jpg'
            cv2.imwrite(str(output_path), section)
            section_paths.append(output_path)
            
        return section_paths

    def process_document(self, file_path: Path):
        """Main processing pipeline for a document"""
        self.current_file = file_path.stem
        
        try:
            # Convert PDF if necessary
            if file_path.suffix.lower() == '.pdf':
                file_path = self.convert_pdf_to_image(file_path)
            
            # Preprocess image
            preprocessed_path = self.base_path / 'temp' / f'{file_path.stem}_preprocessed.jpg'
            self.enhance_document(file_path, preprocessed_path)
            
            # Find section boundaries
            section_coords = self.find_section_boundaries(str(preprocessed_path))
            
            # Verify sections with user
            if not self.verify_sections(preprocessed_path, section_coords):
                print(f"Section detection cancelled for {file_path}")
                return False
            
            # Split into sections and process each one
            section_paths = self.split_document(preprocessed_path)
            
            # Process each section
            for i, section_path in enumerate(section_paths):
                if not self.process_section(i, section_path):
                    return False  # User cancelled or error occurred
            
            return True
            
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
            self.revert_current_document()
            return False

    def process_section(self, section_index: int, section_path: Path) -> bool:
        """Process a single section with user interaction"""
        # Create UI for user interaction
        return self.show_section_ui(section_index, section_path)

    def create_form_fields(self, parent, json_data, prefix=''):
        """Recursively create form fields for JSON structure"""
        fields = {}
        entry_widgets = []
        
        for key, value in json_data.items():
            full_key = f"{prefix}.{key}" if prefix else key
            
            if isinstance(value, dict):
                # Recursively handle nested dictionaries
                nested_frame = ttk.LabelFrame(parent, text=key)
                nested_frame.pack(fill='x', padx=5, pady=5)
                nested_fields, nested_entries = self.create_form_fields(nested_frame, value, full_key)
                fields.update(nested_fields)
                entry_widgets.extend(nested_entries)
            else:
                # Create a frame for each field
                field_frame = ttk.Frame(parent)
                field_frame.pack(fill='x', padx=5, pady=2)
                
                # Create label
                label = ttk.Label(field_frame, text=key)
                label.pack(side='left', padx=5)
                
                if isinstance(value, bool):
                    # Create checkbox for boolean values
                    var = tk.BooleanVar(value=value)
                    field = ttk.Checkbutton(field_frame, variable=var)
                    field.pack(side='left')
                    fields[full_key] = var
                else:
                    # Create text entry for string values
                    var = tk.StringVar(value=str(value))
                    field = ttk.Entry(field_frame, textvariable=var)
                    field.pack(side='left', fill='x', expand=True)
                    fields[full_key] = var
                    entry_widgets.append(field)  # Store the Entry widget itself
        
        return fields, entry_widgets

    def show_section_ui(self, section_index: int, section_path: Path) -> bool:
        """Show UI for processing a section"""
        root = tk.Tk()
        root.title(f"Process Section {section_index}")
        
        # Create main container
        main_frame = ttk.Frame(root)
        main_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Load and scale image
        img = Image.open(section_path)
        
        # Calculate new dimensions maintaining aspect ratio
        MAX_WIDTH = 800
        aspect_ratio = img.width / img.height
        
        if img.width > MAX_WIDTH:
            new_width = MAX_WIDTH
            new_height = int(MAX_WIDTH / aspect_ratio)
        else:
            new_width = img.width
            new_height = img.height
        
        img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(img)
        
        # Create image label
        image_label = ttk.Label(main_frame, image=photo)
        image_label.image = photo
        image_label.pack(padx=5, pady=5)
        
        # Create form container with fixed height
        form_container = ttk.Frame(main_frame)
        form_container.pack(fill='x', padx=5, pady=5)
        
        # Add scrollable form
        form_canvas = tk.Canvas(form_container, height=300)  # Fixed height for form area
        form_scrollbar = ttk.Scrollbar(form_container, orient="vertical", command=form_canvas.yview)
        scrollable_frame = ttk.Frame(form_canvas)
        
        # Configure scrolling
        scrollable_frame.bind(
            "<Configure>",
            lambda e: form_canvas.configure(scrollregion=form_canvas.bbox("all"))
        )
        
        # Create window in canvas for form content
        form_canvas.create_window((0, 0), window=scrollable_frame, anchor="nw", width=form_canvas.winfo_reqwidth())
        form_canvas.configure(yscrollcommand=form_scrollbar.set)
        
        # Pack form components
        form_scrollbar.pack(side="right", fill="y")
        form_canvas.pack(side="left", fill="both", expand=True)
        
        # Load JSON template
        template = json.loads(self.responses.get(section_index, "{}"))
        
        # Create form fields
        fields, entry_widgets = self.create_form_fields(scrollable_frame, template)
        
        # Button frame at the bottom
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill='x', pady=10)
        
        def save_and_continue():
            result = template.copy()
            for key, var in fields.items():
                keys = key.split('.')
                current = result
                for k in keys[:-1]:
                    current = current[k]
                current[keys[-1]] = var.get()
            
            try:
                self.add_to_dataset(section_index, section_path, result)
                root.quit()
                root.destroy()
            except Exception as e:
                messagebox.showerror("Error", f"Error saving data: {str(e)}")
        
        def cancel():
            root.quit()
            root.destroy()
        
        ttk.Button(button_frame, text="Save and Continue", command=save_and_continue).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Cancel", command=cancel).pack(side=tk.LEFT, padx=5)
        
        # Enable keyboard navigation
        def on_tab(event):
            event.widget.tk_focusNext().focus()
            return "break"
            
        # Bind tab key to all Entry widgets
        for entry in entry_widgets:
            entry.bind("<Tab>", on_tab)
        
        # Configure form canvas scrolling
        def _on_mousewheel(event):
            form_canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        
        form_canvas.bind_all("<MouseWheel>", _on_mousewheel)
        
        # Set initial window size and position
        root.update_idletasks()
        window_width = new_width + 50  # image width + padding
        window_height = new_height + 400  # image height + form height + buttons + padding
        
        # Center the window
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        x = (screen_width - window_width) // 2
        y = (screen_height - window_height) // 2
        
        root.geometry(f"{window_width}x{window_height}+{x}+{y}")
        
        root.mainloop()
        
        try:
            root.winfo_exists()
            return False
        except tk.TclError:
            return True

    def add_to_dataset(self, section_index: int, image_path: Path, json_data: Dict):
        """Add a processed section to the dataset"""
        entry = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": str(image_path.relative_to(self.base_path))},
                        {"type": "text", "text": self.prompts.get(section_index, "")}
                    ]
                },
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": json.dumps(json_data)}
                    ]
                }
            ]
        }
        self.dataset.append(entry)
        self.save_dataset()

    def revert_current_document(self):
        """Revert changes for the current document"""
        if not self.current_file:
            return
            
        # Remove images
        for path in (self.base_path / 'images').glob(f'{self.current_file}*.jpg'):
            path.unlink()
            
        # Remove from dataset
        self.dataset = [entry for entry in self.dataset 
                       if not entry["messages"][0]["content"][0]["image"].startswith(f'{self.current_file}_')]
        self.save_dataset()

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Dataset Creation Tool for Vision LLM Finetuning')
    parser.add_argument('--base-path', type=str, default='dataset_root',
                      help='Base path for the dataset directory structure')
    parser.add_argument('--num-sections', type=int, default=6,
                      help='Number of sections to split documents into')
    parser.add_argument('--tesseract-path', type=str,
                      help='Path to Tesseract executable')
    parser.add_argument('--poppler-path', type=str,
                      help='Path to Poppler executable')
    
    args = parser.parse_args()
    
    # Create tool instance
    tool = DatasetCreationTool(args.base_path, args.num_sections)
    
    # Override paths if provided
    if args.tesseract_path:
        tool.tesseract_path = args.tesseract_path
        pytesseract.pytesseract.tesseract_cmd = args.tesseract_path
    if args.poppler_path:
        tool.pdf_params['poppler_path'] = args.poppler_path
    
    # Process all files in input directory
    input_dir = tool.base_path / 'input'
    for file_path in input_dir.glob('*.*'):
        if file_path.suffix.lower() in ['.pdf', '.jpg', '.jpeg']:
            print(f"Processing {file_path}...")
            try:
                if not tool.process_document(file_path):
                    print(f"Processing cancelled for {file_path}")
                    tool.revert_current_document()
            except Exception as e:
                print(f"Error processing {file_path}: {str(e)}")
                tool.revert_current_document()
                if not messagebox.askyesno("Error", 
                    f"Error processing {file_path}: {str(e)}\nContinue with next file?"):
                    break

if __name__ == "__main__":
    main()