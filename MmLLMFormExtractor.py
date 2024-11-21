import cv2
import numpy as np
from PIL import Image, ImageEnhance
import pytesseract
import os
from pathlib import Path
import gradio as gr
import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import shutil

# Configure Tesseract path for Linux
pytesseract.pytesseract.tesseract_cmd = 'tesseract'

class DocumentProcessor:
    def __init__(self):
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2-VL-7B-Instruct", 
            torch_dtype=torch.bfloat16, 
            device_map="auto"
        )
        self.processor = AutoProcessor.from_pretrained(
            "Qwen/Qwen2-VL-7B-Instruct", 
            min_pixels=256*28*28, 
            max_pixels=1280*28*28
        )
        self.setup_directories()
        
    def setup_directories(self):
        """Create necessary directories if they don't exist"""
        self.dirs = {
            'preprocess': Path('preprocess'),
            'preprocessed': Path('preprocessed'),
            'debug': Path('debug'),
            'split': Path('split'),
            'prompts': Path('prompts')
        }
        for dir_path in self.dirs.values():
            dir_path.mkdir(exist_ok=True)
            
    def clean_directories(self, keep_uploaded=None):
        """Clean all processing directories except the uploaded file"""
        for dir_name, dir_path in self.dirs.items():
            if dir_name == 'prompts':  # Don't clean prompts directory
                continue
            for file_path in dir_path.glob('*'):
                if keep_uploaded and file_path == keep_uploaded:
                    continue
                try:
                    if file_path.is_file():
                        file_path.unlink()
                except Exception as e:
                    print(f"Error cleaning {file_path}: {e}")

    def enhance_document(self, input_path, output_path):
        """Enhanced document processing function"""
        params = {
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
        
        img = cv2.imread(str(input_path))
        if img is None:
            raise FileNotFoundError(f"Could not load image at {input_path}")
        
        original_height, original_width = img.shape[:2]
        
        # Scale up and convert to grayscale
        scaled_dims = (int(original_width * params['scale_factor']), 
                      int(original_height * params['scale_factor']))
        img_scaled = cv2.resize(img, scaled_dims, interpolation=cv2.INTER_LANCZOS4)
        gray = cv2.cvtColor(img_scaled, cv2.COLOR_BGR2GRAY)
        
        # Apply bilateral filtering
        denoised = cv2.bilateralFilter(gray, 
                                     params['bilateral_d'],
                                     params['bilateral_sigma_color'],
                                     params['bilateral_sigma_space'])
        
        # Enhance contrast and brightness using PIL
        pil_img = Image.fromarray(denoised)
        enhanced = np.array(ImageEnhance.Brightness(
            ImageEnhance.Contrast(pil_img).enhance(params['contrast_factor'])
        ).enhance(params['brightness_factor']))
        
        # Apply adaptive thresholding
        block_size = int(params['adaptive_block_size'] * params['scale_factor'])
        if block_size % 2 == 0:
            block_size += 1
        
        binary = cv2.adaptiveThreshold(enhanced, 255,
                                     cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY,
                                     block_size,
                                     params['adaptive_C'])
        
        # Final processing and scaling
        smoothed = cv2.GaussianBlur(binary, (3, 3), 0)
        _, binary = cv2.threshold(smoothed, 127, 255, cv2.THRESH_BINARY)
        
        final_dims = (int(original_width * params['final_scale_factor']),
                     int(original_height * params['final_scale_factor']))
        final_image = cv2.resize(binary, final_dims, interpolation=cv2.INTER_LANCZOS4)
        
        cv2.imwrite(str(output_path), final_image)
        return final_image

    def find_section_boundaries(self, image_path):
        """Find document section boundaries"""
        image = cv2.imread(str(image_path))
        height = image.shape[0]
        
        boxes = pytesseract.image_to_data(image, 
                                        config=r'--oem 3 --psm 1 -l deu',
                                        output_type=pytesseract.Output.DICT)
        
        sections = [
            ("Der Bausparer", "Der Bausparer wünscht", 1),
            ("Adresse/Namen", "Adresse/Namen ändern", 2),
            ("Vertragsänderung", "Vertragsänderung beantragen", 3),
            ("Umbuchung", "Umbuchung beantragen", 4),
            ("Staatliche", "Staatliche Förderung", 5),
            ("Betreuung", "Betreuung ändern", 6)
        ]
        
        section_candidates = {full_name: [] for _, full_name, _ in sections}
        for idx, text in enumerate(boxes['text']):
            if text.strip() and int(boxes['conf'][idx]) > 0:
                y = boxes['top'][idx]
                for search_term, full_name, _ in sections:
                    if search_term.lower() in text.lower():
                        section_candidates[full_name].append({
                            'y': y,
                            'confidence': boxes['conf'][idx]
                        })
        
        found_sections = {'top': 0}
        section_coords = [0]
        last_y = 0
        min_section_distance = 50
        
        for _, full_name, expected_order in sections:
            candidates = sorted(section_candidates[full_name], key=lambda x: x['y'])
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
        
        section_coords.sort()
        section_coords.append(height)
        
        return section_coords

    def process_image(self, image):
        """Process a single image through the pipeline"""
        # Clean previous files
        self.clean_directories(keep_uploaded=None)
        
        # Save uploaded image
        input_path = self.dirs['preprocess'] / "input.jpg"
        image.save(str(input_path))
        
        # Enhance document
        output_path = self.dirs['preprocessed'] / "enhanced.jpg"
        self.enhance_document(input_path, output_path)
        
        # Find and visualize sections
        section_coords = self.find_section_boundaries(output_path)
        debug_image = self.visualize_sections(output_path, section_coords)
        
        # Split into sections
        section_images = []
        if len(section_coords) > 1:
            image = cv2.imread(str(output_path))
            for i in range(len(section_coords) - 1):
                section = image[section_coords[i]:section_coords[i + 1], :]
                section_path = self.dirs['split'] / f"section_{i}.jpg"
                cv2.imwrite(str(section_path), section)
                section_images.append(Image.open(section_path))
        
        return debug_image, section_images

    def visualize_sections(self, image_path, section_coords):
        """Create debug visualization of detected sections"""
        image = cv2.imread(str(image_path))
        debug_image = image.copy()
        
        for i, y in enumerate(section_coords):
            cv2.line(debug_image, (0, y), (image.shape[1], y), (0, 0, 255), 2)
            if i < len(section_coords) - 1:
                cv2.putText(debug_image, f"Section {i}", 
                           (10, y + 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 
                           1, (0, 0, 255), 2)
        
        debug_path = self.dirs['debug'] / "annotated_sections.jpg"
        cv2.imwrite(str(debug_path), debug_image)
        return Image.fromarray(cv2.cvtColor(debug_image, cv2.COLOR_BGR2RGB))

    def load_prompt(self, section_idx):
        """Load prompt from file"""
        prompt_path = self.dirs['prompts'] / f"prompt_{section_idx}.txt"
        try:
            with open(prompt_path, 'r', encoding='utf-8') as f:
                return f.read().strip()
        except:
            return f"Describe section {section_idx} of the document."

    def process_section(self, image, section_idx):
        """Process a single section with the vision model"""
        prompt = self.load_prompt(section_idx)
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        )
        inputs = inputs.to("cuda" if torch.cuda.is_available() else "cpu")

        generated_ids = self.model.generate(**inputs, max_new_tokens=4096)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        return output_text[0]

def create_interface():
    processor = DocumentProcessor()
    
    with gr.Blocks() as demo:
        gr.Markdown("# Multimodal Document Inference @labsl001w002t 🖼️📄")
        
        with gr.Row():
            # Left column for input and controls
            with gr.Column(scale=1):
                input_image = gr.Image(type="pil", label="Upload Document", height=500)
                with gr.Row():
                    detect_btn = gr.Button("1. Detect & Split Sections", variant="primary")
                    process_btn = gr.Button("2. Analyze Sections", variant="primary", interactive=False)
                    stop_btn = gr.Button("Reset", variant="stop")
            
            # Right column for debug visualization
            with gr.Column(scale=1):
                debug_image = gr.Image(type="pil", label="Section Separation", height=500)
        
        # Store section images for analysis
        section_images_store = gr.State([])
        
        # Grid layout for sections
        section_containers = []
        for i in range(6):  # Up to 6 sections
            with gr.Row(visible=False) as container:
                with gr.Column(scale=1):
                    section_image = gr.Image(
                        type="pil",
                        label=f"Section {i}",  # Changed to match internal indexing (0-5)
                        height=320,
                        visible=False
                    )
                
                with gr.Column(scale=1):
                    with gr.Accordion("Prompt", open=False):
                        prompt = gr.Textbox(
                            label=None,
                            visible=False,
                            lines=10
                        )
                    
                    output = gr.Textbox(
                        label="Analysis",
                        visible=False,
                        lines=10
                    )
                    
                section_containers.append({
                    'container': container,
                    'image': section_image,
                    'prompt': prompt,
                    'output': output
                })

        def detect_sections(image):
            if image is None:
                return [None, gr.update(interactive=False), [], *([gr.update(visible=False), None, gr.update(visible=False), "", gr.update(visible=False), "", gr.update(visible=False)] * 6)]

            try:
                # Process image for detection and splitting only
                debug_img, section_images = processor.process_image(image)
                
                # Update UI components
                outputs = [debug_img, gr.update(interactive=True), section_images]
                
                # Update section displays
                for i in range(6):
                    if i < len(section_images):
                        outputs.extend([
                            gr.update(visible=True),  # Container
                            section_images[i],        # Image
                            gr.update(visible=True),  # Image visibility
                            processor.load_prompt(i), # Prompt
                            gr.update(visible=True),  # Prompt visibility
                            "",                      # Output (empty)
                            gr.update(visible=True)   # Output visibility
                        ])
                    else:
                        outputs.extend([
                            gr.update(visible=False),
                            None,
                            gr.update(visible=False),
                            "",
                            gr.update(visible=False),
                            "",
                            gr.update(visible=False)
                        ])
                
                return outputs
                
            except Exception as e:
                return [None, gr.update(interactive=False), [], *([gr.update(visible=False), None, gr.update(visible=False), "", gr.update(visible=False), "", gr.update(visible=False)] * 6)]

        def process_sections(image, stored_images):
            if not stored_images:
                return [""] * 6
            
            try:
                outputs = []
                
                # Process each section with the LLM
                for i, section_img in enumerate(stored_images):
                    if section_img is not None:
                        analysis_text = processor.process_section(section_img, i)
                        outputs.append(analysis_text)
                    else:
                        outputs.append("")
                
                # Fill remaining outputs if needed
                while len(outputs) < 6:  # 6 sections
                    outputs.append("")
                
                return outputs
                
            except Exception as e:
                return [""] * 6

        def reset_interface():
            return [
                None, gr.update(interactive=False), [], 
                *([gr.update(visible=False), None, gr.update(visible=False), 
                   "", gr.update(visible=False), "", gr.update(visible=False)] * 6)
            ]

        # Connect components
        detect_outputs = [
            debug_image, process_btn, section_images_store
        ]
        for section in section_containers:
            detect_outputs.extend([
                section['container'],
                section['image'],
                section['image'],
                section['prompt'],
                section['prompt'],
                section['output'],
                section['output']
            ])

        process_outputs = []
        for section in section_containers:
            process_outputs.append(section['output'])

        detect_btn.click(
            fn=detect_sections,
            inputs=[input_image],
            outputs=detect_outputs
        )
        
        process_btn.click(
            fn=process_sections,
            inputs=[input_image, section_images_store],
            outputs=process_outputs
        )
        
        stop_btn.click(
            fn=reset_interface,
            outputs=detect_outputs
        )
        
    return demo

if __name__ == "__main__":
    demo = create_interface()
    demo.launch(share=True)
