from pix2text import Pix2Text, merge_line_texts

image_fps = [
    'example.png'
]
p2t = Pix2Text.from_config()
outs = p2t.recognize_formula(image_fps)  # recognize pure formula images

outs2 = p2t.recognize('example.png', file_type='text_formula', return_text=True, save_analysis_res='example_out.png')  # recognize mixed images
print(outs2)
