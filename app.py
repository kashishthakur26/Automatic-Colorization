import torch
import matplotlib.pyplot as plt
import numpy as np
import gradio as gr


from src import eccv16, siggraph17, load_img, preprocess_img, postprocess_tens


colorizer_eccv16 = eccv16(pretrained=True).eval()
colorizer_siggraph17 = siggraph17(pretrained=True).eval()

use_gpu = torch.cuda.is_available()

if use_gpu:
    colorizer_eccv16.cuda()
    colorizer_siggraph17.cuda()

def colorize_image(input_img):
    img = np.array(input_img)
    (tens_l_orig, tens_l_rs) = preprocess_img(img, HW=(256,256))

    if use_gpu:
        tens_l_rs = tens_l_rs.cuda()

    img_bw = postprocess_tens(tens_l_orig, torch.cat((0 * tens_l_orig, 0*tens_l_orig ), dim =1))
    out_img_eccv16 = postprocess_tens(tens_l_orig, colorizer_eccv16(tens_l_rs).cpu())
    out_img_siggraph17 = postprocess_tens(tens_l_orig, colorizer_siggraph17(tens_l_rs).cpu())

    return [img_bw, out_img_eccv16, out_img_siggraph17]

demo = gr.Interface(
    fn=colorize_image,
    inputs=gr.Image(type="numpy"),
    outputs=[gr.Image(type="numpy", label="Black & White"),
             gr.Image(type="numpy", label="ECCV 2016 Colorization"),
             gr.Image(type="numpy", label="SIGGRAPH 2017 Colorization")]
)


if __name__ == "__main__":
    demo.launch()

