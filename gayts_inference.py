import pystiche
from pystiche import demo, enc, loss, optim
from pystiche.image import show_image
from pystiche.misc import get_input_image
import utils
import cv2
import os

device = "cuda"

multi_layer_encoder = enc.vgg19_multi_layer_encoder()

content_layer = "relu4_2"
content_encoder = multi_layer_encoder.extract_encoder(content_layer)
content_weight = 1e0
content_loss = loss.FeatureReconstructionLoss(
    content_encoder, score_weight=content_weight
)

style_layers = ("relu1_1", "relu2_1", "relu3_1", "relu4_1", "relu5_1")
style_weight = 1e3

def get_style_op(encoder, layer_weight):
    return loss.GramLoss(encoder, score_weight=layer_weight)

style_loss = loss.MultiLayerEncodingLoss(
    multi_layer_encoder, style_layers, get_style_op, score_weight=style_weight,
)

perceptual_loss = loss.PerceptualLoss(content_loss, style_loss).to(device)

content_img = utils.prepare_img("data/golden_gate.jpg", [600, 800], device)
style_img = utils.prepare_img("data/styles/8.jpg", [600, 800], device)

perceptual_loss.set_content_image(content_img)
perceptual_loss.set_style_image(style_img)

starting_point = "content"
input_image = get_input_image(starting_point, content_image=content_img)

output_image = optim.image_optimization(input_image, perceptual_loss, num_steps=1000)

show_image(output_image)

print(output_image.max())
print(output_image.min())

cv2.imwrite(os.path.join('results', 'pystiche_output.png'), output_image.cpu().squeeze().permute(1, 2, 0).numpy()[:,:,::-1])
