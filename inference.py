import net
import torch
import os
# from face_alignment import align
import numpy as np
from PIL import Image
import time


adaface_models = {
    'ir_50':"pretrained/adaface_ir50_ms1mv2.ckpt",
}

def load_pretrained_model(architecture='ir_50'):
    # load model and pretrained statedict
    assert architecture in adaface_models.keys()
    model = net.build_model(architecture)

    # TODO Change to `GPU` when running on `MetaCentrum`.
    statedict = torch.load(adaface_models[architecture], map_location=torch.device('cpu'))['state_dict']

    model_statedict = {key[6:]:val for key, val in statedict.items() if key.startswith('model.')}
    model.load_state_dict(model_statedict)
    model.eval()
    return model

def to_input(pil_rgb_image):
    np_img = np.array(pil_rgb_image)
    brg_img = ((np_img[:,:,::-1] / 255.) - 0.5) / 0.5
    tensor = torch.tensor([brg_img.transpose(2,0,1)]).float()
    return tensor

if __name__ == '__main__':

    model = load_pretrained_model('ir_50')
    feature, norm = model(torch.randn(2,3,112,112))

    test_image_path = 'data/tmp'
    features = []
    for fname in sorted(os.listdir(test_image_path)):
        start = time.time()

        path = os.path.join(test_image_path, fname)
        aligned_rgb_img = Image.open(path)
        bgr_tensor_input = to_input(aligned_rgb_img)
        feature, _ = model(bgr_tensor_input)
        features.append(feature)

        print(time.time() - start)

    print(torch.cat(features))
    print(torch.cat(features).shape)

    similarity_scores = torch.cat(features) @ torch.cat(features).T
    print(similarity_scores)
    

