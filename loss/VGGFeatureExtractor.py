import os
import glob

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")	
Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor

def load_model(model_name, model_dir):
    model  = eval('models.%s(init_weights=False)' % model_name)
    path_format = os.path.join(model_dir, '%s-[a-z0-9]*.pth' % model_name)
    
    model_path = glob.glob(path_format)[0]
    
    model.load_state_dict(torch.load(model_path))
    return model

class VGGFeatureExtractor(nn.Module):
	def __init__(self):
		super(VGGFeatureExtractor, self).__init__()
		vgg19_model = load_model('vgg19' , '/home/sayam.choudhary.cse17.iitbhu/HNAS-SR-master/src/loss/')
		self.vgg19_54 = nn.Sequential(*list(vgg19_model.features.children())[:35])
		self.mean = Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
		self.std = Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

	def forward(self, img):
		img = (img - self.mean) / self.std
		return self.vgg19_54(img)
