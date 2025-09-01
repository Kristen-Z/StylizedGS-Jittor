import torch
import jittor as jt
clip = torch.load('vgg16.pth')

for k in clip.keys():
    clip[k] = clip[k].float().cpu()
jt.save(clip, '/.cache/jittor/jt1.3.9/g++11.4.0/py3.10.16/Linux-6.8.0-60xb3/IntelRXeonRSilx06/2237/default/cu11.8.89_sm_80_86/checkpoints/vgg16.pkl')