from framework.dataloader.transforms import *

"""
All transform methods are registered here. If the default transform method path is here, import the transform method.
"""
TRANSFORM = {}
TRANSFORM['ToTensor'] = ToTensor
TRANSFORM['ToNumpy'] = ToNumpy