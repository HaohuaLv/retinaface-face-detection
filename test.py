import timm
from model import YoloModel
from transformers import PretrainedConfig
# model = YoloModel._from_config(PretrainedConfig.from_json_file("config.json"))

# print(model.reductions)
# # print(model.)
print(timm.list_models("mobile*"))