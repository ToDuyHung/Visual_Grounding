from typing import List
from PIL import Image
from io import BytesIO
import base64

from TMTChatbot import (
    BaseDataModel,
    BaseServiceSingleton
)

from config.config import Config
from process.visual_grounding import VisualGrounding
from common.common_keys import *


class Processor(BaseServiceSingleton):
    def __init__(self, config: Config = None):
        super(Processor, self).__init__(config=config)
        self.model_visual_grounding = VisualGrounding(config=config)

    def process(self, input_data: BaseDataModel):
        # if not input_data.data['img']:
        #     input_data.data['img'] = test_img()
        # object_texts = self.extract_objects(input_data)
        # input_data.data = self.vqa_inference(input_data, object_texts)
        
        image = Image.open(BytesIO(base64.urlsafe_b64decode(input_data.data['img']))).convert('RGB')
        text = input_data.data['text']
        start_point, end_point = self.model_visual_grounding.get_bounding_box(text, image)
        input_data.data = {'start_point':start_point, 'end_point': end_point}
        return input_data
    
    # def extract_objects(self, input_data: BaseDataModel):
    #     return self.model_object_detection.extract_object(input_data.data['img'])
    
    # def vqa_inference(self, input_data: BaseDataModel, object_texts: str):
    #     received_data = {k: v for k, v in input_data.data.items() if k not in ['img_type', 'img']}
    #     base64_str = input_data.data['img']
    #     return self.vqa_utils.vqa_inference(received_data, base64_str, object_texts)