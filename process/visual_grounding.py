import torch
import cv2
import numpy as np
from fairseq import utils, tasks
from fairseq import checkpoint_utils
from process.services.utils.eval_utils import eval_step
from process.services.tasks.mm_tasks.refcoco import RefcocoTask
from process.services.models.ofa import OFAModel
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO
from torchvision import transforms
import base64
import requests
import re
import pickle
from tqdm import tqdm
from glob import glob

from TMTChatbot import BaseServiceSingleton
from config.config import Config
from common.common_keys import *

class VisualGrounding(BaseServiceSingleton):
    def __init__(self, config: Config = None):
        super(VisualGrounding, self).__init__(config=config)
        # Register refcoco task
        tasks.register_task('refcoco', RefcocoTask)
        # turn on cuda if GPU is available
        self.use_cuda = torch.cuda.is_available()
        # use fp16 only when GPU is available
        self.use_fp16 = False
        # Load pretrained ckpt & config
        overrides={"bpe_dir": config.bpe_dir}
        self.models, self.cfg, self.task = checkpoint_utils.load_model_ensemble_and_task(
                utils.split_paths(config.visual_grounding_checkpoint),
                arg_overrides=overrides
            )
        self.cfg.common.seed = 7
        self.cfg.generation.beam = 5
        self.cfg.generation.min_len = 4
        self.cfg.generation.max_len_a = 0
        self.cfg.generation.max_len_b = 4
        self.cfg.generation.no_repeat_ngram_size = 3

        # Fix seed for stochastic decoding
        if self.cfg.common.seed is not None and not self.cfg.generation.no_seed_provided:
            np.random.seed(self.cfg.common.seed)
            utils.set_torch_seed(self.cfg.common.seed)

        # Move models to GPU
        self.model = None
        for model in self.models:
            model.eval()
            if self.use_fp16:
                model.half()
            if self.use_cuda and not self.cfg.distributed_training.pipeline_model_parallel:
                model.cuda()
            model.prepare_for_inference_(self.cfg)
            self.model = model
            
        # Initialize generator
        self.generator = self.task.build_generator(self.models, self.cfg.generation)

        # Image transform
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]

        self.patch_resize_transform = transforms.Compose([
            lambda image: image.convert("RGB"),
            transforms.Resize((self.cfg.task.patch_image_size, self.cfg.task.patch_image_size), interpolation=Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

        # Text preprocess
        self.bos_item = torch.LongTensor([self.task.src_dict.bos()])
        self.eos_item = torch.LongTensor([self.task.src_dict.eos()])
        self.pad_idx = self.task.src_dict.pad()

        # Construct input for refcoco task
        self.patch_image_size = self.cfg.task.patch_image_size
    
    def encode_text(self, text, length=None, append_bos=False, append_eos=False):
        s = self.task.tgt_dict.encode_line(
            line=self.task.bpe.encode(text.lower()),
            add_if_not_exist=False,
            append_eos=False
        ).long()
        if length is not None:
            s = s[:length]
        if append_bos:
            s = torch.cat([self.bos_item, s])
        if append_eos:
            s = torch.cat([s, self.eos_item])
        return s
    
    def construct_sample(self, image: Image, text: str):
        w, h = image.size
        w_resize_ratio = torch.tensor(self.patch_image_size / w).unsqueeze(0)
        h_resize_ratio = torch.tensor(self.patch_image_size / h).unsqueeze(0)
        patch_image = self.patch_resize_transform(image).unsqueeze(0)
        patch_mask = torch.tensor([True])
        src_text = self.encode_text(' which region does the text " {} " describe?'.format(text), append_bos=True, append_eos=True).unsqueeze(0)
        src_length = torch.LongTensor([s.ne(self.pad_idx).long().sum() for s in src_text])
        sample = {
            "id":np.array(['42']),
            "net_input": {
                "src_tokens": src_text,
                "src_lengths": src_length,
                "patch_images": patch_image,
                "patch_masks": patch_mask,
            },
            "w_resize_ratios": w_resize_ratio,
            "h_resize_ratios": h_resize_ratio,
            "region_coords": torch.randn(1, 4)
        }
        return sample
  
    # Function to turn FP32 to FP16
    def apply_half(t):
        if t.dtype is torch.float32:
            return t.to(dtype=torch.half)
        return t
    
    def get_bounding_box(self, text, image):
        # Construct input sample & preprocess for GPU if cuda available
        sample = self.construct_sample(image, text)
        sample = utils.move_to_cuda(sample) if self.use_cuda else sample
        sample = utils.apply_to_sample(apply_half, sample) if self.use_fp16 else sample
        
        # Run eval step for refcoco
        with torch.no_grad():
            result, scores = eval_step(self.task, self.generator, self.models, sample)
        
        return (int(result[0]["box"][0]), int(result[0]["box"][1])), (int(result[0]["box"][2]), int(result[0]["box"][3]))