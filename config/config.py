import os

from TMTChatbot.Common.common_keys import *
from TMTChatbot.Common.config import Config as BaseConfig

from common.common_keys import *


class Config(BaseConfig):
    def __init__(self):

        super(Config, self).__init__(
            kafka_consume_topic=os.getenv(KAFKA_CONSUME_TOPIC, 'message'),
            kafka_publish_topic=os.getenv(KAFKA_PUBLISH_TOPIC, 'node_search_message'),
            kafka_bootstrap_servers=os.getenv(KAFKA_BOOTSTRAP_SERVERS, '172.29.13.24:35000'),
            kafka_auto_offset_reset=os.getenv(KAFKA_AUTO_OFFSET_RESET, 'earliest'),
            kafka_group_id=os.getenv(KAFKA_GROUP_ID, 'NODE_SEARCH'),
            max_process_workers=int(os.getenv(MAX_PROCESS_WORKERS, 3))
        )
        self.api_port = int(os.getenv(API_PORT, 35515))
        
        self.bpe_dir =  os.getenv(BPE_DIR, "model/utils/BPE")
        self.visual_grounding_checkpoint = os.getenv(VISUAL_GROUNDING_CHECKPOINT, "model/checkpoints/visual_grounding/refcoco_base_best.pt")
