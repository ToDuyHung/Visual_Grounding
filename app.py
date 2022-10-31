from TMTChatbot.Common.utils.logging_utils import setup_logging
from TMTChatbot.ServiceWrapper import BaseApp

from config.config import Config
from process.processor import Processor


class App(BaseApp):
    def __init__(self, config: Config = None):
        super(App, self).__init__(config=config)
        self.processor = Processor(config=config)
        # self.add_process_function(self.processor.process)
        # self.api_app.add_endpoint("/vg_api",
        #                           self.processor.process, methods=["POST"])
        self.api_app.add_endpoint("/vg_api",
                                  func=self.processor.process,
                                  methods=["POST"],
                                  description="Visual grounding",
                                  use_thread=False,
                                  use_async=False,
                                  # request_data_model=BaseDataModel,
                                  # response_data_model=BaseDataModel,
                                  tags=["Visual Grounding"])

def create_app(multiprocess: bool = False):
    _config = Config()
    setup_logging(logging_folder=_config.logging_folder, log_name=_config.log_name)
    if multiprocess:
        raise NotImplementedError("Multiprocessing app not created")
    else:
        _app = App(config=_config)
    return _app

main_app = create_app(False)
app = main_app.api_app.app

# if __name__ == "__main__":
    # _config = Config()
    # setup_logging(logging_folder=_config.logging_folder, log_name=_config.log_name)
    # app = App(config=_config)
    # app.start()
    # app.join()
