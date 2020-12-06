from framework.dataloader.Dataset import BaseDataset

class IndoorFlying(BaseDataset):
    # CONSTANT
    FRAMES_FILTER_FOR_TRAINING = {
        'indoor_flying': {
            1: list(range(80, 1260)),
            2: list(range(160, 1580)),
            3: list(range(125, 1815)),
            4: list(range(190, 290))
        }
    }
    FRAMES_FILTER_FOR_TEST = {
        'indoor_flying': {
            1: list(range(140, 1201)),
            2: list(range(120, 1421)),
            3: list(range(73, 1616)),
            4: list(range(190, 290))
        }
    }

    def __init__(self, examples, transform_list, *args, **kwargs):
        super().__init__()
        print(args)
        self._examples = examples
        self._transformers = transform_list

    @classmethod
    def datasetFactory(cls, *args, **kwargs):

        pass

