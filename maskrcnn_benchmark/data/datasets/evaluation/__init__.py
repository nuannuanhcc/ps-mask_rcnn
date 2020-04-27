from maskrcnn_benchmark.data import datasets

from .sysu import sysu_evaluation, sysu_evaluation_reid
from .prw import prw_evaluation, prw_evaluation_reid
from .coco import coco_evaluation
from .voc import voc_evaluation
from .cityscapes import abs_cityscapes_evaluation

def evaluate(dataset, predictions, output_folder, **kwargs):
    """evaluate dataset using different methods based on dataset type.
    Args:
        dataset: Dataset object
        predictions(list[BoxList]): each item in the list represents the
            prediction results for one image.
        output_folder: output folder, to save evaluation files or results.
        **kwargs: other args.
    Returns:
        evaluation result
    """
    args = dict(
        dataset=dataset, predictions=predictions, output_folder=output_folder, **kwargs
    )
    if isinstance(dataset, datasets.COCODataset):
        return coco_evaluation(**args)
    elif isinstance(dataset, datasets.PascalVOCDataset):
        return voc_evaluation(**args)
    elif isinstance(dataset, datasets.AbstractDataset):
        return abs_cityscapes_evaluation(**args)
    elif isinstance(dataset, datasets.SYSUDataset):
        return sysu_evaluation(**args)
    elif isinstance(dataset, datasets.PRWDataset):
        return prw_evaluation(**args)
    elif isinstance(dataset, (list, tuple)) and isinstance(dataset[0], datasets.SYSUDataset):
        return sysu_evaluation_reid(**args)
    elif isinstance(dataset, (list, tuple)) and isinstance(dataset[0], datasets.PRWDataset):
        return prw_evaluation_reid(**args)
    else:
        dataset_name = dataset.__class__.__name__
        raise NotImplementedError("Unsupported dataset type {}.".format(dataset_name))
