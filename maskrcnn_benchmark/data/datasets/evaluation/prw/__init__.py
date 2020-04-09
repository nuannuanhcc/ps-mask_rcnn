import logging
from .prw_eval import do_prw_evaluation



def prw_evaluation(dataset, predictions, output_folder, box_only, **_):
    logger = logging.getLogger("maskrcnn_benchmark.inference")
    if box_only:
        logger.warning("prw evaluation doesn't support box_only, ignored.")
    logger.info("performing prw evaluation, ignored iou_types.")
    return do_prw_evaluation(
        dataset=dataset,
        predictions=predictions,
        output_folder=output_folder,
        logger=logger,
    )
