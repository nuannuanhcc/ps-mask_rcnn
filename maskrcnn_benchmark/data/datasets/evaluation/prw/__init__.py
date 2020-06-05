import logging
from .prw_eval import do_prw_evaluation
from .prw_eval_reid import do_prw_evaluation_reid


def prw_evaluation(dataset, predictions, output_folder, box_only, **_):
    logger = logging.getLogger("maskrcnn_benchmark.inference")
    if box_only:
        logger.warning("prw evaluation doesn't support box_only, ignored.")
    logger.info("performing prw evaluation---detection, ignored iou_types.")
    return do_prw_evaluation(
        dataset=dataset,
        predictions=predictions,
        output_folder=output_folder,
        logger=logger,
    )


def prw_evaluation_reid(dataset, predictions, output_folder, box_only, **_):
    logger = logging.getLogger("maskrcnn_benchmark.inference")
    if box_only:
        logger.warning("prw evaluation doesn't support box_only, ignored.")
    logger.info("performing prw evaluation---person search, ignored iou_types.")
    return do_prw_evaluation_reid(
        dataset=dataset,
        predictions=predictions,
        output_folder=output_folder,
        logger=logger,
    )
