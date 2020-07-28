import logging
from .sysu_eval import do_sysu_evaluation
from .sysu_eval_reid import do_sysu_evaluation_reid


def sysu_evaluation(dataset, predictions, output_folder, box_only, **_):
    logger = logging.getLogger("maskrcnn_benchmark.inference")
    if box_only:
        logger.warning("sysu evaluation doesn't support box_only, ignored.")
    logger.info("performing sysu evaluation---detection, ignored iou_types.")
    return do_sysu_evaluation(
        dataset=dataset,
        predictions=predictions,
        output_folder=output_folder,
        logger=logger,
    )


def sysu_evaluation_reid(dataset, predictions, output_folder, box_only, **_):
    logger = logging.getLogger("maskrcnn_benchmark.inference")
    if box_only:
        logger.warning("sysu evaluation doesn't support box_only, ignored.")
    logger.info("performing sysu evaluation---person search, ignored iou_types.")

    return do_sysu_evaluation_reid(
        dataset=dataset,
        predictions=predictions,
        output_folder=output_folder,
        logger=logger,
    )