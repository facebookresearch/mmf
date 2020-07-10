# Copyright (c) Facebook, Inc. and its affiliates.

from abc import ABC
from typing import Any, Dict, Tuple, Type

import torch
import tqdm

from mmf.common.meter import Meter
from mmf.common.report import Report
from mmf.utils.distributed import is_master


class MultiTaskTrainerEvaluationLoopMixin(ABC):
    def evaluation_loop(
        self, loader, use_tqdm: bool = False, single_batch: bool = False
    ) -> Tuple[Dict[str, Any], Type[Meter]]:

        meters = {}
        combined_reports = {}
        for task in self.config.multi_task_config.tasks:
            task_id = task.id
            meter = Meter()
            meters[task_id] = meter
            combined_reports[task_id] = None

        with torch.no_grad():
            self.model.eval()
            disable_tqdm = not use_tqdm or not is_master()

            for batch in tqdm.tqdm(loader, disable=disable_tqdm):

                task_id = self.dataset_to_taskid[batch.dataset_name]
                meter = meters[task_id]

                report = self._forward(batch)
                self.update_meter(report, meter)

                # accumulate necessary params for metric calculation
                if combined_reports[task_id] is None:
                    combined_reports[task_id] = report
                else:
                    combined_reports[task_id].accumulate_tensor_fields(
                        report, self.metrics.required_params
                    )
                    combined_reports[task_id].batch_size += report.batch_size

                if single_batch is True:
                    break

            meter = Meter()
            for task_id in combined_reports.keys():
                combined_reports[task_id].metrics = self.metrics(
                    combined_reports[task_id], combined_reports[task_id]
                )
                self.update_meter(
                    combined_reports[task_id], meters[task_id], eval_mode=True
                )

                self.update_meter(combined_reports[task_id], meter, eval_mode=True)
            # enable train mode again
            self.model.train()

        return combined_reports, meters, meter

    def prediction_loop(self, dataset_type: str) -> None:
        reporter = self.dataset_loader.get_test_reporter(dataset_type)
        with torch.no_grad():
            self.model.eval()
            message = f"Starting {dataset_type} inference predictions"
            self.writer.write(message)

            while reporter.next_dataset():
                dataloader = reporter.get_dataloader()

                for batch in tqdm.tqdm(dataloader):
                    prepared_batch = reporter.prepare_batch(batch)
                    model_output = self.model(prepared_batch)
                    report = Report(prepared_batch, model_output)
                    reporter.add_to_report(report, self.model)

            self.writer.write("Finished predicting")
            self.model.train()
