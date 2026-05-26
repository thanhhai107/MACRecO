import os
import pandas as pd
from abc import abstractmethod
from tqdm import tqdm
from typing import Any
from loguru import logger
from argparse import ArgumentParser

from macrec.llms.basellm import BaseLLM, SampleDeferredError
from macrec.tasks.base import Task
from macrec.utils import init_openai_api, read_json
from macrec.utils.prompt_builder import PromptBuilder
from macrec.utils.token_tracker import token_tracker
from macrec.systems import CollaborationSystem

class GenerationTask(Task):
    bad_sample_retry_threshold = 12
    bad_sample_queue_passes = 1

    @staticmethod
    def parse_task_args(parser: ArgumentParser) -> ArgumentParser:
        parser.add_argument('--api_config', type=str, default='config/api-config.json', help='Api configuration file')
        parser.add_argument('--dataset', type=str, default='None', help='Dataset name')
        parser.add_argument('--data_file', type=str, required=True, help='Dataset file')
        parser.add_argument('--system', type=str, default='collaboration', choices=['collaboration'], help='System name')
        parser.add_argument('--system_config', type=str, required=True, help='System configuration file')
        parser.add_argument('--task', type=str, default='rp', choices=['rp', 'sr', 'gen'], help='Task name')
        parser.add_argument('--max_his', type=int, default=10, help='Max history length')
        return parser

    def get_data(self, data_file: str, max_his: int) -> pd.DataFrame:
        """Load minimal CSV data and initialize prompt builder."""
        df = pd.read_csv(data_file)
        
        # Initialize prompt builder for on-demand text formatting
        data_dir = os.path.dirname(data_file)
        self.prompt_builder = PromptBuilder(data_dir, self.dataset)
        
        # For SR tasks, determine n_candidate from first row
        if self.task == 'sr' and 'candidate_item_id' in df.columns:
            # Parse first candidate list to get count
            import ast
            first_candidates = df['candidate_item_id'].iloc[0]
            if isinstance(first_candidates, str):
                try:
                    first_candidates = ast.literal_eval(first_candidates)
                except:
                    pass
            
            if isinstance(first_candidates, list):
                self.n_candidate = len(first_candidates)
                self.system_kwargs['n_candidate'] = self.n_candidate
                logger.info(f"Detected {self.n_candidate} candidates for SR task")
        
        return df

    def prompt_data(self, df: pd.DataFrame) -> list[tuple[str, int | float | str, pd.Series]]:
        """Build prompts on-demand from minimal CSV data."""
        import ast
        
        # First pass: Filter samples where GT not in candidates (BEFORE building prompts)
        if self.task in ['sr', 'rr'] and 'candidate_item_id' in df.columns:
            logger.info(f"Pre-filtering samples for {self.task} task...")
            valid_indices = []
            skipped_count = 0
            
            for i in range(len(df)):
                row = df.iloc[i]
                gt_item = row['item_id']
                candidate_ids = row['candidate_item_id']
                
                # Parse candidate list if it's a string
                if isinstance(candidate_ids, str):
                    try:
                        candidate_ids = ast.literal_eval(candidate_ids)
                    except:
                        logger.warning(f"Failed to parse candidate_item_id for sample {i+1}, skipping")
                        skipped_count += 1
                        continue
                
                # Check if GT in candidates
                if isinstance(candidate_ids, list):
                    if gt_item in candidate_ids:
                        valid_indices.append(i)
                    else:
                        logger.trace(f"Skipping sample {i+1} (User {row['user_id']}): GT item {gt_item} not in candidates")
                        skipped_count += 1
                else:
                    # If candidates not a list, include by default
                    valid_indices.append(i)
            
            # Filter dataframe to only valid samples
            if skipped_count > 0:
                logger.warning(f"Pre-filtered: Skipped {skipped_count}/{len(df)} samples where GT item not in candidates")
                df = df.iloc[valid_indices].reset_index(drop=True)
                logger.success(f"Building prompts for {len(df)} valid samples (filtered from {len(df) + skipped_count} total)")
            else:
                logger.info(f"All {len(df)} samples have GT in candidates")
        
        # Second pass: Build prompts only for valid samples
        data_prompt = self.system.prompts['data_prompt']
        prompts = []
        
        logger.info(f"Building prompts for {len(df)} samples...")
        
        for i in tqdm(range(len(df)), desc="Building prompts"):
            row = df.iloc[i]
            
            # Build formatted fields on-demand
            fields = self.prompt_builder.build_prompt_fields(row, max_his=self.max_his)
            
            # Build prompt based on task type
            if self.task == 'rp':
                prompt = data_prompt.format(
                    user_id=row['user_id'],
                    user_profile=fields['user_profile'],
                    history=fields['history'],
                    target_item_id=row['item_id'],
                    target_item_attributes=fields['target_item_attributes']
                )
                target = row['rating']
            
            elif self.task == 'sr':
                prompt = data_prompt.format(
                    user_id=row['user_id'],
                    user_profile=fields['user_profile'],
                    history=fields['history'],
                    candidate_item_attributes=fields['candidate_item_attributes']
                )
                target = row['item_id']
            
            elif self.task == 'rr':
                # Retrieve & Rank: no candidate list in CSV; candidates provided by retrieval
                prompt = data_prompt.format(
                    user_id=row['user_id'],
                    user_profile=fields['user_profile'],
                    history=fields['history']
                )
                target = row['item_id']
            
            elif self.task == 'gen':
                prompt = data_prompt.format(
                    user_id=row['user_id'],
                    user_profile=fields['user_profile'],
                    history=fields['history'],
                    target_item_id=row['item_id'],
                    target_item_attributes=fields['target_item_attributes'],
                    rating=row['rating']
                )
                target = row['rating']
            
            else:
                raise NotImplementedError(f"Task {self.task} not implemented")
            
            prompts.append((prompt, target, row))
        
        logger.info(f"Built {len(prompts)} prompts")
        return prompts

    def get_system(self, system: str, system_config: str):
        if system == 'collaboration':
            self.system = CollaborationSystem(config_path=system_config, **self.system_kwargs)
        else:
            raise NotImplementedError

    @property
    @abstractmethod
    def running_steps(self) -> int:
        """Return the steps to run for each trial.

        Raises:
            `NotImplementedError`: Subclasses should implement this method.
        Returns:
            `int`: The steps to run for each trial.
        """
        raise NotImplementedError

    @abstractmethod
    def before_generate(self) -> None:
        """The process to run before generating.

        Raises:
            `NotImplementedError`: Subclasses should implement this method.
        """
        raise NotImplementedError

    @abstractmethod
    def after_step(self, answer: Any, gt_answer: int | float | str, step: int, record: dict) -> None:
        """The process to run after each system step during one trial.

        Args:
            `answer` (`Any`): The answer given by the system.
            `gt_answer` (`int | float | str`): The ground truth answer.
            `step` (`int`): The current step. Starts from 0.
            `record` (`dict`): The record of the current trial. Can be used to store intermediate results.
        Raises:
            `NotImplementedError`: Subclasses should implement this method.
        """
        raise NotImplementedError

    @abstractmethod
    def after_iteration(self, answer: Any, gt_answer: int | float | str, record: dict, pbar: tqdm) -> None:
        """The process to run after each trial.

        Args:
            `answer` (`Any`): The final answer given by the system.
            `gt_answer` (`int | float | str`): The ground truth answer.
            `record` (`dict`): The record of the current trial. Can be used to store intermediate results.
            `pbar` (`tqdm`): The progress bar. Can be used to update the information of the progress bar.
        Raises:
            `NotImplementedError`: Subclasses should implement this method.
        """
        raise NotImplementedError

    @abstractmethod
    def after_generate(self) -> None:
        """The process to run after generating.

        Raises:
            `NotImplementedError`: Subclasses should implement this method.
        """
        raise NotImplementedError

    def _iter_llms(self):
        seen: set[int] = set()
        agents = getattr(getattr(self, 'system', None), 'agents', {})
        for agent in agents.values():
            for value in vars(agent).values():
                if isinstance(value, BaseLLM) and id(value) not in seen:
                    seen.add(id(value))
                    yield value

    def _reset_sample_llm_metrics(self) -> None:
        for llm in self._iter_llms():
            llm.reset_sample_metrics()

    def _max_sample_retry_count(self) -> int:
        retry_counts = [getattr(llm, 'sample_retry_count', 0) for llm in self._iter_llms()]
        return max(retry_counts, default=0)

    @staticmethod
    def _sample_label(data_sample: pd.Series) -> str:
        parts = []
        for key in ('user_id', 'item_id'):
            if key in data_sample:
                parts.append(f"{key}={data_sample[key]}")
        return ', '.join(parts) if parts else f"index={getattr(data_sample, 'name', 'unknown')}"

    def _process_generation_sample(
        self,
        test_data: str,
        gt_answer: int | float | str,
        data_sample: pd.Series,
        steps: int,
        pbar: tqdm,
        final_attempt: bool = False,
    ) -> tuple[bool, str | None]:
        record = dict()
        self._reset_sample_llm_metrics()
        token_tracker.start_sample()

        try:
            self.system.set_data(input=test_data, context="", gt_answer=gt_answer, data_sample=data_sample)
            self.system.reset(clear=True)
            for i in range(steps):
                logger.debug(f'===================================Running step {i}...===================================')
                self.after_step(answer=self.system(), gt_answer=gt_answer, step=i, record=record)

            max_retry_count = self._max_sample_retry_count()
            if not final_attempt and max_retry_count > self.bad_sample_retry_threshold:
                return False, f"sample used {max_retry_count} LLM retries"

            self.after_iteration(answer=self.system.answer, gt_answer=gt_answer, record=record, pbar=pbar)
            return True, None
        except SampleDeferredError as e:
            if final_attempt:
                logger.error(f"Error processing deferred sample: {e}. Skipping this sample.")
                return True, None
            return False, str(e)
        except Exception as e:
            if final_attempt:
                logger.error(f"Error processing deferred sample: {e}. Skipping this sample.")
                return True, None
            return False, str(e)
        finally:
            token_tracker.end_sample()

    def generate(self, data: list[tuple[str, int | float | str, pd.Series]], steps: int = 2):
        self.before_generate()
        
        # Start tracking duration
        token_tracker.start_tracking()
        
        bad_sample_queue = []
        with tqdm(total=len(data)) as pbar:
            for test_data, gt_answer, data_sample in data:
                completed, reason = self._process_generation_sample(
                    test_data=test_data,
                    gt_answer=gt_answer,
                    data_sample=data_sample,
                    steps=steps,
                    pbar=pbar,
                    final_attempt=False,
                )
                if completed:
                    pbar.update(1)
                else:
                    bad_sample_queue.append((test_data, gt_answer, data_sample, reason))
                    logger.warning(
                        f"Queued bad sample for later ({self._sample_label(data_sample)}): {reason}"
                    )

            for queue_pass in range(self.bad_sample_queue_passes):
                if not bad_sample_queue:
                    break

                queued_samples = bad_sample_queue
                bad_sample_queue = []
                logger.warning(
                    f"Retrying {len(queued_samples)} queued bad samples "
                    f"(pass {queue_pass + 1}/{self.bad_sample_queue_passes})"
                )

                for test_data, gt_answer, data_sample, reason in queued_samples:
                    logger.info(
                        f"Retrying queued sample ({self._sample_label(data_sample)}). "
                        f"Original reason: {reason}"
                    )
                    completed, retry_reason = self._process_generation_sample(
                        test_data=test_data,
                        gt_answer=gt_answer,
                        data_sample=data_sample,
                        steps=steps,
                        pbar=pbar,
                        final_attempt=True,
                    )
                    pbar.update(1)
                    if not completed:
                        logger.error(
                            f"Queued sample still failed ({self._sample_label(data_sample)}): "
                            f"{retry_reason}. Skipping this sample."
                        )
        
        # End tracking duration
        token_tracker.end_tracking()
        
        self.after_generate()

    def run(self, api_config: str, dataset: str, data_file: str, system: str, system_config: str, task: str, max_his: int):
        if dataset == 'None':
            dataset = os.path.basename(os.path.dirname(data_file))
        self.dataset = dataset
        self.task = task
        self.max_his = max_his
        self.system_kwargs = {
            'task': self.task,
            'leak': False,
            'dataset': self.dataset,
        }
        init_openai_api(read_json(api_config))
        data_df = self.get_data(data_file, max_his)
        self.get_system(system, system_config)
        data = self.prompt_data(data_df)
        self.generate(data, steps=self.running_steps)
