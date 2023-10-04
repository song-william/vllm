"""A GPU worker class."""
import os
from typing import Dict, List, Tuple, Optional

import torch
import torch.distributed

from vllm.config import (CacheConfig, ModelConfig, ParallelConfig,
                         SchedulerConfig)
from vllm.model_executor import get_model, InputMetadata, set_random_seed
from vllm.model_executor.parallel_utils.parallel_state import (
    initialize_model_parallel)
from vllm.sampling_params import SamplingParams
from vllm.sequence import DraftOutput, DraftOutputs, SamplerOutput, SequenceData, SequenceGroupMetadata
from vllm.worker.cache_engine import CacheEngine
from vllm.utils import get_gpu_memory

DRAFT_LENGTH = 4

class Worker:
    """A worker class that executes (a partition of) the model on a GPU.

    Each worker is associated with a single GPU. The worker is responsible for
    maintaining the KV cache and executing the model on the GPU. In case of
    distributed inference, each worker is assigned a partition of the model.
    """

    def __init__(
        self,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
        scheduler_config: SchedulerConfig,
        rank: Optional[int] = None,
        distributed_init_method: Optional[str] = None,
        draft: bool = False,
    ) -> None:
        self.model_config = model_config
        self.draft = draft
        self.parallel_config = parallel_config
        self.scheduler_config = scheduler_config
        print(f"worker.py: scheduler_config: {self.scheduler_config.__dict__}")
        self.rank = rank
        self.distributed_init_method = distributed_init_method

        # Uninitialized cache engine. Will be initialized by
        # self.init_cache_engine().
        self.cache_config = None
        self.block_size = None
        self.cache_engine = None
        self.cache_events = None
        self.gpu_cache = None

    def init_model(self):
        # This env var set by Ray causes exceptions with graph building.
        os.environ.pop("NCCL_ASYNC_ERROR_HANDLING", None)
        # Env vars will be set by Ray.
        self.rank = self.rank if self.rank is not None else int(
            os.getenv("RANK", "-1"))
        local_rank = int(os.getenv("LOCAL_RANK", "0"))
        self.device = torch.device(f"cuda:{local_rank}")
        if self.rank < 0:
            raise ValueError("Invalid or unspecified rank.")
        torch.cuda.set_device(self.device)
        print(f"init_model rank {local_rank}, os.getenv rank: {os.getenv('LOCAL_RANK')}, device: {self.device}")

        # Initialize the distributed environment.
        _init_distributed_environment(self.parallel_config, self.rank,
                                      self.draft,
                                      (1 if self.draft else None),
                                      self.distributed_init_method)

        # Initialize the model.
        set_random_seed(self.model_config.seed)
        self.model = get_model(self.model_config, draft=self.draft)

    @torch.inference_mode()
    def profile_num_available_blocks(
        self,
        block_size: int,
        gpu_memory_utilization: float,
        cpu_swap_space: int,
    ) -> Tuple[int, int]:
        # Profile the memory usage of the model and get the maximum number of
        # cache blocks that can be allocated with the remaining free memory.
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        print(f"{block_size=}, {gpu_memory_utilization=}, {cpu_swap_space=}")

        # Profile memory usage with max_num_sequences sequences and the total
        # number of tokens equal to max_num_batched_tokens.

        # Enable top-k sampling to reflect the accurate memory usage.
        vocab_size = self.model.config.vocab_size
        sampling_params = SamplingParams(top_p=0.99, top_k=vocab_size - 1)
        max_num_batched_tokens = self.scheduler_config.max_num_batched_tokens
        max_num_seqs = self.scheduler_config.max_num_seqs
        seqs = []
        for group_id in range(max_num_seqs):
            seq_len = (max_num_batched_tokens // max_num_seqs +
                       (group_id < max_num_batched_tokens % max_num_seqs))
            seq_data = SequenceData([0] * seq_len)
            seq = SequenceGroupMetadata(
                request_id=str(group_id),
                is_prompt=True,
                seq_data={group_id: seq_data},
                sampling_params=sampling_params,
                block_tables=None,
            )
            seqs.append(seq)

        input_tokens, input_positions, input_metadata = self._prepare_inputs(
            seqs)

        # Execute the model.
        num_layers = self.model_config.get_num_layers(self.parallel_config)
        # print(f"input_ids={input_tokens}, positions={input_positions}, kv_caches={[(None, None)] * num_layers}, input_metadata={input_metadata}, cache_events={None},")
        self.model(
            input_ids=input_tokens,
            positions=input_positions,
            kv_caches=[(None, None)] * num_layers,
            input_metadata=input_metadata,
            cache_events=None,
        )

        # Calculate the number of blocks that can be allocated with the
        # profiled peak memory.
        torch.cuda.synchronize()
        peak_memory = torch.cuda.max_memory_allocated()
        total_gpu_memory = get_gpu_memory()
        cache_block_size = CacheEngine.get_cache_block_size(
            block_size, self.model_config, self.parallel_config)
        num_gpu_blocks = int(
            (total_gpu_memory * gpu_memory_utilization - peak_memory) //
            cache_block_size)
        num_cpu_blocks = int(cpu_swap_space // cache_block_size)
        num_gpu_blocks = max(num_gpu_blocks, 0)
        num_cpu_blocks = max(num_cpu_blocks, 0)
        torch.cuda.empty_cache()

        # Reset the seed to ensure that the random state is not affected by
        # the model initialization and profiling.
        set_random_seed(self.model_config.seed)
        return num_gpu_blocks, num_cpu_blocks

    def init_cache_engine(self, cache_config: CacheConfig) -> None:
        self.cache_config = cache_config
        self.block_size = cache_config.block_size
        self.cache_engine = CacheEngine(self.cache_config, self.model_config,
                                        self.parallel_config)
        self.cache_events = self.cache_engine.events
        self.gpu_cache = self.cache_engine.gpu_cache

    def _prepare_inputs(
        self,
        seq_group_metadata_list: List[SequenceGroupMetadata],
    ) -> Tuple[torch.Tensor, torch.Tensor, InputMetadata]:
        seq_groups: List[Tuple[List[int], SamplingParams]] = []
        input_tokens: List[int] = []
        input_positions: List[int] = []
        slot_mapping: List[int] = []

        # Add prompt tokens.
        prompt_lens: List[int] = []
        for seq_group_metadata in seq_group_metadata_list:
            if not seq_group_metadata.is_prompt:
                continue

            seq_ids = list(seq_group_metadata.seq_data.keys())
            sampling_params = seq_group_metadata.sampling_params
            seq_groups.append((seq_ids, sampling_params))

            # Use any sequence in the group.
            seq_id = seq_ids[0]

            seq_data = seq_group_metadata.seq_data[seq_id]
            prompt_tokens = seq_data.get_token_ids()
            prompt_len = len(prompt_tokens)
            prompt_lens.append(prompt_len)

            input_tokens.extend(prompt_tokens)
            # NOTE(woosuk): Here we assume that the first token in the prompt
            # is always the first token in the sequence.
            input_positions.extend(range(len(prompt_tokens)))

            if seq_group_metadata.block_tables is None:
                # During memory profiling, the block tables are not initialized
                # yet. In this case, we just use a dummy slot mapping.
                slot_mapping.extend([0] * prompt_len)
                continue

            # Compute the slot mapping.
            block_table = seq_group_metadata.block_tables[seq_id]
            for i in range(prompt_len):
                block_number = block_table[i // self.block_size]
                block_offset = i % self.block_size
                slot = block_number * self.block_size + block_offset
                slot_mapping.append(slot)

        # Add generation tokens.
        max_context_len = 0
        max_num_blocks_per_seq = 0
        context_lens: List[int] = []
        generation_block_tables: List[List[int]] = []
        for seq_group_metadata in seq_group_metadata_list:
            if seq_group_metadata.is_prompt:
                continue

            seq_ids = list(seq_group_metadata.seq_data.keys())
            sampling_params = seq_group_metadata.sampling_params
            seq_groups.append((seq_ids, sampling_params))

            for seq_id in seq_ids:
                seq_data = seq_group_metadata.seq_data[seq_id]
                generation_token = seq_data.get_last_token_id()
                input_tokens.append(generation_token)

                context_len = seq_data.get_len()
                position = context_len - 1
                input_positions.append(position)

                block_table = seq_group_metadata.block_tables[seq_id]
                generation_block_tables.append(block_table)

                max_context_len = max(max_context_len, context_len)
                max_num_blocks_per_seq = max(max_num_blocks_per_seq,
                                             len(block_table))
                context_lens.append(context_len)

                block_number = block_table[position // self.block_size]
                block_offset = position % self.block_size
                slot = block_number * self.block_size + block_offset
                slot_mapping.append(slot)

        # Optimization: Pad the input length to be a multiple of 8.
        # This is required for utilizing the Tensor Cores in NVIDIA GPUs.
        input_tokens = _pad_to_alignment(input_tokens, multiple_of=8)
        input_positions = _pad_to_alignment(input_positions, multiple_of=8)

        # Convert to tensors.
        tokens_tensor = torch.cuda.LongTensor(input_tokens)
        positions_tensor = torch.cuda.LongTensor(input_positions)
        slot_mapping_tensor = torch.cuda.IntTensor(slot_mapping)
        context_lens_tensor = torch.cuda.IntTensor(context_lens)
        padded_block_tables = [
            _pad_to_max(block_table, max_num_blocks_per_seq)
            for block_table in generation_block_tables
        ]
        block_tables_tensor = torch.cuda.IntTensor(padded_block_tables)

        seq_data: Dict[int, SequenceData] = {}
        for seq_group_metadata in seq_group_metadata_list:
            seq_data.update(seq_group_metadata.seq_data)

        input_metadata = InputMetadata(
            seq_groups=seq_groups,
            seq_data=seq_data,
            prompt_lens=prompt_lens,
            slot_mapping=slot_mapping_tensor,
            context_lens=context_lens_tensor,
            max_context_len=max_context_len,
            block_tables=block_tables_tensor,
        )
        return tokens_tensor, positions_tensor, input_metadata

    def _prepare_inputs_draft(
        self,
        seq_group_metadata_list: List[SequenceGroupMetadata],
    ) -> Tuple[torch.Tensor, torch.Tensor, InputMetadata]:
        seq_groups: List[Tuple[List[int], SamplingParams]] = []
        input_tokens: List[int] = []
        input_positions: List[int] = []
        slot_mapping: List[int] = []

        prompt_lens: List[int] = []
        context_lens: List[int] = []
        max_context_len = 0
        for seq_group_metadata in seq_group_metadata_list:

            seq_ids = list(seq_group_metadata.seq_data.keys())
            sampling_params = seq_group_metadata.sampling_params
            seq_groups.append((seq_ids, sampling_params))

            # Use any sequence in the group.
            seq_id = seq_ids[0]

            sequence_data = seq_group_metadata.seq_data[seq_id]
            context_len = sequence_data.get_len(include_draft=True)
            prompt_lens.append(sequence_data.get_prompt_len())
            context_lens.append(context_len)
            context_tokens = sequence_data.get_token_ids(include_draft=True)

            input_tokens.extend(context_tokens)
            # NOTE(woosuk): Here we assume that the first token in the prompt
            # is always the first token in the sequence.
            input_positions.extend(range(len(context_tokens)))

            # TODO Will: the slot mapping is only neccesary for KV caching
            # we don't need it right now since we are using uncached inferences
            # enable when we combine KV cache with SSP
            # Will require scheduler to allocate an extra block if the draft crosses a block boundary in `Scheduler._append_slot`
            # Sequence.append_token_id also assumes that last block is the correct block to append to, which isn't the case if we preallocate blocks to accomodate draft length
            # Compute the slot mapping.
            # block_table = seq_group_metadata.block_tables[seq_id]
            # for i in range(context_len):
            #     block_number = block_table[i // self.block_size]
            #     block_offset = i % self.block_size
            #     slot = block_number * self.block_size + block_offset
            #     slot_mapping.append(slot)
            max_context_len = max(max_context_len, context_len)


        # Optimization: Pad the input length to be a multiple of 8.
        # This is required for utilizing the Tensor Cores in NVIDIA GPUs.
        input_tokens = _pad_to_alignment(input_tokens, multiple_of=8)
        input_positions = _pad_to_alignment(input_positions, multiple_of=8)

        # Convert to tensors.
        tokens_tensor = torch.cuda.LongTensor(input_tokens)
        positions_tensor = torch.cuda.LongTensor(input_positions)
        slot_mapping_tensor = torch.cuda.IntTensor(slot_mapping)
        context_lens_tensor = torch.cuda.IntTensor(context_lens)
        block_tables_tensor = torch.cuda.IntTensor([])  # TODO enable this when we leverage KV cache

        seq_data: Dict[int, SequenceData] = {}
        for seq_group_metadata in seq_group_metadata_list:
            seq_data.update(seq_group_metadata.seq_data)

        input_metadata = InputMetadata(
            seq_groups=seq_groups,
            seq_data=seq_data,
            prompt_lens=prompt_lens,
            slot_mapping=slot_mapping_tensor,
            context_lens=context_lens_tensor,
            max_context_len=max_context_len,
            block_tables=block_tables_tensor,  # Will: only relevent for single query generation
            draft_length=4, # TODO Will: parameterize?
        )
        return tokens_tensor, positions_tensor, input_metadata

    @torch.inference_mode()
    def execute_model(
        self,
        seq_group_metadata_list: List[SequenceGroupMetadata],
        blocks_to_swap_in: Dict[int, int],
        blocks_to_swap_out: Dict[int, int],
        blocks_to_copy: Dict[int, List[int]],
    ) -> SamplerOutput:
        # Issue cache operations.
        issued_cache_op = False
        if blocks_to_swap_in:
            self.cache_engine.swap_in(blocks_to_swap_in)
            issued_cache_op = True
        if blocks_to_swap_out:
            self.cache_engine.swap_out(blocks_to_swap_out)
            issued_cache_op = True
        if blocks_to_copy:
            self.cache_engine.copy(blocks_to_copy)
            issued_cache_op = True

        if issued_cache_op:
            cache_events = self.cache_events
        else:
            cache_events = None

        # If there is no input, we don't need to execute the model.
        if not seq_group_metadata_list:
            if cache_events is not None:
                for event in cache_events:
                    event.wait()
            return {}

        # Prepare input tensors.
        input_tokens, input_positions, input_metadata = self._prepare_inputs(
            seq_group_metadata_list)

        # Execute the model.
        # print(f"input_ids={input_tokens}, positions={input_positions}, input_metadata={input_metadata}, cache_events={None},")
        # 
        output = self.model(
            input_ids=input_tokens,
            positions=input_positions,
            kv_caches=self.gpu_cache,
            input_metadata=input_metadata,
            cache_events=cache_events,
        )
        print(f"execute_model {input_tokens.shape=}, {input_positions.shape=}")
        print(f"execute_model {output=}")
        return output

    @torch.inference_mode()
    def execute_draft_model(
        self,
        seq_group_metadata_list: List[SequenceGroupMetadata],
    ) -> DraftOutput:

        # print(f"{seq_group_metadata_list=}")
        # seq_data = seq_group_metadata_list[0].seq_data[0]
        # print(f"{seq_data=}")
        # input_tokens = seq_data.prompt_token_ids + seq_data.output_token_ids
        # tokens_tensor = torch.cuda.LongTensor(input_tokens).unsqueeze(0)
        # print(f"{tokens_tensor.shape=}")

        # find max len
        max_input_len = 0
        for seq_group_metadata in seq_group_metadata_list:
            # assuming only single seq per request
            seq_data = list(seq_group_metadata.seq_data.values())[0]
            max_input_len = max(max_input_len, len(seq_data.prompt_token_ids + seq_data.output_token_ids))

        # build input tensor with max padding
        batch_input_tokens = []
        batch_idx_to_seq_id = {}
        for i, seq_group_metadata in enumerate(seq_group_metadata_list):
            # assuming only single seq per request
            seq_id, seq_data = list(seq_group_metadata.seq_data.items())[0]
            tokens = _pad_to_max(seq_data.prompt_token_ids + seq_data.output_token_ids, max_input_len)
            batch_input_tokens.append(tokens)
            batch_idx_to_seq_id[i] = seq_id
        
        tokens_tensor = torch.cuda.LongTensor(batch_input_tokens)
        print('TESTING tokens_tensor')
        print(f"tokens_tensor {tokens_tensor.shape=}")
        # Execute the model.
        # print(f"input_ids={input_tokens}, positions={input_positions}, input_metadata={input_metadata}, cache_events={None},")
        
        seqs, probs = sample_from_draft_model(self.model, tokens_tensor, DRAFT_LENGTH)
        print(f"seqs {seqs.shape=}, probs {probs.shape=}")

        output = []
        for i, seq in enumerate(seqs):
            seq_id = batch_idx_to_seq_id[i]
            output.append([DraftOutputs(seq_id, seq, probs[i])])

        return output
    
    @torch.inference_mode()
    def execute_draft_scoring(
        self,
        seq_group_metadata_list: List[SequenceGroupMetadata],
        draft_output: DraftOutput,
    ) -> SamplerOutput:

        # seq_group_metadata_list - add to prompt
        # print(f"{seq_group_metadata_list=}")
        # seq_data = seq_group_metadata_list[0].seq_data[0]
        # print(f"{seq_data=}")
        # input_tokens = seq_data.prompt_token_ids + seq_data.output_token_ids
        # tokens_tensor = torch.cuda.LongTensor(input_tokens).unsqueeze(0)
        # print(f"{tokens_tensor.shape=}")

        test_draft_output: DraftOutput = [
            [DraftOutputs(0, [1,1,1,1], [0.7,0.7,0.7,0.7])],
            [DraftOutputs(1, [2,2,2,2], [0.8,0.8,0.8,0.8])],
        ]

        for seq_group_metadata in seq_group_metadata_list:
            for seq_id, seq_data in seq_group_metadata.seq_data.items():
                # find the draft output for this seq_id and unpack nested list structure
                draft_outputs = next(draft_output for draft_output in test_draft_output if draft_output[0].parent_seq_id == seq_id)[0]
                # We need to undo the append here at some point, if the drafts are not accepted we need to clean this up
                # probably better to just add a new field at this point
                seq_data.draft_token_ids = draft_outputs.output_tokens

        # Prepare input tensors.
        input_tokens, input_positions, input_metadata = self._prepare_inputs_draft(
            seq_group_metadata_list)

        # Execute the model.
        # print(f"input_ids={input_tokens}, positions={input_positions}, input_metadata={input_metadata}, cache_events={None},")
        # do rejection sampling in here, since sampling logic is here already
        # we want to return type DraftOutput = List[List[DraftOutputs]] 
        output = self.model.forward_draft(
            input_ids=input_tokens,
            positions=input_positions,
            kv_caches=self.gpu_cache,
            input_metadata=input_metadata,
            draft_output=test_draft_output,
            cache_events=None,
        )
        print(f"execute_model {input_tokens.shape=}, {input_positions.shape=}")
        print(f"execute_model {output=}")
        # 
        return output
        # target model call
        # rejection sampling
        # final target sampling

def get_distribution(logits, temperature):
    probs = torch.softmax(logits / (temperature + 1e-10), dim=-1)
    return probs

def sample(logits, temperature):
    probs = get_distribution(logits, temperature)
    return torch.multinomial(probs, num_samples=1)

# def sample_from_draft_model(model, initial_prompt_seq, new_tokens, temperature=1.0):
#     fin_prompt_seq = initial_prompt_seq.detach().clone()
#     out_logits = []

#     for _ in range(new_tokens):
#         sample_token_logits = model(fin_prompt_seq).logits[:, -1, :]
#         sample_token = sample(sample_token_logits, temperature=temperature)
#         fin_prompt_seq = torch.concat([fin_prompt_seq, sample_token[None,...]], dim=-1)
#         out_logits.append(sample_token_logits)

#     out_logits = torch.stack(out_logits, dim=1)
#     return fin_prompt_seq, out_logits

def sample_from_draft_model(model, initial_prompt_seq, new_tokens, temperature=1.0):
    """ Returns:
    fin_prompt_seq - (batch_size, draft_token_len) token ids with same order as batch
    out_probs - (batch_size, draft_token_len) probs of token ids in fin_prompt_seq
    """
    # (batch_size, token_len)
    fin_prompt_seq = initial_prompt_seq.detach().clone()
    out_probs = []

    for _ in range(new_tokens):
        # (batch_size, vocab_size: e.g 32000)
        sample_logits = model(fin_prompt_seq).logits[:, -1, :]
        print(f"TESTING {sample_logits.shape=}")
        
        # Compute the probabilities from logits
        # (batch_size, vocab_size: e.g 32000)
        sample_probs = get_distribution(sample_logits, temperature)
        print(f"TESTING {sample_probs.shape=}")
        
        # (batch_size, token_index: always len 1)
        sample_tokens = sample(sample_logits, temperature=temperature)
        print(f"TESTING {sample_tokens.shape=}")
        
        # Extract the probability corresponding to the sampled token
        # (batch_size, prob: always len 1)
        selected_probs = sample_probs.gather(1, sample_tokens)
        print(f"TESTING {selected_probs.shape=}")
        out_probs.append(selected_probs)
        
        # Append sampled token to fin_prompt_seq
        fin_prompt_seq = torch.cat([fin_prompt_seq, sample_tokens], dim=-1)

    # TODO: double check that the correct tokens are added to each sequence
    out_probs = torch.stack(out_probs, dim=1).squeeze(-1)
    return fin_prompt_seq, out_probs

def _init_distributed_environment(
    parallel_config: ParallelConfig,
    rank: int,
    draft: bool = False,
    world_size_override: Optional[int] = None,
    distributed_init_method: Optional[str] = None,
) -> None:
    """Initialize the distributed environment."""
    print(f"testing cuda: {torch.cuda.is_available()}")
    world_size = world_size_override or parallel_config.world_size
    print(f"{world_size=}, {world_size_override=}, {parallel_config.world_size=}")
    if torch.distributed.is_initialized():
        torch_world_size = torch.distributed.get_world_size()
        print(f"{draft=}, {torch_world_size=}")
        if torch_world_size != world_size:
            raise RuntimeError(
                "torch.distributed is already initialized but the torch world "
                "size does not match parallel_config.world_size "
                f"({torch_world_size=} vs. {world_size=}) Debug ({draft=}, {world_size_override=}, {parallel_config.world_size=}).")
    elif not distributed_init_method:
        raise ValueError(
            "distributed_init_method must be set if torch.distributed "
            "is not already initialized")
    else:
        print('torch.distributed.init_process_group')
        torch.distributed.init_process_group(
            backend="nccl",
            world_size=world_size,
            rank=rank,
            init_method=distributed_init_method,
        )

    # A small all_reduce for warmup.
    torch.distributed.all_reduce(torch.zeros(1).cuda())
    initialize_model_parallel((1 if draft else parallel_config.tensor_parallel_size),
                              (1 if draft else parallel_config.pipeline_parallel_size))



def _pad_to_alignment(x: List[int], multiple_of: int) -> List[int]:
    return x + [0] * ((-len(x)) % multiple_of)


def _pad_to_max(x: List[int], max_len: int) -> List[int]:
    return x + [0] * (max_len - len(x))
