import copy
import logging
import os
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import jinja2
import torch
import torch.nn.functional as F
import transformers
from accelerate import (
    Accelerator,
    InitProcessGroupKwargs,
    find_executable_batch_size,
)
from accelerate.utils import get_max_memory
from huggingface_hub import HfApi
from packaging import version
from peft import PeftModel
from peft import __version__ as PEFT_VERSION
from tqdm import tqdm
from transformers.models.auto.modeling_auto import (
    MODEL_FOR_CAUSAL_LM_MAPPING_NAMES,
    MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING_NAMES,
)

from lm_eval import utils
from lm_eval.api.instance import Instance
from lm_eval.api.model import TemplateLM
from lm_eval.api.registry import register_model
from lm_eval.models.utils import (
    Collator,
    clear_torch_cache,
    configure_pad_token,
    get_dtype,
    handle_stop_sequences,
    pad_and_concat,
    stop_sequences_criteria,
)


eval_logger = logging.getLogger(__name__)


@register_model("hf-auto", "hf", "huggingface")
class HFLM(TemplateLM):
    """
    An abstracted Huggingface model class. Enables usage with both models of
    `transformers.AutoModelForCausalLM` and `transformers.AutoModelForSeq2SeqLM` classes.

    Supports data-parallel multi-GPU with HF Accelerate.
    """

    AUTO_MODEL_CLASS = None
    _DEFAULT_MAX_LENGTH = 2048

    def __init__(
        self,
        pretrained: Union[str, transformers.PreTrainedModel],
        backend: Literal["default", "causal", "seq2seq"] = "default",
        # override whether the model should be treated as decoder-only (causal) or encoder-decoder (seq2seq)
        revision: Optional[str] = "main",
        subfolder: Optional[str] = None,
        tokenizer: Optional[
            Union[
                str,
                transformers.PreTrainedTokenizer,
                transformers.PreTrainedTokenizerFast,
            ]
        ] = None,
        truncation: Optional[bool] = False,
        logits_cache: bool = True,
        max_length: Optional[int] = None,
        add_special_tokens: Optional[bool] = None,
        use_accelerate: Optional[bool] = False,
        low_cpu_mem_usage: Optional[bool] = True,
        device_map_option: Optional[str] = "auto",
        max_memory_per_gpu: Optional[Union[int, str]] = None,
        max_cpu_memory: Optional[Union[int, str]] = None,
        offload_folder: Optional[str] = "./offload",
        dtype: Optional[Union[str, torch.dtype]] = None,
        device: Optional[Union[int, str]] = "cuda",
        peft: str = None,
        load_in_8bit: Optional[bool] = False,
        load_in_4bit: Optional[bool] = False,
        trust_remote_code: Optional[bool] = False,
        gptq_use_triton: Optional[bool] = False,
        inject_fused_attention: Optional[bool] = True,
        bnb_4bit_quant_type: Optional[str] = None,
        bnb_4bit_compute_dtype: Optional[Union[str, torch.dtype]] = None,
        bnb_4bit_use_double_quant: Optional[bool] = False,
    ):
        """Initializes a HuggingFace `AutoModel` and `AutoTokenizer` for evaluation.
        Args:
            pretrained (str):
                The HuggingFace Hub model ID name or the path to a pre-trained
                model to load. This is effectively the `pretrained_model_name_or_path`
                argument of `from_pretrained` in the HuggingFace `transformers` API.
            quantized (str or bool, optional, defaults to False):
                File name of a GPTQ quantized model to load. Set to `True` to use the
                default name of the quantized model.
            add_special_tokens (bool, optional, defaults to True):
                Whether to add special tokens to the input sequences. If `None`, the
                default value will be set to `True` for seq2seq models (e.g. T5) and
                `False` for causal models.
                WARNING: Evaluating causal models with `add_special_tokens=True` is
                currently __not__ supported.
            > Large model loading `accelerate` arguments
            use_accelerate (bool, optional, defaults to False):
                If True, uses the `accelerate` library to load a large model across
                multiple devices.
            low_cpu_mem_usage (bool, optional, defaults to True):
                It True, uses the `accelerate` library to accelerate loading the model.
            device_map_option (str, optional, defaults to "auto"):
                The device map option to use when loading the model with
                `accelerate`.
                Options:
                    "auto", "balanced", "balanced_low_0", "sequential"
                See the `accelerate` docs for more details on these options:
                https://huggingface.co/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel.from_pretrained.device_map
            max_memory_per_gpu (Union[int, str], optional, defaults to None):
                The maximum memory available for each GPU in bytes as `int` or in
                the format f"{significand}{unit_symbol}" where {unit_symbol} is
                any of ["GB", "MB", "GIB", "MIB"]. Refer to the `max_memory` arg in
                the "Parameters for big model inference" section of the following
                docs:
                https://huggingface.co/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel.from_pretrained.max_memory
            max_cpu_memory (Union[int, str], optional, defaults to None):
                The maximum available CPU RAM in bytes as `int` or in the format
                f"{significand}{unit_symbol}" where {unit_symbol} is any of
                ["GB", "MB", "GIB", "MIB"]. Refer to the `max_memory` arg in the
                "Parameters for big model inference" section of the following docs:
                https://huggingface.co/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel.from_pretrained.max_memory
            offload_folder (str, optional, defaults to "./offload"):
                The folder to offload weights into if `device_map` contains any
                "disk" value.
            dtype (Union[str, torch.dtype], optional, defaults to None):):
                Converts the model weights to `dtype`, if specified. Strings get
                converted to `torch.dtype` objects (e.g. `float16` -> `torch.float16`).
                Use `dtype="auto"` to derive the type from the model’s weights.
            peft (str, optional, defaults to None):
                Path of the adapter weights to load from Huggingface. This will usually
                include a directory that includes the files `adapter_config.json` and
                `adapter_model.bin`. Compatible with [PEFT](https://github.com/huggingface/peft)
            load_in_8bit (bool, optional, defaults to False):
                If True, will convert the loaded model into mixed-8bit quantized model. See:
                https://huggingface.co/docs/transformers/main/en/main_classes/quantization#load-a-large-model-in-8bit
            load_in_4bit (bool, optional, defaults to False):
                If True, will convert the loaded model into mixed-4bit quantized model. See:
                https://huggingface.co/docs/transformers/main/en/main_classes/quantization#load-a-large-model-in-4bit
            trust_remote_code (bool, optional, defaults to False):
                If True, will trust the remote code when loading the model.
            gptq_use_triton (bool, optional, defaults to False):
                Use Triton for GPTQ inference.
            inject_fused_attention (bool, optional, defaults to True):
                Inject fused attention into GPTQ model.
            bnb_4bit_quant_type (str, optional, defaults to None):
                The quantization type to use for BnB 4bit quantization. See:
                https://github.com/huggingface/transformers/blob/main/src/transformers/utils/quantization_config.py#L77
            bnb_4bit_compute_dtype (Union[str, torch.dtype], optional, defaults to None):
                The compute dtype to use for BnB 4bit quantization. See:
                https://github.com/huggingface/transformers/blob/main/src/transformers/utils/quantization_config.py#L74
            bnb_4bit_use_double_quant (bool, optional, defaults to False):
                Whether or not to use double quant to quantize the absmax.
                https://github.com/huggingface/transformers/blob/main/src/transformers/utils/quantization_config.py#L80

        """
        super().__init__()

        assert isinstance(pretrained, str)
        assert isinstance(device, str)
        assert isinstance(batch_size, (int, str))
        if (
            add_special_tokens is not None
            and self.AUTO_MODEL_CLASS is transformers.AutoModelForCausalLM
        ):
            # TODO: Support evaluating causal models with special tokens. Currently,
            # this is not possible because the `_loglikelihood_tokens()` method for
            # causal LMs makes a no-special-tokens assumption given that contexts
            # and labels/continuations are tokenized separately without special
            # tokens, concatenated, and then processed as inputs.
            assert (
                not add_special_tokens
            ), "Evaluating causal models with `add_special_tokens=True` is currently not supported."

        # setup for automatic batch size detection
        if str(batch_size).startswith("auto"):
            batch_size = batch_size.split(":")
            self._batch_size = batch_size[0]
            self.batch_schedule = float(batch_size[1]) if len(batch_size) > 1 else 1
        else:
            self._batch_size = int(batch_size)
        self.max_batch_size = max_batch_size

        self._max_gen_toks = max_gen_toks
        self._max_length = max_length
        self._config = self.AUTO_CONFIG_CLASS.from_pretrained(
            pretrained,
            trust_remote_code=trust_remote_code,
            revision=revision + ("/" + subfolder if subfolder is not None else ""),
        )

        self._add_special_tokens = add_special_tokens
        self.tokenizer = self._create_auto_tokenizer(
            pretrained=pretrained,
            revision=revision,
            subfolder=subfolder,
            tokenizer=tokenizer,
            trust_remote_code=trust_remote_code,
        )
        self.tokenizer.model_max_length = self.max_length

        model_kwargs = {}
        if use_accelerate:
            model_kwargs = _get_accelerate_args(
                device_map_option,
                max_memory_per_gpu,
                max_cpu_memory,
                offload_folder,
            )
        self.model = self._create_auto_model(
            pretrained=pretrained,
            quantized=quantized,
            trust_remote_code=trust_remote_code,
            revision=revision,
            subfolder=subfolder,
            torch_dtype=_get_dtype(dtype, self._config),
            gptq_use_triton=gptq_use_triton,
            inject_fused_attention=inject_fused_attention,
            load_in_8bit=load_in_8bit,
            load_in_4bit=load_in_4bit,
            bnb_4bit_quant_type=bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=bnb_4bit_compute_dtype,
            bnb_4bit_use_double_quant=bnb_4bit_use_double_quant,
            low_cpu_mem_usage=low_cpu_mem_usage,
            **model_kwargs,
        )
        # note: peft_path can be different than pretrained model path
        if peft is not None:
            self.model = self._create_auto_model_peft(
                model=self.model,
                peft=peft,
                delta=delta,
                autogptq=autogptq,
                gptqmodel=gptqmodel,
                gguf_file=gguf_file,
                quantization_config=getattr(self.config, "quantization_config", None),
                **kwargs,
            )
        self.model.eval()
        torch.set_grad_enabled(False)

        self._device = device
        if use_accelerate and "lm_head" in self.model.hf_device_map:
            # `accelerate` can place `lm_head` weights on a different device than
            # the user specified one so we force `self._device` to be the same as
            # `lm_head`'s.
            self._device = self.model.hf_device_map["lm_head"]
        if not use_accelerate and not (load_in_4bit or load_in_8bit):
            try:
                self.model.to(self._device)
            except:
                print(
                    "Failed to place model onto specified device. This may be because the model is quantized via `bitsandbytes`. If the desired GPU is being used, this message is safe to ignore."
                )

    def _create_auto_model(
        self,
        *,
        pretrained: str,
        quantized: Optional[Union[bool, str]] = False,
        revision: str,
        subfolder: str,
        low_cpu_mem_usage: Optional[bool] = True,
        device_map: Optional[Union[str, _DeviceMapping]] = None,
        max_memory: Optional[dict] = None,
        offload_folder: Optional[str] = None,
        load_in_8bit: Optional[bool] = False,
        load_in_4bit: Optional[bool] = False,
        trust_remote_code: Optional[bool] = False,
        torch_dtype: Optional[Union[str, torch.dtype]] = None,
        gptq_use_triton: Optional[bool] = False,
        inject_fused_attention: Optional[bool] = True,
        bnb_4bit_quant_type: Optional[str] = None,
        bnb_4bit_compute_dtype: Optional[Union[str, torch.dtype]] = None,
        bnb_4bit_use_double_quant: Optional[bool] = False,
    ) -> transformers.AutoModel:
        """Returns a pre-trained pytorch model from a pre-trained model configuration."""
        if not quantized:
            if load_in_4bit:
                assert (
                    transformers.__version__ >= "4.30.0"
                ), "load_in_4bit requires transformers >= 4.30.0"
            model_kwargs = {}
            if transformers.__version__ >= "4.30.0":
                model_kwargs["load_in_4bit"] = load_in_4bit
                if load_in_4bit:
                    if bnb_4bit_quant_type:
                        model_kwargs["bnb_4bit_quant_type"] = bnb_4bit_quant_type
                    if bnb_4bit_compute_dtype:
                        model_kwargs["bnb_4bit_compute_dtype"] = _get_dtype(
                            bnb_4bit_compute_dtype
                        )
                    if bnb_4bit_use_double_quant:
                        model_kwargs[
                            "bnb_4bit_use_double_quant"
                        ] = bnb_4bit_use_double_quant
            model = self.AUTO_MODEL_CLASS.from_pretrained(
                pretrained,
                revision=revision + ("/" + subfolder if subfolder is not None else ""),
                low_cpu_mem_usage=low_cpu_mem_usage,
                device_map=device_map,
                max_memory=max_memory,
                offload_folder=offload_folder,
                load_in_8bit=load_in_8bit,
                trust_remote_code=trust_remote_code,
                torch_dtype=torch_dtype,
                **model_kwargs,
            )
        else:
            from auto_gptq import AutoGPTQForCausalLM

            model = AutoGPTQForCausalLM.from_quantized(
                pretrained,
                model_basename=None if quantized == True else Path(quantized).stem,
                device_map=device_map,
                max_memory=max_memory,
                trust_remote_code=trust_remote_code,
                use_safetensors=True
                if quantized == True
                else quantized.endswith(".safetensors"),
                use_triton=gptq_use_triton,
                warmup_triton=gptq_use_triton,
                inject_fused_attention=inject_fused_attention,
            )
        return model

    def _create_auto_model_peft(
        self,
        *,
        model: transformers.PreTrainedModel,
        peft: str,
        revision: str,
        subfolder: str,
        load_in_4bit: Optional[bool] = False,
    ):
        if load_in_4bit:
            assert PEFT_VERSION >= "0.4.0", "load_in_4bit requires peft >= 0.4.0"
        model = self.AUTO_PEFT_CLASS.from_pretrained(
            model,
            peft,
            revision=revision + ("/" + subfolder if subfolder is not None else ""),
        )
        return model

    def _create_auto_tokenizer(
        self,
        *,
        pretrained: str,
        revision: str,
        subfolder: str,
        tokenizer: Optional[str] = None,
        trust_remote_code: Optional[bool] = False,
    ) -> transformers.PreTrainedTokenizer:
        """Returns a pre-trained tokenizer from a pre-trained tokenizer configuration."""
        tokenizer = self.AUTO_TOKENIZER_CLASS.from_pretrained(
            pretrained if tokenizer is None else tokenizer,
            revision=revision + ("/" + subfolder if subfolder is not None else ""),
            trust_remote_code=trust_remote_code,
        )
        tokenizer.pad_token = tokenizer.eos_token
        return tokenizer

    @property
    def add_special_tokens(self) -> bool:
        """Whether to include special tokens in encoded text. This should be
        determined by whether or not the model was trained with special tokens.
        TODO: Remove these conditionals once HuggingFace supports a way to
        check whether or not an arbitrary model was trained with special tokens.
        """
        if self._add_special_tokens is not None:
            return self._add_special_tokens
        elif self.AUTO_MODEL_CLASS is transformers.AutoModelForCausalLM:
            return False
        elif self.AUTO_MODEL_CLASS is transformers.AutoModelForSeq2SeqLM:
            return True
        else:
            raise ValueError(
                "Could not determine `add_special_tokens` value from the model "
                "class. Set to `True` or `False` depending on whether the model "
                "was pre-trained with special tokens."
            )
        else:
            args["max_memory"] = None
            args["device_map"] = None
            eval_logger.info("Model parallel was set to False.")

        return args

    @property
    def config(self):
        # return the associated transformers.AutoConfig for the given pretrained model.
        return self._config

    @property
    def model(self):
        # returns the model, unwrapping it if using Accelerate
        if hasattr(self, "accelerator"):
            return self.accelerator.unwrap_model(self._model)
        else:
            return self._model

    @property
    def eot_token_id(self):
        # we use EOT because end of *text* is more accurate for what we're doing than end of *sentence*
        return self.tokenizer.eos_token_id

    @property
    def prefix_token_id(self):
        # it is used as prefix for loglikelihood
        if self.custom_prefix_token_id is not None:
            return self.custom_prefix_token_id
        if self.tokenizer.bos_token_id is not None:
            return self.tokenizer.bos_token_id
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        if self._max_length:  # if max length manually set, return it
            return self._max_length
        seqlen_config_attrs = ("n_positions", "max_position_embeddings", "n_ctx")
        for attr in seqlen_config_attrs:
            if hasattr(self.model.config, attr):
                return getattr(self.model.config, attr)
        if hasattr(self.tokenizer, "model_max_length"):
            if self.tokenizer.model_max_length == 1000000000000000019884624838656:
                return self._DEFAULT_MAX_LENGTH
            return self.tokenizer.model_max_length
        return self._DEFAULT_MAX_LENGTH

    @property
    def max_gen_toks(self) -> int:
        return 256

    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    @property
    def device(self):
        return self._device

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    @property
    def tokenizer_name(self) -> str:
        return self.tokenizer.name_or_path.replace("/", "__")

    def tok_encode_batch(self, strings: List[str]) -> TokenSequence:
        return self.tokenizer(
            strings,
            padding=True,
            add_special_tokens=self.add_special_tokens,
            return_tensors="pt",
        )

    def tok_decode(self, tokens: torch.LongTensor) -> List[str]:
        return self.tokenizer.batch_decode(tokens, skip_special_tokens=True)

    def greedy_until(
        self, requests: List[Tuple[str, Union[List[str], str]]]
    ) -> List[str]:
        def _collate(x):
            tokens = self.tok_encode(x[0])
            return len(tokens), x[0]

        results = []
        reorder = utils.Reorderer(requests, _collate)

        adaptive_batch_size = None
        if self.batch_size == "auto":
            # using rolling window with maximum context
            print("Passed argument batch_size = auto. Detecting largest batch size")
            batch_size = self._detect_batch_size()
            print(f"Determined Largest batch size: {batch_size}")
            adaptive_batch_size = batch_size

        for chunk in utils.chunks(
            tqdm(reorder.get_reordered(), disable=False),
            self.batch_size if self.batch_size != "auto" else adaptive_batch_size,
        ):
            context = [c[0] for c in chunk]
            request_args = chunk[0][1]
            stop = request_args.get("until", None)
            stop_sequences = stop if isinstance(stop, list) else [stop]
            max_generation_length = request_args.get("max_length", None)

            assert (
                isinstance(max_generation_length, int) or max_generation_length is None
            )
            assert isinstance(stop_sequences, list) or stop_sequences is None

            # TODO: Find a better way to handle stop sequences for 0-shot.
            if stop_sequences is None:
                until = [self.eot_token]
            else:
                until = stop_sequences + [self.eot_token]

            if max_generation_length is None:
                max_tokens = self.max_gen_toks
            else:
                max_tokens = max_generation_length

            token_context = self.tok_encode_batch(context)

            responses = self._model_generate(
                inputs=token_context,
                max_tokens=max_tokens,
                stop=until,
            )
            responses = self.tok_decode(responses.tolist())

            for response in responses:
                # Ensure the generated responses do not contain the stop sequences.
                for term in until:
                    response = response.split(term)[0]
                # partial caching
                self.cache_hook.add_partial("greedy_until", (context, until), response)
                results.append(response)
        return reorder.get_original(results)


class AutoCausalLM(HuggingFaceAutoLM):
    """Causal language modeling.
    You can find a set of supported models in the HF documentation:
    https://huggingface.co/docs/transformers/main/model_doc/auto#transformers.AutoModelForCausalLM
    """

    AUTO_MODEL_CLASS = transformers.AutoModelForCausalLM
    AUTO_PEFT_CLASS = peft.PeftModel

    def _create_auto_tokenizer(
        self,
        *,
        pretrained: str,
        revision: str,
        subfolder: str,
        tokenizer: Optional[str] = None,
        trust_remote_code: Optional[bool] = False,
    ) -> transformers.PreTrainedTokenizer:
        tokenizer = super()._create_auto_tokenizer(
            pretrained=pretrained,
            revision=revision,
            subfolder=subfolder,
            tokenizer=tokenizer,
            trust_remote_code=trust_remote_code,
        )
        tokenizer.padding_side = "left"
        return tokenizer

    def _model_call(
        self, inputs: TokenSequence, labels: Optional[TokenSequence] = None
    ) -> TokenSequence:
        return self.model(inputs)["logits"]

    def _model_generate(
        self,
        pretrained: Union[str, transformers.PreTrainedModel],
        tokenizer: Optional[
            Union[
                str,
                transformers.PreTrainedTokenizer,
                transformers.PreTrainedTokenizerFast,
            ]
        ],
        revision: Optional[str] = "main",
        trust_remote_code: Optional[bool] = False,
        use_fast_tokenizer: Optional[bool] = True,
        gguf_file: Optional[str] = None,
        add_bos_token: Optional[bool] = False,
    ) -> None:
        """
        Helper method during initialization.

        Create a tokenizer object corresponding to the correct
        tokenizer for value of `pretrained`, or use the pre-initialized tokenizer passed.
        """
        kwargs = {
            "revision": revision,
            "trust_remote_code": trust_remote_code,
        }

        # gguf format embeds tokenizer and is not compatible with hf tokenizer `use_fast` param
        if gguf_file is not None:
            kwargs["gguf_file"] = gguf_file
        else:
            kwargs["use_fast"] = use_fast_tokenizer

        if add_bos_token:
            kwargs["add_bos_token"] = True

        if tokenizer:
            if isinstance(tokenizer, str):
                self.tokenizer = transformers.AutoTokenizer.from_pretrained(
                    tokenizer, **kwargs
                )
            else:
                assert isinstance(
                    tokenizer, transformers.PreTrainedTokenizer
                ) or isinstance(tokenizer, transformers.PreTrainedTokenizerFast)
                self.tokenizer = tokenizer
        else:
            # Get tokenizer based on 'pretrained'
            if isinstance(pretrained, str):
                model_name = pretrained
            else:
                # get the HF hub name via accessor on model
                model_name = self.model.name_or_path
            self.tokenizer = transformers.AutoTokenizer.from_pretrained(
                model_name, **kwargs
            )
        return None

    def _detect_batch_size(self, requests=None, pos: int = 0):
        if requests:
            _, context_enc, continuation_enc = requests[pos]
            max_length = len(
                (context_enc + continuation_enc)[-(self.max_length + 1) :][:-1]
            )
            max_context_enc = len(context_enc[-(self.max_length + 1) :])
            max_cont_enc = len(continuation_enc[-(self.max_length + 1) :])
        else:
            max_length = self.max_length
            max_context_enc = max_length
            max_cont_enc = max_length

        # if OOM, then halves batch_size and tries again
        @find_executable_batch_size(starting_batch_size=self.max_batch_size)
        def forward_batch(batch_size):
            if self.backend == "seq2seq":
                length = max(max_context_enc, max_cont_enc)
                batched_conts = torch.ones(
                    (batch_size, length), device=self.device
                ).long()
                test_batch = torch.ones((batch_size, length), device=self.device).long()
                call_kwargs = {
                    "attn_mask": test_batch,
                    "labels": batched_conts,
                }
            else:
                call_kwargs = {}
                test_batch = torch.ones(
                    (batch_size, max_length), device=self.device
                ).long()
            for _ in range(5):
                out = F.log_softmax(  # noqa: F841
                    self._model_call(test_batch, **call_kwargs),
                    dim=-1,
                    dtype=self.softmax_dtype,
                )

            return batch_size

        try:
            batch_size = forward_batch()
        except RuntimeError as e:
            if "No executable batch size found" in str(e):
                batch_size = 1
            else:
                raise

        if self.world_size > 1:
            # if multi-GPU, always take minimum over all selected batch sizes
            max_rnk_bs = torch.tensor([batch_size], device=self.device)
            gathered = (
                self.accelerator.gather(max_rnk_bs).cpu().detach().numpy().tolist()
            )
            batch_size = min(gathered)
            clear_torch_cache()
            return batch_size

        clear_torch_cache()
        return batch_size

    def tok_encode(
        self, string: str, left_truncate_len=None, add_special_tokens=None
    ) -> List[int]:
        """ """
        # default for None - empty dict, use predefined tokenizer param
        # used for all models except for CausalLM or predefined value
        special_tokens_kwargs = {}

        # by default for CausalLM - false or self.add_bos_token is set
        if add_special_tokens is None:
            if self.backend == "causal":
                special_tokens_kwargs = {
                    "add_special_tokens": False or self.add_bos_token
                }
        # otherwise the method explicitly defines the value
        else:
            special_tokens_kwargs = {"add_special_tokens": add_special_tokens}

        encoding = self.tokenizer.encode(string, **special_tokens_kwargs)

        # left-truncate the encoded context to be at most `left_truncate_len` tokens long
        if left_truncate_len:
            encoding = encoding[-left_truncate_len:]

        return encoding

    def tok_batch_encode(
        self,
        strings: List[str],
        padding_side: str = "left",
        left_truncate_len: int = None,
        truncation: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # encode a batch of strings. converts to tensors and pads automatically, unlike tok_encode.
        old_padding_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = padding_side

        add_special_tokens = {}
        if self.backend == "causal":
            add_special_tokens = {"add_special_tokens": False or self.add_bos_token}

        encoding = self.tokenizer(
            strings,
            truncation=truncation,
            padding="longest",
            return_tensors="pt",
            **add_special_tokens,
        )
        if left_truncate_len:
            original_lengths = encoding["input_ids"].size(1)
            if original_lengths > left_truncate_len:
                eval_logger.warn(
                    f"Left truncation applied. Original sequence length was {original_lengths}, "
                    f"truncating to last {left_truncate_len} tokens. Some content will be lost.",
                )
            encoding["input_ids"] = encoding["input_ids"][:, -left_truncate_len:]
            encoding["attention_mask"] = encoding["attention_mask"][
                :, -left_truncate_len:
            ]
        self.tokenizer.padding_side = old_padding_side

        return encoding["input_ids"], encoding["attention_mask"]

    def tok_decode(self, tokens, skip_special_tokens=True):
        return self.tokenizer.decode(tokens, skip_special_tokens=skip_special_tokens)

    def _model_call(self, inps, attn_mask=None, labels=None):
        """
        :param inps: torch.Tensor
            A torch tensor of shape [batch, (sequence_ctx + sequence_cont)] or of shape
            [batch, sequence_ctx]. the size of sequence may vary from call to call
        :param attn_mask: torch.Tensor, optional
            A torch tensor of shape [batch, (sequence_ctx + sequence_cont)]. Only passed
            (and must be passed) if self.AUTO_MODEL_CLASS is transformers.AutoModelForSeq2SeqLM
        :param labels: torch.Tensor, optional
            A torch tensor of shape [batch, (sequence_ctx + sequence_cont)]. Only passed
            (and must be passed) if self.AUTO_MODEL_CLASS is transformers.AutoModelForSeq2SeqLM
        :return
            A torch tensor of shape [batch, sequence, vocab] with the
        logits returned from the model's decoder
        """
        with torch.no_grad():
            if attn_mask is not None or labels is not None:
                assert attn_mask is not None and labels is not None
                assert self.AUTO_MODEL_CLASS == transformers.AutoModelForSeq2SeqLM
                return self.model(
                    input_ids=inps, attention_mask=attn_mask, labels=labels
                ).logits
            else:
                assert self.AUTO_MODEL_CLASS in (
                    transformers.AutoModelForCausalLM,
                    transformers.AutoModelForVision2Seq,
                )
                return self.model(inps).logits

    def _model_generate(self, context, max_length, stop, **generation_kwargs):
        # temperature = 0.0 if not set
        # if do_sample is false and temp==0.0:
        # remove temperature, as do_sample=False takes care of this
        # and we don't want a warning from HF
        generation_kwargs["temperature"] = generation_kwargs.get("temperature", 0.0)
        do_sample = generation_kwargs.get("do_sample", None)

        # The temperature has to be a strictly positive float -- if it is 0.0, use greedy decoding strategies
        if generation_kwargs.get("temperature") == 0.0 and do_sample is None:
            generation_kwargs["do_sample"] = do_sample = False

        if do_sample is False and generation_kwargs.get("temperature") == 0.0:
            generation_kwargs.pop("temperature")
        # build stopping criteria
        stopping_criteria = stop_sequences_criteria(
            self.tokenizer, stop, context.shape[1], context.shape[0]
        )
        return self.model.generate(
            input_ids=context,
            max_length=max_length,
            stopping_criteria=stopping_criteria,
            pad_token_id=self.tokenizer.pad_token_id,
            use_cache=True,
            **generation_kwargs,
        )

    def _select_cont_toks(
        self, logits: torch.Tensor, contlen: int = None, inplen: int = None
    ) -> torch.Tensor:
        if self.backend == "causal":
            assert contlen and inplen, (
                "Must pass input len and cont. len to select scored logits for causal LM"
            )
            # discard right-padding.
            # also discard the input/context tokens. we'll only score continuations.
            logits = logits[inplen - contlen : inplen]
        elif self.backend == "seq2seq":
            assert contlen and not inplen, (
                "Selecting scored logits for Seq2SeqLM requires only cont. len"
            )
            # only discard right-padding.
            # the logits input to this fn only contain decoder-side tokens.
            logits = logits[:contlen]

        return logits

    def loglikelihood_rolling(
        self, requests: List[Instance], disable_tqdm: bool = False
    ) -> List[float]:
        adaptive_batch_size = None
        if self.batch_size == "auto":
            # using rolling window with maximum context
            print("Passed argument batch_size = auto. Detecting largest batch size")
            batch_size = self._detect_batch_size()
            print(f"Determined Largest batch size: {batch_size}")
            adaptive_batch_size = batch_size

        # First, collect all windows from all requests
        all_windows = []  # List of (request_idx, window) tuples
        request_window_counts = []  # Track number of windows per request

        for req_idx, (string,) in enumerate(
            tqdm(
                [req.args for req in requests],
                disable=(disable_tqdm or (self.rank != 0)),
            )
        ):
            rolling_token_windows: List[Tuple[List[int], List[int]]] = list(
                map(
                    utils.make_disjoint_window,
                    utils.get_rolling_token_windows(
                        token_list=self.tok_encode(string),
                        prefix_token=self.prefix_token_id,
                        max_seq_len=self.max_length,
                        context_len=1,
                    ),
                )
            )

            # TODO: Right now, we pass single EOT token to the Encoder and the full context to the decoder, in seq2seq case
            windows = [(None,) + x for x in rolling_token_windows]

            # Store windows with their request index
            all_windows.extend((req_idx, window) for window in windows)
            request_window_counts.append(len(windows))

        # Handle distributed case padding
        pad_amnt = 0
        if self.world_size > 1:
            mytensor = torch.tensor(len(all_windows), device=self.device)
            gathered = self.accelerator.gather(mytensor).cpu().detach().numpy().tolist()
            pad_amnt = max(gathered) - gathered[self.rank]
            if pad_amnt > 0:
                all_windows += pad_amnt * [all_windows[0]]

        all_nlls = []
        batch_size = adaptive_batch_size or self.batch_size
        for i in range(0, len(all_windows), batch_size):
            batch = all_windows[i : i + batch_size]
            # Extract just the windows for processing, keeping track of request indices
            batch_indices, batch_windows = zip(*batch)

            batch_nlls = self._loglikelihood_tokens(
                requests=batch_windows,
                disable_tqdm=False,
                override_bs=len(batch_windows),
            )
            # Store results with their request indices
            all_nlls.extend(zip(batch_indices, batch_nlls))

        # Remove padding if necessary
        if (self.world_size > 1) and (pad_amnt > 0):
            all_nlls = all_nlls[:-pad_amnt]

        # Reconstruct per-request loglikelihoods
        loglikelihoods = []
        current_idx = 0
        for window_count in request_window_counts:
            # Get all nlls for this request
            request_nlls = all_nlls[current_idx : current_idx + window_count]
            # Sum up the nlls for this request (discarding is_greedy)
            request_total = sum(nll[0] for _, nll in request_nlls)
            loglikelihoods.append(request_total)
            current_idx += window_count

            string = requests[len(loglikelihoods) - 1].args[0]
            self.cache_hook.add_partial(
                "loglikelihood_rolling", (string,), request_total
            )

        return loglikelihoods

    def _batch_scheduler(self, pos, n_reordered_requests):
        sched = pos // int(len(n_reordered_requests) / self.batch_schedule)
        if sched in self.batch_sizes:
            return self.batch_sizes[sched]
        if (len(self.batch_sizes) > 1) and (
            self.batch_sizes[sched - 1] == self.max_batch_size
        ):
            # if previous batch size is already maximal, skip recomputation
            self.batch_sizes[sched] = self.max_batch_size
            return self.batch_sizes[sched]
        print(
            f"Passed argument batch_size = auto:{self.batch_schedule}. Detecting largest batch size"
        )
        self.batch_sizes[sched] = self._detect_batch_size(n_reordered_requests, pos)
        print(f"Determined largest batch size: {self.batch_sizes[sched]}")
        return self.batch_sizes[sched]

    def _loglikelihood_tokens(
        self,
        requests: List[Tuple[Tuple[str, str], List[int], List[int]]],
        disable_tqdm: bool = False,
        override_bs: int = None,
    ) -> List[Tuple[float, bool]]:
        # TODO: implement some kind of efficient-request-middleware that lumps together requests with the same context
        res = []

        def _collate(req: Tuple[Tuple[str, str], List[int], List[int]]):
            """Defines the key for the sorted method"""
            # the negative sign on len(toks) sorts descending - this has a few advantages:
            # - time estimates will always be over not underestimates, which is more useful for planning
            # - to know the size of a batch when going through the list, you know the first one is always the batch
            #   padded context length. this is useful to simplify the batching logic and more importantly to make
            #   automatic adaptive batches much much easier to implement
            # - any OOMs will happen right away rather than near the end

            toks = req[1] + req[2]
            return -len(toks), tuple(toks)

        def _lookup_one_token_cont(req: Tuple[Tuple[str, str], List[int], List[int]]):
            """Defines the key to group and lookup one-token continuations"""
            # Use with group_by="contexts" (optional)"
            # allows for the creation of a lookup, so we can reuse logits in case of one-token continuations.
            # speeds up some multiple-choice tasks proportionally to the number of choices.
            # groups requests by context+continuation[:-1] and infer on one request/group.
            return req[-2] + req[-1][:-1]

        re_ord = Collator(
            requests,
            sort_fn=_collate,
            group_by="contexts"
            if self.backend == "causal" and self.logits_cache
            else None,
            group_fn=_lookup_one_token_cont,
        )

        # automatic (variable) batch size detection for vectorization
        # pull longest context sample from request
        n_reordered_requests = len(re_ord)
        batch_size = (
            self.batch_size
            if self.batch_size != "auto"
            else override_bs
            if override_bs is not None
            else 0
        )
        batch_fn = (
            self._batch_scheduler
            if self.batch_size == "auto"
            and n_reordered_requests > 0
            and not override_bs
            else None
        )

        chunks = re_ord.get_batched(n=batch_size, batch_fn=batch_fn)
        pbar = tqdm(
            total=len(requests),
            disable=(disable_tqdm or (self.rank != 0)),
            desc="Running loglikelihood requests",
        )
        for chunk in chunks:
            inps = []
            cont_toks_list = []
            inplens = []

            conts = []
            encoder_attns = []

            padding_len_inp = None
            padding_len_cont = None
            # because vectorizing is annoying, we first convert each (context, continuation) pair to padded
            # tensors, then we pack them together into a batch, call the model, and then pick it all apart
            # again because vectorizing is annoying

            for _, context_enc, continuation_enc in chunk:
                # sanity check
                assert len(context_enc) > 0
                assert len(continuation_enc) > 0
                assert len(continuation_enc) <= self.max_length

                # how this all works (illustrated on a causal decoder-only setup):
                #          CTX      CONT
                # inp    0 1 2 3|4 5 6 7 8 9   <- last token is deleted by inp[:, :-1]
                # model  \               \
                # logits   1 2 3|4 5 6 7 8 9   <- the ctx half gets tossed out by the
                # cont_toks      4 5 6 7 8 9      [:, -len(continuation_enc):, :self.vocab_size] slice

                # when too long to fit in context, truncate from the left
                if self.backend == "causal":
                    total_length = len(context_enc) + len(continuation_enc)
                    if total_length > self.max_length + 1:
                        eval_logger.warning(
                            f"Combined length of context ({len(context_enc)}) and continuation ({len(continuation_enc)}) "
                            f"exceeds model's maximum length ({self.max_length}). "
                            f"Truncating {total_length - self.max_length + 1} tokens from the left."
                        )
                    inp = torch.tensor(
                        (context_enc + continuation_enc)[-(self.max_length + 1) :][:-1],
                        dtype=torch.long,
                        device=self.device,
                    )
                    (inplen,) = inp.shape
                elif self.backend == "seq2seq":
                    inp = torch.tensor(
                        (context_enc)[-self.max_length :],
                        dtype=torch.long,
                        device=self.device,
                    )
                    (inplen,) = inp.shape

                    # build encoder attn masks
                    encoder_attns.append(torch.ones_like(inp))

                    cont = torch.tensor(
                        (continuation_enc)[-self.max_length :],
                        # TODO: left-shift these?
                        # TODO: our code assumes we never end up truncating conts for either model type
                        dtype=torch.long,
                        device=self.device,
                    )
                    (contlen,) = cont.shape

                    conts.append(cont)

                    padding_len_cont = (
                        max(padding_len_cont, contlen)
                        if padding_len_cont is not None
                        else contlen
                    )

                padding_len_inp = (
                    max(padding_len_inp, inplen)
                    if padding_len_inp is not None
                    else inplen
                )

                inps.append(inp)  # [1, inp_length]
                cont_toks_list.append(continuation_enc)
                inplens.append(inplen)

            # create encoder attn mask and batched conts, if seq2seq
            call_kwargs = {}
            if self.backend == "causal":
                batched_inps = pad_and_concat(
                    padding_len_inp, inps, padding_side="right"
                )  # [batch, padding_len_inp]
            elif self.backend == "seq2seq":
                # TODO: left-pad encoder inps and mask?
                batched_inps = pad_and_concat(
                    padding_len_inp, inps
                )  # [batch, padding_len_inp]
                batched_conts = pad_and_concat(
                    padding_len_cont, conts
                )  # [batch, padding_len_cont]
                batched_encoder_mask = pad_and_concat(
                    padding_len_inp, encoder_attns
                )  # [batch, padding_len_inp]
                call_kwargs = {
                    "attn_mask": batched_encoder_mask,
                    "labels": batched_conts,
                }

            multi_logits = F.log_softmax(
                self._model_call(batched_inps, **call_kwargs),
                dim=-1,
                dtype=self.softmax_dtype,
            )  # [batch, padding_length (inp or cont), vocab]

            for (request_str, ctx_tokens, _), logits, inplen, cont_toks in zip(
                chunk, multi_logits, inplens, cont_toks_list
            ):
                # Slice to original seq length
                contlen = len(cont_toks)
                # take only logits in the continuation
                # (discard context toks if decoder-only ; discard right-padding)
                # also discards + checks for "virtual tokens" in the causal LM's input window
                # from prompt/prefix tuning tokens, if applicable
                ctx_len = (
                    inplen + (logits.shape[0] - padding_len_inp)
                    if self.backend == "causal"
                    else None
                )
                logits = self._select_cont_toks(logits, contlen=contlen, inplen=ctx_len)
                logits = logits.unsqueeze(0)  # [1, seq, vocab]

                # Check if per-token argmax is exactly equal to continuation
                greedy_tokens = logits.argmax(dim=-1)

                # check for one-token continuation cache hits.
                # noop in case group_by != "contexts" or no cache hit and returns the
                # original args. Otherwise, expands the logits batch dimension and yields each
                # batch along with matching continuation tokens and prompt strings.
                # logits -> [1, seq, vocab]
                for request_str, cont_toks, logits in re_ord.get_cache(
                    req_str=request_str,
                    cxt_toks=ctx_tokens,
                    cont_toks=cont_toks,
                    logits=logits,
                ):
                    cont_toks = torch.tensor(
                        cont_toks, dtype=torch.long, device=self.device
                    ).unsqueeze(0)  # [1, seq]
                    # Use trailing slice [-cont_toks.shape[1]:] to handle variable length cont_len (but same ctx+cont[:-1]).
                    # i.e. continuations can be sliced at diff points. Collator ensures we have sufficient greedy_tokens
                    # by choosing key with longest cont if group_by="contexts".
                    max_equal = (
                        greedy_tokens[:, -cont_toks.shape[1] :] == cont_toks
                    ).all()

                    # Obtain log-probs at the corresponding continuation token indices
                    # last_token_slice = logits[:, -1, :].squeeze(0).tolist()
                    logits = torch.gather(logits, 2, cont_toks.unsqueeze(-1)).squeeze(
                        -1
                    )  # [1, seq]

                    # Answer: (log prob, is-exact-match)
                    answer = (float(logits.sum()), bool(max_equal))

                    res.append(answer)

                    if request_str is not None:
                        # special case: loglikelihood_rolling produces a number of loglikelihood requests
                        # all with cache key None. instead do add_partial on the per-example level
                        # in the loglikelihood_rolling() function for those.
                        self.cache_hook.add_partial(
                            "loglikelihood", request_str, answer
                        )
                    pbar.update(1)

        pbar.close()

        return re_ord.get_original(res)

    def generate_until(
        self, requests: List[Instance], disable_tqdm: bool = False
    ) -> List[str]:
        res = []

        def _collate(req: Tuple[str, dict]):
            """Defines the key for the sorted method"""
            # the negative sign on len(toks) sorts descending - this has a few advantages:
            # - time estimates will always be over not underestimates, which is more useful for planning
            # - to know the size of a batch when going through the list, you know the first one is always the batch
            #   padded context length. this is useful to simplify the batching logic and more importantly to make
            #   automatic adaptive batches much much easier to implement
            # - any OOMs will happen right away rather than near the end
            toks = self.tok_encode(req[0])
            return -len(toks), req[0]

        pbar = tqdm(
            total=len(requests),
            disable=(disable_tqdm or (self.rank != 0)),
            desc="Running generate_until requests",
        )
        adaptive_batch_size = None
        if self.batch_size == "auto":
            # using rolling window with maximum context
            print("Passed argument batch_size = auto. Detecting largest batch size")
            batch_size = self._detect_batch_size()
            print(f"Determined Largest batch size: {batch_size}")
            adaptive_batch_size = batch_size
        # for each different set of kwargs, we execute all requests, by batch.
        batch_size = (
            self.batch_size
            if self.batch_size != "auto"
            else adaptive_batch_size
            if adaptive_batch_size is not None
            else 0
        )
        batch_fn = (
            self._batch_scheduler
            if self.batch_size == "auto" and not adaptive_batch_size
            else None
        )

        # we group requests by their generation_kwargs,
        # so that we don't try to execute e.g. greedy sampling and temp=0.8 sampling
        # in the same batch.
        # group_fn=lambda x: x[1] -> x=(context, gen_kwargs)
        re_ords = Collator(
            [reg.args for reg in requests],
            sort_fn=_collate,
            group_by="gen_kwargs",
            group_fn=lambda x: x[1],
        )
        chunks = re_ords.get_batched(n=batch_size, batch_fn=batch_fn)
        eos = self.tok_decode(self.eot_token_id, skip_special_tokens=False)
        for chunk in chunks:
            contexts, all_gen_kwargs = zip(*chunk)
            # we assume all gen kwargs in the batch are the same
            # this is safe to assume because the `grouper` object ensures it.
            gen_kwargs = all_gen_kwargs[0]
            # unpack our keyword arguments.
            if isinstance(gen_kwargs, dict):
                kwargs = copy.deepcopy(gen_kwargs)  # edge case for repeats > 1
                # add EOS token to stop sequences
                until = handle_stop_sequences(kwargs.pop("until", None), eos=eos)
            else:
                raise ValueError(
                    f"Expected `kwargs` to be of type `dict` but got {type(gen_kwargs)}"
                )
            if "max_gen_toks" in kwargs.keys():
                max_gen_toks = kwargs.pop("max_gen_toks")
            else:
                max_gen_toks = self.max_gen_toks

            # set the max length in tokens of inputs ("context_enc")
            if self.backend == "causal":
                # max len for inputs = max length, minus room to generate the max new tokens
                max_ctx_len = self.max_length - max_gen_toks
                assert max_ctx_len > 0, (
                    f"Invalid configuration: requested max tokens to generate ({max_gen_toks}) must be less than model's maximum sequence length ({self.max_length})."
                )
            elif self.backend == "seq2seq":
                # max len for inputs = encoder's whole max_length
                max_ctx_len = self.max_length

            # encode, pad, and truncate contexts for this batch
            context_enc, attn_masks = self.tok_batch_encode(
                contexts,
                left_truncate_len=max_ctx_len,
                truncation=self.truncation,
            )
            context_enc = context_enc.to(self.device)
            attn_masks = attn_masks.to(self.device)

            if "max_length" not in kwargs:
                kwargs["max_length"] = context_enc.shape[1] + max_gen_toks

            # perform batched generation
            cont = self._model_generate(
                context=context_enc,
                attention_mask=attn_masks,
                stop=until,
                **kwargs,
            )

            cont_toks_list = cont.tolist()
            for cont_toks, context in zip(cont_toks_list, contexts):
                # discard context + left-padding toks if using causal decoder-only LM
                if self.backend == "causal":
                    cont_toks = cont_toks[context_enc.shape[1] :]

                s = self.tok_decode(cont_toks)

                # use secondary stop seqs to cut off should-have-been-stopped content post-hoc
                for term in until:
                    if len(term) > 0:
                        # ignore '' separator,
                        # for seq2seq case where self.tok_decode(self.eot_token_id) = ''
                        s = s.split(term)[0]

                res.append(s)

                self.cache_hook.add_partial("generate_until", (context, gen_kwargs), s)
                pbar.update(1)
        # reorder this group of results back to original unsorted form
        res = re_ords.get_original(res)

        pbar.close()

        return res

    def apply_chat_template(
        self, chat_history: List[Dict[str, str]], add_generation_prompt: bool = True
    ) -> str:
        """
        Method to apply a chat template to a list of chat history between user and model.
        """
        # Chat history is expected to be a list of dictionaries with keys "role" and "content". However, we are sending a string that came from a Dict[str, str] to the tokenizer. We need to ensure that the chat history is in the correct format.
        if "{'role':" in chat_history[0]['content']:
            chat_history = eval(chat_history[0]['content'])

        try:
            chat_templated = self.tokenizer.apply_chat_template(
                chat_history,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
                continue_final_message=not add_generation_prompt,
            )
        except jinja2.exceptions.TemplateError:
            eval_logger.warning(
                "Failed to apply chat template. removing the system role in chat history."
            )
            chat_history = [msg for msg in chat_history if msg["role"] != "system"]
            chat_templated = self.tokenizer.apply_chat_template(
                chat_history,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
                continue_final_message=not add_generation_prompt,
            )

        return chat_templated

    def get_model_info(self) -> dict:
        """
        Method to get Hugging Face model information for experiment reproducibility.
        """

        def get_model_num_params(model) -> int:
            if hasattr(model, "num_parameters"):
                return model.num_parameters()
            if hasattr(model, "parameters"):
                return sum(p.numel() for p in model.parameters())
            else:
                return -1

        def get_model_dtype(model) -> str:
            if hasattr(model, "dtype"):
                return model.dtype
            else:
                return ""

        def get_model_sha(pretrained: str, revision: str) -> str:
            try:
                model_info = HfApi().model_info(repo_id=pretrained, revision=revision)
                return model_info.sha
            except Exception as e:
                eval_logger.debug(
                    f"Failed to get model SHA for {pretrained} at revision {revision}. Error: {e}"
                )
                return ""

        model_info = {
            "model_num_parameters": get_model_num_params(self._model),
            "model_dtype": get_model_dtype(self._model),
            "model_revision": self.revision,
            "model_sha": get_model_sha(self.pretrained, self.revision),
        }
        if self.peft:
            model_info["peft_sha"] = get_model_sha(self.peft, self.revision)
        if self.delta:
            model_info["delta_sha"] = get_model_sha(self.delta, self.revision)
        return model_info
