"""Regression tests for the fork's --no-ple option with upstream's converter.

Run with PYTHONPATH=gguf-py:. python3 tests/test-qwen4exp-no-ple.py.
Only GGUF metadata I/O and inherited parameter setup are mocked; tensor
filtering uses the real converter and real PyTorch tensors.
"""
import unittest
from unittest.mock import MagicMock, patch

import torch

from conversion.qwen4exp import Qwen4ExpTextModel
from conversion.qwen import _Qwen35MRopeMixin


class NoPLE(unittest.TestCase):
    def setUp(self):
        self.model = object.__new__(Qwen4ExpTextModel)
        self.model._no_ple = True
        self.model.mtp_only = False
        self.model.block_count = 1
        self.model.hparams = {
            "hc_count": 4, "hc_lowrank": 2, "num_hidden_layers": 1,
            "indexer_n_heads": 1, "indexer_head_dim": 4,
            "indexer_budget": 8, "indexer_compress_ratio": 1,
            "layer_types": ["full_attention"], "ple_layer_ids": [1],
            "ngram_size": 2, "heads_per_ngram": 1,
            "ple_conv_kernel_size": 3, "eos_token_id": 2,
        }
        self.writer = MagicMock()
        self.model.gguf_writer = self.writer
        self.model._ple_row_dim = None

    def test_no_ple_filters_table_hash_constants_and_layer_tensors(self):
        for name in (
            "model.layers.0.ple.norm_key.weight",
            "model.ple_embedding.layer_multipliers",
            "model.ple_embedding.ngram_embedding.shard_0.weight",
        ):
            with self.subTest(name=name):
                self.assertEqual(self.model.modify_tensors(torch.ones(2, 2), name, 0), [])

    def test_no_ple_omits_metadata_and_does_not_read_hash_constants(self):
        with patch.object(_Qwen35MRopeMixin, "set_gguf_parameters"), \
                patch.object(self.model, "_read_hash_constants", side_effect=AssertionError("unexpected read")):
            self.model.set_gguf_parameters()
        self.writer.add_ple_layers.assert_not_called()
        self.writer.add_hyper_connection_count.assert_called_once_with(4)

    def test_default_ple_still_exports_metadata(self):
        self.model._no_ple = False
        with patch.object(_Qwen35MRopeMixin, "set_gguf_parameters"), \
                patch.object(self.model, "_read_hash_constants", return_value=[1, 2]):
            self.model.set_gguf_parameters()
        self.writer.add_ple_layers.assert_called_once_with([0])
        self.writer.add_ple_layer_multipliers.assert_called_once_with([1, 2])

    def test_upstream_mtp_projection_merge_is_unchanged(self):
        self.model._mtp_fc = {}
        self.model.format_tensor_name = MagicMock(return_value="nextn_eh_proj.weight")
        embedding, hidden = torch.ones(2, 3), torch.zeros(2, 3)
        self.assertEqual(self.model.modify_tensors(embedding, "model.layers.1.nextn_fc_embedding.weight", 1), [])
        tensors = list(self.model.modify_tensors(hidden, "model.layers.1.nextn_fc_hidden.weight", 1))
        self.assertEqual(tensors[0][0], "nextn_eh_proj.weight")
        torch.testing.assert_close(tensors[0][1], torch.cat([embedding, hidden], dim=1))
        self.assertEqual(self.model._mtp_fc, {})


if __name__ == "__main__":
    unittest.main()
