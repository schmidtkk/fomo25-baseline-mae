#!/usr/bin/env python3
import unittest
import torch
import os, sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from models.networks.unet import UNet
from models.networks.heads import ClsRegHead


class TestClsHeadLinear(unittest.TestCase):
    def test_cls_head_linear_single_encoder(self):
        net = UNet(mode="classification", input_channels=4, output_channels=2, use_multi_encoder=False)
        self.assertIsInstance(net.decoder, ClsRegHead)
        self.assertEqual(net.decoder.in_channels, 64 * 16)
        x = torch.randn(1, 4, 32, 128, 128)
        y = net(x)
        self.assertEqual(y.shape[-1], 2)

    def test_cls_head_linear_multi_encoder(self):
        net = UNet(
            mode="classification",
            input_channels=2,
            output_channels=2,
            use_multi_encoder=True,
            multi_encoder_modalities=["DWI", "T2FLAIR"],
            multi_encoder_num_modalities_global=2,
            modality_to_global_group={"DWI": "dwi", "T2FLAIR": "flair"},
            global_vocab=["t1", "t2", "flair", "dwi", "other"],
        )
        self.assertIsInstance(net.decoder, ClsRegHead)
        self.assertEqual(net.decoder.in_channels, 64 * 16)
        x = torch.randn(1, 2, 32, 128, 128)
        y = net(x)
        self.assertEqual(y.shape[-1], 2)


if __name__ == "__main__":
    unittest.main()


