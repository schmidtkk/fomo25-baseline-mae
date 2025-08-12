import unittest


def remap_encoder_key_to_modality(key: str, modality: str) -> str:
    # Helper replicating the remap logic in finetune.py
    assert key.startswith("model.encoder."), key
    rest = key[len("model.encoder.") :]
    return f"model.encoder.encoders.{modality}.{rest}"


class TestWeightKeyRemap(unittest.TestCase):
    def test_remap(self):
        src = "model.encoder.in_conv.conv1.conv.weight"
        self.assertEqual(
            remap_encoder_key_to_modality(src, "DWI"),
            "model.encoder.encoders.DWI.in_conv.conv1.conv.weight",
        )
        src2 = "model.encoder.encoder_conv3.conv2.norm.bias"
        self.assertEqual(
            remap_encoder_key_to_modality(src2, "T2FLAIR"),
            "model.encoder.encoders.T2FLAIR.encoder_conv3.conv2.norm.bias",
        )




