import unittest
from data.datamodule import PretrainDataModule

class TestModalityFilter(unittest.TestCase):
    def setUp(self):
        self.samples = [
            'sub_001_ses_001_t1',
            'sub_002_ses_001_t1_aug.npy',
            'sub_003_ses_002_t2',
            'sub_004_ses_002_flair',
            'sub_005_ses_003_dwi_extra',
            'sub_006_ses_003_scan.npy',
            'sub_007_ses_004_pd',
            'sub_008_ses_004_swi',
            'sub_009_ses_005_t2s',
            'bad_format_name',
        ]

    def test_all_mode(self):
        kept, stats = PretrainDataModule.filter_samples_by_modality(self.samples, 'all')
        self.assertEqual(len(kept), len(self.samples))
        self.assertIn('t1', stats['kept_modalities'])

    def test_single_t1(self):
        kept, stats = PretrainDataModule.filter_samples_by_modality(self.samples, 't1')
        self.assertTrue(all('t1' in s for s in kept))
        self.assertEqual(stats['kept_modalities'].get('t1', 0), 2)

    def test_other_group(self):
        kept, stats = PretrainDataModule.filter_samples_by_modality(self.samples, 'other')
        self.assertTrue(all(any(m in s for m in ['scan','pd','swi','t2s']) for s in kept))
        # Expect 4 kept (scan, pd, swi, t2s)
        self.assertEqual(stats['total_filtered'], 4)

if __name__ == '__main__':
    pass
