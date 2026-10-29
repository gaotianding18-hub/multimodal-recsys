import unittest
from recsys.vlm_metrics import attribute_metrics

class AttributeTests(unittest.TestCase):
    def test_invalid_outputs_count_as_misses_and_macro_exposes_imbalance(self):
        targets=[{'product_type':'Socks','color':'Black'}]*3+[{'product_type':'Dress','color':'White'}]
        predictions=['{"product_type":"Socks","color":"Black"}']*2+['not JSON','{"product_type":"Socks","color":"White","material":"cotton"}']
        result=attribute_metrics(predictions,targets)
        self.assertEqual(result['json_valid_rate'],.5)
        self.assertEqual(result['joint_agreement'],.5)
        self.assertAlmostEqual(result['product_type_macro_recall'],1/3)
