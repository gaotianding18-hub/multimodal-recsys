import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from recsys.hm_serve import Engine
from recsys.hm_train import ROOT


class ServingContractTests(unittest.TestCase):
    def test_real_model_rejects_misaligned_catalogue(self):
        with tempfile.TemporaryDirectory() as folder:
            directory=Path(folder)
            (directory/'catalogue.json').write_text(json.dumps([{'article_id':'wrong-order'}]))
            with patch('recsys.hm_serve.DATA',directory):
                with self.assertRaisesRegex(RuntimeError,'ID order mismatch'):Engine()

    def test_real_model_rejects_changed_training_protocol(self):
        original=json.loads
        def load_changed(text,*args,**kwargs):
            result=original(text,*args,**kwargs)
            if isinstance(result,dict) and set(result)==set(original((ROOT/'configs/hm.json').read_text())):
                result['history_length']=31
            return result
        with patch('recsys.hm_serve.json.loads',side_effect=load_changed):
            with self.assertRaisesRegex(RuntimeError,'differs from model protocol'):Engine()


if __name__=='__main__':unittest.main()
