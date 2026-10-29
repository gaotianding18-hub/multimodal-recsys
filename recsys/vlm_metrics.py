"""Generated attribute agreement with catalogue metadata, not human visual truth."""
import json
from collections import defaultdict


def attribute_metrics(predictions,targets):
    if not targets or len(predictions)!=len(targets):raise ValueError('Nonempty aligned predictions required')
    totals=defaultdict(float);classes={key:defaultdict(list) for key in ['product_type','color']}
    for output,target in zip(predictions,targets):
        try:parsed=json.loads(output)
        except (json.JSONDecodeError,TypeError):parsed=None
        valid=isinstance(parsed,dict) and set(parsed)=={'product_type','color'} and all(isinstance(v,str) for v in parsed.values())
        totals['json_valid_rate']+=valid;both=True
        for key in classes:
            correct=valid and parsed[key].strip().casefold()==target[key].strip().casefold()
            totals[key+'_agreement']+=correct;classes[key][target[key]].append(float(correct));both=both and correct
        totals['joint_agreement']+=both
    result={key:value/len(targets) for key,value in totals.items()}
    for key,groups in classes.items():result[key+'_macro_recall']=sum(sum(v)/len(v) for v in groups.values())/len(groups)
    result['items']=len(targets)
    return result
