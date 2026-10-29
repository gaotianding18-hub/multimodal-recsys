import requests,json,hashlib
from pathlib import Path
from recsys.hm_train import ROOT
root=ROOT;out=root/'artifacts/backbones/smolvlm-500m';out.mkdir(parents=True,exist_ok=True)
s=requests.Session();base='https://hf-mirror.com';model='HuggingFaceTB/SmolVLM-500M-Instruct'
r=s.get(f'{base}/api/models/{model}',timeout=20);r.raise_for_status();meta=r.json();sha='a7da5b986cb59b408707209984f360a5f4ad7e47'
if meta.get('cardData',{}).get('license')!='apache-2.0' and 'license:apache-2.0' not in meta.get('tags',[]):raise RuntimeError('Unexpected model license')
(root/'reports/smolvlm_source_audit.json').write_text(json.dumps(meta,indent=2))
for f in meta['siblings']:
 name=f['rfilename']
 if '/' in name or not (name.endswith(('.json','.txt','.model','.safetensors')) or name in ['README.md','LICENSE']):continue
 p=out/name
 if p.exists():continue
 r=s.get(f'{base}/{model}/resolve/{sha}/{name}',stream=True,timeout=(20,60));r.raise_for_status()
 temp=p.with_suffix(p.suffix+'.part')
 with temp.open('wb') as h:
  for chunk in r.iter_content(1024*1024):h.write(chunk)
 temp.rename(p);print(name,p.stat().st_size,flush=True)

weight=out/'model.safetensors'
digest=hashlib.sha256(weight.read_bytes()).hexdigest()
if digest!='d05b567eeaf534e83d375551f068ed57b5f52d37c657197f644af5ef9db091a2':raise RuntimeError('Weight checksum mismatch')
print('Verified model SHA256',digest,flush=True)
