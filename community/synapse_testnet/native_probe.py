"""Unbilled probe of the previous native Synapse /v1/chat/completions contract.
Uses only a locally configured endpoint and an original synthetic prompt; never awards credits.
"""
import argparse
import json
import os
import stat
import time
from pathlib import Path
from urllib.request import Request,build_opener,ProxyHandler,HTTPRedirectHandler
from .ledger import require,integer,canonical,sha

class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self,*args,**kwargs):return None

def read_token(path):
    p=Path(path); fd=os.open(p,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK)
    with os.fdopen(fd,'rb') as f:
        st=os.fstat(f.fileno()); require(stat.S_ISREG(st.st_mode) and st.st_uid==os.getuid() and st.st_mode&0o077==0 and st.st_size<=1024,'UNSAFE_NATIVE_TOKEN'); token=f.read(1025).decode().strip()
    require(16<=len(token)<=512 and token.isascii() and not any(c.isspace() for c in token),'INVALID_NATIVE_TOKEN'); return token

def probe(port,token_file,model,max_tokens=8):
    integer(port,1,65535); integer(max_tokens,1,128); require(isinstance(model,str) and 0<len(model)<=200,'MODEL_ID_REQUIRED')
    token=read_token(token_file); opener=build_opener(ProxyHandler({}),NoRedirect()); prompt='<s> </s> Hello'
    body={'model':model,'messages':[{'role':'user','content':prompt}],'max_tokens':max_tokens,'temperature':0,'stream':False}
    r=Request(f'http://127.0.0.1:{port}/v1/chat/completions',data=canonical(body),headers={'Authorization':'Bearer '+token,'Content-Type':'application/json'}); start=time.monotonic()
    with opener.open(r,timeout=60) as response:
        require(response.headers.get_content_type()=='application/json','NATIVE_CONTENT_TYPE'); data=response.read(1024*1024+1); require(len(data)<=1024*1024,'NATIVE_RESPONSE_LIMIT')
    elapsed=time.monotonic()-start; obj=json.loads(data)
    require(obj.get('model')==model and len(obj.get('choices',[]))==1,'NATIVE_MODEL_OR_CHOICE_MISMATCH')
    choice=obj['choices'][0]; text=choice['message']['content']; require(isinstance(text,str) and choice.get('finish_reason') in ('stop','length'),'NATIVE_INCOMPLETE')
    u=obj['usage']; n=integer(u['completion_tokens'],0,max_tokens); input_count=integer(u['prompt_tokens'],1,32768); require(u.get('total_tokens')==n+input_count,'NATIVE_USAGE_MISMATCH')
    return {'mode':'unbilled_native_contract_probe','model':model,'output_sha256':sha(text.encode()),'prompt_fixture_sha256':sha(prompt.encode()),'prompt_tokens':input_count,'completion_tokens':n,'finish_reason':choice['finish_reason'],'elapsed_seconds':elapsed,'credits_charged':0}

def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--port',type=int,required=True); p.add_argument('--token-file',required=True); p.add_argument('--model',required=True); p.add_argument('--max-tokens',type=int,default=8); p.add_argument('--reference-port',type=int); p.add_argument('--reference-token-file'); a=p.parse_args(); result={'candidate':probe(a.port,a.token_file,a.model,a.max_tokens)}
    if a.reference_port is not None:
        if not a.reference_token_file:p.error('reference token file required')
        result['reference']=probe(a.reference_port,a.reference_token_file,a.model,a.max_tokens); fields=('output_sha256','prompt_tokens','completion_tokens','finish_reason'); result['parity']=all(result['candidate'][f]==result['reference'][f] for f in fields)
    print(json.dumps(result,indent=2)); return 0 if result.get('parity',True) else 1
if __name__=='__main__':raise SystemExit(main())
