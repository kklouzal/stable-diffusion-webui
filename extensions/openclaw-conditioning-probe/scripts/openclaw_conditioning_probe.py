from __future__ import annotations
import hashlib, os, threading, time
from typing import Any
import torch
from fastapi import FastAPI
from modules import script_callbacks, shared, openclaw_cache_epochs, sd_hijack

_LOCK=threading.RLock()

def _sha_bytes(data: bytes): return hashlib.sha256(data).hexdigest()
def _tensor(t):
    if not isinstance(t, torch.Tensor): return None
    with torch.no_grad():
        x=t.detach()
        meta={"shape":list(x.shape),"dtype":str(x.dtype),"device":str(x.device),"object_id":id(t),"storage_ptr":None,"version":None}
        try: meta["storage_ptr"]=x.untyped_storage().data_ptr()
        except Exception: pass
        try:
            y=x.to(device="cpu").contiguous()
            meta["sha256"]=_sha_bytes(y.view(torch.uint8).numpy().tobytes())
            if y.is_floating_point():
                z=y.float(); meta.update(min=float(z.min()) if z.numel() else None,max=float(z.max()) if z.numel() else None,sum=float(z.double().sum()))
        except Exception as e: meta["error"]=repr(e)
        return meta

def _walk(v, depth=0):
    if depth>10: return {"type":type(v).__name__,"repr":repr(v)[:160]}
    if isinstance(v,torch.Tensor): return _tensor(v)
    if isinstance(v,(list,tuple)): return [_walk(x,depth+1) for x in v]
    if isinstance(v,dict): return {str(k):_walk(x,depth+1) for k,x in sorted(v.items(),key=lambda kv:str(kv[0]))}
    if v is None or isinstance(v,(str,int,float,bool)): return v
    return {"type":type(v).__name__,"object_id":id(v),"repr":repr(v)[:240]}

def _module_digest(name,m):
    attrs={}
    attr_names=["weight","bias","network_weights_backup","network_bias_backup","network_current_names","network_mxfp8_base_weight","network_mxfp8_base_bias","network_nvfp4_base_weight","network_nvfp4_base_bias","network_mxfp8_merged_lora_applied","network_nvfp4_merged_lora_applied","_data","_scale","_scale_2","block_size","weight_scale","weight_scale_2","input_scale","scale","scales"]
    for a in attr_names:
        if hasattr(m,a):
            try: attrs[a]=_walk(getattr(m,a))
            except Exception as e: attrs[a]={"error":repr(e)}
    return {"name":name,"type":type(m).__module__+"."+type(m).__name__,"object_id":id(m),"network_layer_name":getattr(m,"network_layer_name",None),"attrs":attrs}

def _representatives(model):
    adapted=[]; plain=[]
    for name,m in model.named_modules():
        has=bool(getattr(m,"network_layer_name",None)) or any(hasattr(m,a) for a in ("network_nvfp4_base_weight","network_mxfp8_base_weight","network_weights_backup"))
        if hasattr(m,"weight"):
            (adapted if has else plain).append((name,m))
    def pick(xs,n=8):
        if len(xs)<=n:return xs
        ids=sorted(set([0,1,len(xs)//4,len(xs)//2,3*len(xs)//4,len(xs)-2,len(xs)-1]))
        return [xs[i] for i in ids[:n]]
    return {"adapted":[_module_digest(n,m) for n,m in pick(adapted)],"plain":[_module_digest(n,m) for n,m in pick(plain)]}

def _loaded_networks():
    try:
        import networks
        out=[]
        for n in networks.loaded_networks:
            out.append({"name":getattr(n,"name",None),"mentioned_name":getattr(n,"mentioned_name",None),"mtime":getattr(n,"mtime",None),"source_signature":_walk(getattr(n,"source_signature",None)),"te_multiplier":getattr(n,"te_multiplier",None),"unet_multiplier":getattr(n,"unet_multiplier",None),"dyn_dim":getattr(n,"dyn_dim",None),"object_id":id(n)})
        return {"loaded":out,"in_memory_keys":sorted(networks.networks_in_memory.keys()),"loaded_object_id":id(networks.loaded_networks)}
    except Exception as e:return {"error":repr(e)}

def snapshot(label):
    from modules.processing import StableDiffusionProcessing as P, StableDiffusionProcessingTxt2Img as T
    model=shared.sd_model
    conditioners=[]
    for attr in ("conditioner","cond_stage_model"):
        root=getattr(model,attr,None)
        if root is not None:
            conditioners.append({"root":attr,"root_type":type(root).__module__+"."+type(root).__name__,"root_id":id(root),"representatives":_representatives(root)})
    cond={k:_walk(getattr(cls,k)) for cls,k in ((P,"cached_c"),(P,"cached_uc"),(T,"cached_hr_c"),(T,"cached_hr_uc"))}
    
    def _patch_multicond(obj):
        try:
            vals=[]
            for batch in getattr(obj, "batch", []):
                row=[]
                for comp in batch:
                    row.append({"weight":getattr(comp,"weight",None),"schedules":[{"end_at_step":getattr(sched,"end_at_step",None),"cond":_walk(getattr(sched,"cond",None))} for sched in getattr(comp,"schedules",[])]})
                vals.append(row)
            return {"type":type(obj).__name__,"object_id":id(obj),"shape":_walk(getattr(obj,"shape",None)),"batch":vals}
        except Exception as e:return {"type":type(obj).__name__,"object_id":id(obj),"error":repr(e)}
    for k,v in list(cond.items()):
        actual=getattr({'cached_c':P,'cached_uc':P,'cached_hr_c':T,'cached_hr_uc':T}[k],k)
        if isinstance(actual,list) and len(actual)>1 and type(actual[1]).__name__ == "MulticondLearnedConditioning": cond[k][1]=_patch_multicond(actual[1])
    emb=getattr(getattr(shared.sd_model,"embedding_db",None),"word_embeddings",None)
    if emb is None: emb=getattr(getattr(sd_hijack,"model_hijack",None),"embedding_db",None)
    quant_model={a:_walk(getattr(model,a,None)) for a in ("network_nvfp4_active_config_signature","network_nvfp4_prepare_stats","network_nvfp4_prepare_error","network_nvfp4_active_config_ready","network_nvfp4_managed_modules","network_mxfp8_active_config_signature","network_mxfp8_prepare_stats","network_mxfp8_active_config_ready")}
    return {"schema":1,"conditioners":conditioners,"quant_model":quant_model,"label":label,"time":time.time(),"pid":os.getpid(),"epochs":openclaw_cache_epochs.snapshot(),"conditioning":cond,"networks":_loaded_networks(),"model_id":id(model),"model_type":type(model).__module__+"."+type(model).__name__,"representatives":_representatives(model),"textual_inversion":_walk(emb),"torch_rng_sha256":_sha_bytes(torch.get_rng_state().numpy().tobytes()),"cuda_rng":[_sha_bytes(x.cpu().numpy().tobytes()) for x in torch.cuda.get_rng_state_all()] if torch.cuda.is_available() else []}

def on_app_started(_:Any,app:FastAPI):
    @app.get('/sdapi/v1/openclaw/conditioning-probe/self-test')
    async def self_test():
        a=torch.tensor([1.,2.,3.]); b=_tensor(a); c=_tensor(a); a[0]=4.; d=_tensor(a)
        ok=b['sha256']==c['sha256'] and b['sha256']!=d['sha256'] and b['shape']==[3]
        return {"ok":ok,"stable":b['sha256']==c['sha256'],"sensitive":b['sha256']!=d['sha256'],"sample":d}
    @app.get('/sdapi/v1/openclaw/conditioning-probe/snapshot')
    async def snap(label:str='snapshot'):
        with _LOCK:return snapshot(label)
script_callbacks.on_app_started(on_app_started)
