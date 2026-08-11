from __future__ import annotations

import os
import threading
import traceback
import weakref
from collections import OrderedDict
from typing import Any
import torch

from modules import openclaw_cache_epochs

_ENABLED=False

def _read_cache_max():
    try:
        return max(0, int(os.environ.get('OPENCLAW_VAE_DECODE_GRAPH_CACHE_MAX', '4') or 0))
    except ValueError:
        return 4

_CACHE_MAX=_read_cache_max()
_LOCK=threading.RLock()
_CACHE=OrderedDict()
_KEY_LOCKS=weakref.WeakValueDictionary()
_FAILED_KEYS=set()
_COUNTERS={'captures':0,'replays':0,'bypasses':0,'failures':0,'invalidations':0}
_BYPASS_REASONS={}; _INVALIDATION_REASONS={}; _LAST_ERROR=None; _LAST_KEY=None; _LIFECYCLE_STATE={}

def _flag(n,d=False):
    v=os.environ.get(n)
    return d if v is None else v.strip().lower() in {'1','true','yes','on'}

def _bypass(r):
    _COUNTERS['bypasses']+=1; _BYPASS_REASONS[r]=_BYPASS_REASONS.get(r,0)+1
    openclaw_cache_epochs.observe('E11','bypass',reason='cache_disabled' if r in {'disabled','cache_disabled'} else 'unsafe_input')

def _clear_cache_locked():
    had=bool(_CACHE or _KEY_LOCKS or _FAILED_KEYS)
    _CACHE.clear()
    _KEY_LOCKS.clear()
    _FAILED_KEYS.clear()
    return had

def _key_lock(key):
    with _LOCK:
        key_lock=_KEY_LOCKS.get(key)
        if key_lock is None:
            key_lock=threading.RLock()
            _KEY_LOCKS[key]=key_lock
        return key_lock

def status():
    with _LOCK:
        return {'enabled':_ENABLED,'cache_size':len(_CACHE),'max_cache_size':_CACHE_MAX,**_COUNTERS,'bypass_reasons':dict(_BYPASS_REASONS),'invalidation_reasons':dict(_INVALIDATION_REASONS),'last_error':_LAST_ERROR,'last_key':repr(_LAST_KEY) if _LAST_KEY is not None else None,'lifecycle_state_keys':sorted(_LIFECYCLE_STATE)}

def set_enabled(enabled:bool, clear_cache:bool=False):
    global _ENABLED
    with _LOCK:
        _ENABLED=bool(enabled)
        if clear_cache:
            had_state=_clear_cache_locked()
            for k in _COUNTERS: _COUNTERS[k]=0
            _BYPASS_REASONS.clear(); _INVALIDATION_REASONS.clear()
            if had_state: openclaw_cache_epochs.observe('E11','invalidate',reason='cache_cleared')
            openclaw_cache_epochs.set_size('E11',current_size=0,capacity=_CACHE_MAX)
    return status()

def invalidate(reason:str, details:Any|None=None):
    with _LOCK:
        if _clear_cache_locked():
            _COUNTERS['invalidations']+=1; _INVALIDATION_REASONS[reason]=_INVALIDATION_REASONS.get(reason,0)+1
            openclaw_cache_epochs.observe('E11','invalidate',reason='dependency_changed')
            openclaw_cache_epochs.set_size('E11',current_size=0,capacity=_CACHE_MAX)
    return status()

def invalidate_if_changed(boundary:str,state:Any,reason:str|None=None):
    marker=repr(state)
    with _LOCK:
        old=_LIFECYCLE_STATE.get(boundary); _LIFECYCLE_STATE[boundary]=marker
        if old is None or old==marker: return status()
        if _clear_cache_locked():
            why=reason or f'{boundary}_changed'; _COUNTERS['invalidations']+=1; _INVALIDATION_REASONS[why]=_INVALIDATION_REASONS.get(why,0)+1
            openclaw_cache_epochs.observe('E11','invalidate',reason='dependency_changed')
            openclaw_cache_epochs.set_size('E11',current_size=0,capacity=_CACHE_MAX)
    return status()

def note_model_loaded(state=None): return invalidate_if_changed('model', state if state is not None else _runtime_identity()[0], 'model_changed')
def note_vae_loaded(state=None): return invalidate_if_changed('vae', state if state is not None else _runtime_identity()[1], 'vae_changed')

def _tensor_key(x): return (tuple(x.shape),str(x.dtype),str(x.device))

def _runtime_identity():
    try:
        from modules import shared, devices
        sd_model=getattr(shared,'sd_model',None); vae=getattr(sd_model,'first_stage_model',None); info=getattr(sd_model,'sd_checkpoint_info',None)
        return ((getattr(info,'filename',None),getattr(info,'shorthash',None),getattr(sd_model,'sd_model_hash',None),id(sd_model)),(getattr(sd_model,'loaded_vae_file',None),str(type(getattr(sd_model,'base_vae',None)).__name__),str(getattr(vae,'dtype',None)),id(vae)),(str(getattr(devices,'dtype_vae',None)),str(getattr(devices,'device',None))))
    except Exception:
        return (None,None,None)

def _bypass_reason(model,x,approximation):
    if not _ENABLED: return 'disabled'
    if approximation!=0: return 'vae_approximation'
    if not torch.is_tensor(x): return 'not_tensor'
    if not x.is_cuda: return 'not_cuda'
    if x.ndim!=4 or x.shape[1]!=4: return 'unsupported_shape'
    try:
        from modules import shared, lowvram
        opts=getattr(shared,'opts',None)
        if getattr(opts,'sd_vae_decode_method','Full')!='Full': return 'vae_decode_method'
        if getattr(opts,'hypertile_enable_vae',False): return 'hypertile_vae'
        if lowvram.is_enabled(model): return 'lowvram'
    except Exception:
        return 'state_probe_failed'
    if getattr(model,'first_stage_model',None) is None or not hasattr(model,'decode_first_stage'): return 'missing_vae'
    return None

def _key(model,x,approximation): return (_runtime_identity(),_tensor_key(x),approximation,os.environ.get('OPENCLAW_VAE_DECODE_GRAPHS'))

def _decode(model,x):
    from modules import devices
    with torch.no_grad(), devices.without_autocast():
        return model.decode_first_stage(x.to(model.first_stage_model.dtype))

def run(model,x,approximation=0):
    global _LAST_ERROR,_LAST_KEY
    reason=_bypass_reason(model,x,approximation)
    if reason is not None:
        with _LOCK: _bypass(reason)
        return None
    if _CACHE_MAX<=0:
        with _LOCK: _bypass('cache_disabled')
        return None
    key=_key(model,x,approximation)
    with _LOCK:
        _LAST_KEY=key
    key_lock=_key_lock(key)
    # Each graph entry owns mutable static input/output tensors. Serialize the
    # same key from input copy through output clone so concurrent API requests
    # cannot replay with mixed latent data; distinct shapes/devices still run
    # independently under separate locks.
    with key_lock:
        with _LOCK:
            entry=_CACHE.get(key)
            failed_before=key in _FAILED_KEYS
        if entry is not None:
            openclaw_cache_epochs.observe('E11','hit',reason='cache_hit',semantic_key=key)
            entry['input'].copy_(x, non_blocking=True)
            entry['graph'].replay()
            output=entry['output'].clone()
            with _LOCK:
                if key in _CACHE:
                    _CACHE.move_to_end(key)
                _COUNTERS['replays']+=1
            return output
        if failed_before:
            with _LOCK: _bypass('failed_key')
            return None
        openclaw_cache_epochs.observe('E11','miss',reason='cache_miss',semantic_key=key)
        try:
            static_input=x.detach().contiguous().clone()
            stream=torch.cuda.Stream(device=x.device)
            stream.wait_stream(torch.cuda.current_stream(x.device))
            with torch.cuda.stream(stream):
                warmup_output=_decode(model,static_input)
            torch.cuda.current_stream(x.device).wait_stream(stream)
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph): static_output=_decode(model,static_input)
            with _LOCK:
                _CACHE[key]={'graph':graph,'input':static_input,'output':static_output}; _CACHE.move_to_end(key)
                while len(_CACHE)>_CACHE_MAX:
                    evicted_key,_=_CACHE.popitem(last=False)
                    openclaw_cache_epochs.observe('E11','eviction',reason='capacity',semantic_key=evicted_key)
                _COUNTERS['captures']+=1
                openclaw_cache_epochs.observe('E11','publish',reason='published',semantic_key=key)
                openclaw_cache_epochs.set_size('E11',current_size=len(_CACHE),capacity=_CACHE_MAX)
            return warmup_output
        except Exception as exc:
            with _LOCK:
                _FAILED_KEYS.add(key); _COUNTERS['failures']+=1; _LAST_ERROR=''.join(traceback.format_exception_only(type(exc),exc)).strip(); openclaw_cache_epochs.observe('E11','reject',reason='capture_failed',semantic_key=key); _bypass('capture_failed')
            return None

set_enabled(_flag('OPENCLAW_VAE_DECODE_GRAPHS',False), clear_cache=True)
