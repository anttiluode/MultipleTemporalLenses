from __future__ import annotations
import argparse, copy, json, math
from pathlib import Path
import numpy as np
import torch
from multiple_temporal_lenses.baselines import SelectiveStateBaseline, TinyCausalTransformer
from multiple_temporal_lenses.config import canonical_long_config, canonical_short_config
from multiple_temporal_lenses.models import QueryConcatLensControl, QueryGatedLensModel
from multiple_temporal_lenses.tasks import make_query_timescale_batch
from multiple_temporal_lenses.train import TrainResult, classification_loss, evaluate_classifier, fit_classifier, model_metadata, seed_everything, write_receipt

TRAIN_SEED, VALIDATION_SEED, EVAL_BATCH_SIZE = 101, 202, 512
RESULTS = Path(__file__).resolve().parents[1] / "results"
PARTS = RESULTS / "gate4_parts"
STRESS_MODEL_NAMES = ("residual_gated", "raw_concat", "selective_state", "tiny_transformer")

def build_stress_models(bench):
    ts = tuple(float(x) for x in bench.anchor_lags)
    return {
        "residual_gated": QueryGatedLensModel(coordinate_mode="band", timescales=ts),
        "raw_concat": QueryConcatLensControl(coordinate_mode="raw", timescales=ts),
        "selective_state": SelectiveStateBaseline(state_dim=36),
        "tiny_transformer": TinyCausalTransformer(d_model=12, nhead=3, num_layers=1, ff_dim=24),
    }

def classify_routing_vs_selective(gated_per_seed, selective_per_seed):
    g, s = np.asarray(gated_per_seed, float), np.asarray(selective_per_seed, float)
    if g.shape != s.shape or not g.size: raise ValueError("gated/selective seed arrays must be non-empty and matched")
    d = g-s; se = float(d.std(ddof=1)/math.sqrt(d.size)) if d.size > 1 else 0.0
    gm, sm = float(g.mean()), float(s.mean()); matched = sm >= gm-se
    return {"gated_mean_accuracy":gm,"selective_mean_accuracy":sm,"mean_gated_minus_selective":gm-sm,
            "paired_difference_standard_error":se,"within_one_standard_error":bool(matched),
            "status":"SELECTIVE_STATE_EXPLAINS_RESULT" if matched else "GATED_EXCEEDS_THIS_SELECTIVE_CONTROL"}

def summarize_accuracy_by_scale(predictions, targets, scales, num_scales):
    out={}
    for i in range(num_scales):
        m=scales==i; out[str(i)] = float((predictions[m]==targets[m]).float().mean()) if bool(m.any()) else float("nan")
    return out

def _factory(bench, batch_size):
    return lambda seed, split: make_query_timescale_batch(bench, batch_size=batch_size, seed=seed, split=split)

def _fit(model, bench, cfg):
    seed_everything(TRAIN_SEED)
    return fit_classifier(model, _factory(bench,cfg.batch_size), cfg, TRAIN_SEED, VALIDATION_SEED)

def run_training_chunk(model, batch_factory, train_config, train_seed, validation_seed, checkpoint_path, max_new_steps):
    p=Path(checkpoint_path)
    if max_new_steps<=0: raise ValueError("max_new_steps must be positive")
    opt=torch.optim.AdamW(model.parameters(),lr=train_config.lr,weight_decay=0.0)
    if p.exists():
        x=torch.load(p,map_location="cpu",weights_only=False); model.load_state_dict(x["model_state"]); opt.load_state_dict(x["optimizer_state"])
        start=x["next_step"]; tl=list(x["train_losses"]); vl=list(x["validation_losses"]); best=float(x["best_loss"]); bs=x["best_step"]; bst=x["best_state"]; stale=x["stale_steps"]; stopped=x["stopped_early"]
    else:
        seed_everything(train_seed); start=0; tl=[]; vl=[]; best=float("inf"); bs=-1; bst=None; stale=0; stopped=False
    if (stopped or start>=train_config.max_steps) and bst is not None:
        model.load_state_dict(bst); model.eval(); return {"done":True,"next_step":start,"result":TrainResult(tl,vl,bs,best,stopped)}
    val=batch_factory(int(validation_seed),"dev"); stop=min(train_config.max_steps,start+max_new_steps); nxt=start; model.train()
    for step in range(start,stop):
        b=batch_factory(int(train_seed)*100_000+step,"dev"); opt.zero_grad(set_to_none=True); o=model(b.sequence,b.query); loss=classification_loss(o.logits,b.target); loss.backward(); opt.step(); tl.append(float(loss.detach().cpu()))
        model.eval()
        with torch.no_grad(): v=float(classification_loss(model(val.sequence,val.query).logits,val.target).detach().cpu())
        vl.append(v); model.train()
        if v < best-1e-12: best=v; bs=step; bst=copy.deepcopy(model.state_dict()); stale=0
        else:
            stale+=1
            if stale>=train_config.patience: stopped=True; nxt=step+1; break
        nxt=step+1
    done=stopped or nxt>=train_config.max_steps; p.parent.mkdir(parents=True,exist_ok=True)
    torch.save({"model_state":model.state_dict(),"optimizer_state":opt.state_dict(),"next_step":nxt,"train_losses":tl,"validation_losses":vl,"best_loss":best,"best_step":bs,"best_state":bst,"stale_steps":stale,"stopped_early":stopped},p)
    if not done: return {"done":False,"next_step":nxt}
    if bst is None: raise RuntimeError("training produced no model state")
    model.load_state_dict(bst); model.eval(); return {"done":True,"next_step":nxt,"result":TrainResult(tl,vl,bs,best,stopped)}

def _scale_eval(model, bench, seeds, split):
    rows=[]; model.eval()
    with torch.no_grad():
        for seed in seeds:
            b=make_query_timescale_batch(bench,EVAL_BATCH_SIZE,int(seed),split); p=model(b.sequence,b.query).logits.argmax(1); s=b.query.argmax(1)
            rows.append({"seed":int(seed),"overall_accuracy":float((p==b.target).float().mean()),"accuracy_by_scale":summarize_accuracy_by_scale(p,b.target,s,bench.num_scales)})
    return {"per_seed":rows,"mean_accuracy":float(np.mean([r["overall_accuracy"] for r in rows])),"mean_accuracy_by_scale":{str(i):float(np.mean([r["accuracy_by_scale"][str(i)] for r in rows])) for i in range(bench.num_scales)}}

def _short_payload(model, fit, bench):
    ev=evaluate_classifier(model,_factory(bench,EVAL_BATCH_SIZE),bench.eval_seeds,kind="frozen")
    return {"training":{"best_step":fit.best_step,"best_validation_loss":fit.best_validation_loss,"stopped_early":fit.stopped_early},"frozen":{"mean_accuracy":ev.mean_accuracy,"mean_loss":ev.mean_loss,"per_seed":ev.per_seed},"model_metadata":model_metadata(model,bench.sequence_length),"explicit_history_proxy_T72":model.explicit_history_scalar_proxy(72),"explicit_history_proxy_T208":model.explicit_history_scalar_proxy(208)}

def run_short_transformer():
    b,_,c=canonical_short_config(); seed_everything(TRAIN_SEED); m=TinyCausalTransformer(d_model=12,nhead=3,num_layers=1,ff_dim=24); return _short_payload(m,_fit(m,b,c),b)

def run_short_transformer_chunk(n):
    b,_,c=canonical_short_config(); seed_everything(TRAIN_SEED); m=TinyCausalTransformer(d_model=12,nhead=3,num_layers=1,ff_dim=24)
    st=run_training_chunk(m,_factory(b,c.batch_size),c,TRAIN_SEED,VALIDATION_SEED,PARTS/"checkpoints/short_transformer.pt",n)
    if not st["done"]: return st
    payload=_short_payload(m,st["result"],b); write_part("short_transformer",payload); return {"done":True,"next_step":st["next_step"],"payload":payload}

def run_stress_model(name):
    b,_,c=canonical_long_config(); models=build_stress_models(b)
    if name not in models: raise KeyError(name)
    m=models[name]; fit=_fit(m,b,c); return {"model":name,"training":{"best_step":fit.best_step,"best_validation_loss":fit.best_validation_loss,"stopped_early":fit.stopped_early},"stress":_scale_eval(m,b,b.eval_seeds,"stress"),"model_metadata":model_metadata(m,b.sequence_length)}

def run_stress_model_chunk(name,n):
    b,_,c=canonical_long_config(); seed_everything(TRAIN_SEED); models=build_stress_models(b)
    if name not in models: raise KeyError(name)
    m=models[name]; st=run_training_chunk(m,_factory(b,c.batch_size),c,TRAIN_SEED,VALIDATION_SEED,PARTS/f"checkpoints/stress_{name}.pt",n)
    if not st["done"]: return st
    fit=st["result"]; payload={"model":name,"training":{"best_step":fit.best_step,"best_validation_loss":fit.best_validation_loss,"stopped_early":fit.stopped_early},"stress":_scale_eval(m,b,b.eval_seeds,"stress"),"model_metadata":model_metadata(m,b.sequence_length)}; write_part(f"stress_{name}",payload); return {"done":True,"next_step":st["next_step"],"payload":payload}

def write_part(name,payload):
    PARTS.mkdir(parents=True,exist_ok=True); p=PARTS/f"{name}.json"; p.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n",encoding="utf-8"); return p

def _short_refs():
    g=json.loads((RESULTS/"gate1_query_timescale.json").read_text()); s=json.loads((RESULTS/"gate4_selective_state_control.json").read_text())
    return [r["accuracy"] for r in g["metrics"]["models"]["residual_gated"]["per_seed"]],[r["accuracy"] for r in s["metrics"]["frozen"]["per_seed"]]

def assemble():
    sb,_,sc=canonical_short_config(); lb,_,lc=canonical_long_config(); short=json.loads((PARTS/"short_transformer.json").read_text()); stress={n:json.loads((PARTS/f"stress_{n}.json").read_text()) for n in STRESS_MODEL_NAMES}; g,s=_short_refs(); routing=classify_routing_vs_selective(g,s)
    sm={"tiny_transformer":short,"routing_vs_selective_state":routing,"claim_boundary":"Transformer is an explicit-history control with sequence-growing memory; SelectiveStateBaseline is a minimal Mamba-like control, not Mamba."}
    write_receipt(RESULTS/"gate4_controls.json",experiment="gate4_explicit_history_and_selective_state_controls",config={"benchmark":sb,"training":sc,"train_seed":TRAIN_SEED,"validation_seed":VALIDATION_SEED,"transformer":{"d_model":12,"nhead":3,"num_layers":1,"ff_dim":24},"routing_interpretation_rule":"selective state matches/exceeds residual-gated if within one paired standard error"},metrics=sm,seeds=sb.eval_seeds,model_metadata={"tiny_transformer":short["model_metadata"]},kind="frozen")
    lm={"models":stress,"best_overall_model":max(stress,key=lambda n:stress[n]["stress"]["mean_accuracy"]),"transformer_memory_growth":{"T72":short["explicit_history_proxy_T72"],"T208":short["explicit_history_proxy_T208"],"ratio":short["explicit_history_proxy_T208"]/short["explicit_history_proxy_T72"]}}
    write_receipt(RESULTS/"gate4_long_horizon_stress.json",experiment="gate4_long_horizon_stress",config={"benchmark":lb,"training":lc,"train_seed":TRAIN_SEED,"validation_seed":VALIDATION_SEED,"models":list(STRESS_MODEL_NAMES),"no_tuning_note":"Uses preregistered (2,20,200), T=208 task with unchanged optimizer ceiling and no architecture search."},metrics=lm,seeds=lb.eval_seeds,model_metadata={n:p["model_metadata"] for n,p in stress.items()},kind="stress"); return {"short":sm,"long":lm}

def main():
    p=argparse.ArgumentParser(); p.add_argument("--short-transformer",action="store_true"); p.add_argument("--short-transformer-chunk",type=int); p.add_argument("--stress-model",choices=STRESS_MODEL_NAMES); p.add_argument("--stress-chunk",nargs=2,metavar=("MODEL","STEPS")); p.add_argument("--assemble",action="store_true"); p.add_argument("--dry",action="store_true"); a=p.parse_args()
    if a.dry:
        s,_,_=canonical_short_config(); l,_,_=canonical_long_config(); print(json.dumps({"stress_models":list(STRESS_MODEL_NAMES),"short_sequence_length":s.sequence_length,"long_sequence_length":l.sequence_length,"long_anchor_lags":list(l.anchor_lags)},indent=2)); return
    if a.short_transformer: payload=run_short_transformer(); write_part("short_transformer",payload); print(json.dumps(payload["frozen"],indent=2)); return
    if a.short_transformer_chunk is not None: st=run_short_transformer_chunk(a.short_transformer_chunk); print(json.dumps(st["payload"]["frozen"] if st["done"] else st,indent=2)); return
    if a.stress_chunk is not None: n,k=a.stress_chunk; st=run_stress_model_chunk(n,int(k)); print(json.dumps(st["payload"]["stress"] if st["done"] else st,indent=2)); return
    if a.stress_model: payload=run_stress_model(a.stress_model); write_part(f"stress_{a.stress_model}",payload); print(json.dumps(payload["stress"],indent=2)); return
    if a.assemble: payload=assemble(); print(json.dumps({"routing_interpretation":payload["short"]["routing_vs_selective_state"],"best_long_model":payload["long"]["best_overall_model"]},indent=2)); return
    raise SystemExit("choose --short-transformer, --stress-model, --assemble, or --dry")
if __name__ == "__main__": main()
