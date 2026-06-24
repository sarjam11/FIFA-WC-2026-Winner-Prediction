

import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from pathlib import Path
from collections import defaultdict, Counter
from scipy.stats import poisson
from scipy.optimize import minimize
from sklearn.metrics import log_loss
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
import xgboost as xgb, lightgbm as lgb

DATA_DIR = Path("C:/Users/Sarjam/OneDrive/Desktop/FIFA WC Prediction/simulation")
OUT_DIR  = Path("C:/Users/Sarjam/OneDrive/Desktop/FIFA WC Prediction/simulation")

HOSTS = {"United States", "Canada", "Mexico"}
HOST_ELO_BOOST = 50
FRIENDLY_WEIGHT = 0.5
N_SIMS = 50000
np.random.seed(2026)

PEDIGREE_BOOST = {
    "Argentina":   60,  # 2022 WC winner (defending champion)
    "France":      80,  # 2x WC finalist (2018 winner + 2022 finalist), Mbappe CL top scorer, Dembele Ballon d'Or
    "Croatia":     50,  # 2022 WC 3rd + 2018 WC finalist
    "Spain":       40,  # 2024 Euro Winner
    "England":     60,  # 2020 + 2024 Euro finalist, 2018 WC SF
    "Morocco":     50,  # 2022 WC SF + 2026 AFCON championn 1   
    "Senegal":     30   # 2026 AFCON runners
}

CONF_DISCOUNT = {"UEFA":0,"CONMEBOL":0,"CAF":-10,"AFC":-20,"CONCACAF":-30,"OFC":-40}

TEAM_CONF = {
    #UEFA
    "Spain":"UEFA","England":"UEFA","France":"UEFA","Germany":"UEFA",
    "Portugal":"UEFA","Netherlands":"UEFA","Croatia":"UEFA","Switzerland":"UEFA",
    "Norway":"UEFA","Austria":"UEFA","Scotland":"UEFA","Sweden":"UEFA",
    "Czech Republic":"UEFA","Bosnia and Herzegovina":"UEFA","Belgium":"UEFA","Turkey":"UEFA",

    #CONMEBOL
    "Argentina":"CONMEBOL","Brazil":"CONMEBOL","Colombia":"CONMEBOL",
    "Ecuador":"CONMEBOL","Paraguay":"CONMEBOL","Uruguay":"CONMEBOL",

    #CAF
    "Morocco":"CAF","Senegal":"CAF","Ivory Coast":"CAF","Tunisia":"CAF",
    "Egypt":"CAF","South Africa":"CAF","Algeria":"CAF","Ghana":"CAF",
    "DR Congo":"CAF","Cape Verde":"CAF",

    #AFC
    "Japan":"AFC","Australia":"AFC","South Korea":"AFC","Saudi Arabia":"AFC",
    "Iran":"AFC","Iraq":"AFC","Qatar":"AFC","Uzbekistan":"AFC","Jordan":"AFC",

    #CONCACAF
    "Mexico":"CONCACAF","United States":"CONCACAF","Canada":"CONCACAF",
    "Haiti":"CONCACAF","Curaçao":"CONCACAF","Panama":"CONCACAF",

    #OFC
    "New Zealand":"OFC"
}

GROUPS = {
    "A":["Mexico","South Africa","South Korea","Czech Republic"],
    "B":["Canada","Bosnia and Herzegovina","Qatar","Switzerland"],
    "C":["Brazil","Morocco","Haiti","Scotland"],
    "D":["United States","Paraguay","Australia","Turkey"],
    "E":["Germany","Curaçao","Ivory Coast","Ecuador"],
    "F":["Netherlands","Japan","Sweden","Tunisia"],
    "G":["Belgium","Egypt","Iran","New Zealand"],
    "H":["Spain","Cape Verde","Saudi Arabia","Uruguay"],
    "I":["France","Senegal","Iraq","Norway"],
    "J":["Argentina","Algeria","Austria","Jordan"],
    "K":["Portugal","DR Congo","Uzbekistan","Colombia"],
    "L":["England","Croatia","Ghana","Panama"],
}
TEAM_TO_GROUP = {t:g for g,ts in GROUPS.items() for t in ts}

R32_BRACKET = {
    73:("2A","2B"),74:("1E","3rd:A,B,C,D,F"),75:("1F","2C"),76:("1C","2F"),
    77:("1I","3rd:C,D,F,G,H"),78:("2E","2I"),79:("1A","3rd:C,E,F,H,I"),
    80:("1L","3rd:E,H,I,J,K"),81:("1D","3rd:B,E,F,I,J"),82:("1G","3rd:A,E,H,I,J"),
    83:("2K","2L"),84:("1H","2J"),85:("1B","3rd:E,F,G,I,J"),86:("1J","2H"),
    87:("1K","3rd:D,E,I,J,L"),88:("2D","2G"),
}
R16={89:(74,77),90:(73,75),91:(76,78),92:(79,80),93:(83,84),94:(81,82),95:(86,88),96:(85,87)}
QF={97:(89,90),98:(93,94),99:(91,92),100:(95,96)}
SF={101:(97,99),102:(98,100)}
FINAL={104:(101,102)}

CORE_FEATURES = [
    "elo_diff","t1_elo","t2_elo",
    "t1_unbeaten_run","t2_unbeaten_run","diff_unbeaten_run",
    "t1_winning_streak","t2_winning_streak","diff_winning_streak",
    "t1_losing_streak","t2_losing_streak",
    "t1_form_trajectory","t2_form_trajectory",
    "t1_days_since_comp_win","t2_days_since_comp_win",
    "h2h_t1_win_pct","h2h_draw_pct","h2h_avg_gd","h2h_recent_win_pct","h2h_total",
    "is_world_cup","is_wc_qualifier","is_continental","is_friendly",
    "t1_avg_gf_10","t1_avg_ga_10","t1_avg_gd_10","t1_avg_gf_20","t1_avg_ga_20",
    "t2_avg_gf_10","t2_avg_ga_10","t2_avg_gd_10","t2_avg_gf_20","t2_avg_ga_20",
    "t1_win_rate_10","t2_win_rate_10","t1_win_rate_10_comp","t2_win_rate_10_comp",
    "t1_draw_rate_10","t2_draw_rate_10","t1_clean_sheet_rate","t2_clean_sheet_rate",
    "diff_avg_gf_10","diff_avg_ga_10","diff_avg_gd_10","diff_win_rate_10","diff_win_rate_10_comp",
    "t1_is_host","t2_is_host",
]
SQUAD_SUPPLEMENT = [
    "t1_squad_avg_rating","t2_squad_avg_rating","t1_squad_total_goals","t2_squad_total_goals",
    "t1_qualifying_ppg","t2_qualifying_ppg","diff_squad_avg_rating","diff_squad_total_goals",
    "diff_qualifying_ppg",
]
ALL_FEATURES = CORE_FEATURES + SQUAD_SUPPLEMENT

def resolve(df,fl): return [f for f in fl if f in df.columns]

def add_host_features(df, is_wc=False):
    df=df.copy()
    df["t1_is_host"]=df["team_1"].isin(HOSTS).astype(int)
    df["t2_is_host"]=df["team_2"].isin(HOSTS).astype(int)
    if is_wc:
        df.loc[df["team_1"].isin(HOSTS),"t1_elo"]+=HOST_ELO_BOOST
        df.loc[df["team_2"].isin(HOSTS),"t2_elo"]+=HOST_ELO_BOOST
        for team,boost in PEDIGREE_BOOST.items():
            df.loc[df["team_1"]==team,"t1_elo"]+=boost
            df.loc[df["team_2"]==team,"t2_elo"]+=boost
        for team in df["team_1"].unique():
            disc=CONF_DISCOUNT.get(TEAM_CONF.get(team,"UEFA"),0)
            if disc: df.loc[df["team_1"]==team,"t1_elo"]+=disc
        for team in df["team_2"].unique():
            disc=CONF_DISCOUNT.get(TEAM_CONF.get(team,"UEFA"),0)
            if disc: df.loc[df["team_2"]==team,"t2_elo"]+=disc
        df["elo_diff"]=df["t1_elo"]-df["t2_elo"]
    return df

def compute_sample_weights(df):
    return np.where(df.get("is_friendly",pd.Series(0,index=df.index))==1,FRIENDLY_WEIGHT,1.0)

def augment_swap(df):
    mirror=df.copy()
    mirror["team_1"],mirror["team_2"]=df["team_2"].values,df["team_1"].values
    t1c=[c for c in df.columns if c.startswith("t1_")]
    for a,b in [(a,a.replace("t1_","t2_")) for a in t1c if a.replace("t1_","t2_") in df.columns]:
        mirror[a],mirror[b]=df[b].values,df[a].values
    for c in [c for c in df.columns if c.startswith("diff_")]: mirror[c]=-df[c].values
    if "elo_diff" in df.columns: mirror["elo_diff"]=-df["elo_diff"].values
    if "h2h_t1_win_pct" in df.columns:
        mirror["h2h_t1_win_pct"]=1-df["h2h_t1_win_pct"].values-df.get("h2h_draw_pct",0).values
    if "h2h_avg_gd" in df.columns: mirror["h2h_avg_gd"]=-df["h2h_avg_gd"].values
    if "h2h_recent_win_pct" in df.columns: mirror["h2h_recent_win_pct"]=1-df["h2h_recent_win_pct"].values
    if "target_result" in df.columns: mirror["target_result"]=df["target_result"].map({0:2,1:1,2:0}).values
    if "target_t1_goals" in df.columns:
        mirror["target_t1_goals"],mirror["target_t2_goals"]=df["target_t2_goals"].values,df["target_t1_goals"].values
    if "target_gd" in df.columns: mirror["target_gd"]=-df["target_gd"].values
    if "sample_weight" in df.columns: mirror["sample_weight"]=df["sample_weight"].values
    return pd.concat([df,mirror]).sort_values("date").reset_index(drop=True)

# Models 

def _tau(hg,ag,lh,la,rho):
    if hg==0 and ag==0: return 1-lh*la*rho
    if hg==0 and ag==1: return 1+lh*rho
    if hg==1 and ag==0: return 1+la*rho
    if hg==1 and ag==1: return 1-rho
    return 1.0

def _dc_nll(params,teams,hi,ai,hg,ag,w):
    n=len(teams);att,dfn=params[:n],params[n:2*n];gamma,rho=params[2*n],params[2*n+1]
    lh=np.exp(att[hi]-dfn[ai]+gamma);la=np.exp(att[ai]-dfn[hi])
    lp_h=poisson.logpmf(hg,lh);lp_a=poisson.logpmf(ag,la)
    tau=np.array([_tau(int(hg[i]),int(ag[i]),lh[i],la[i],rho) for i in range(len(hg))])
    tau=np.clip(tau,1e-10,None)
    return -(w*(lp_h+lp_a+np.log(tau))).sum()

class DixonColes:
    def __init__(s,hl=365): s.hl=hl
    def fit(s,df):
        s.teams=sorted(set(df["team_1"])|set(df["team_2"]))
        t2i={t:i for i,t in enumerate(s.teams)};n=len(s.teams)
        hi=df["team_1"].map(t2i).values;ai=df["team_2"].map(t2i).values
        hg=df["target_t1_goals"].values.astype(float);ag=df["target_t2_goals"].values.astype(float)
        d=(df["date"].max()-df["date"]).dt.days.values.astype(float)
        fw=df.get("sample_weight",pd.Series(1.0,index=df.index)).values
        w=np.exp(-np.log(2)/s.hl*d)*fw
        x0=np.zeros(2*n+2);x0[2*n]=0.25;x0[2*n+1]=-0.05
        res=minimize(_dc_nll,x0,args=(s.teams,hi,ai,hg,ag,w),method="SLSQP",
                     constraints=[{"type":"eq","fun":lambda p,n=n:p[:n].sum()}],
                     options={"maxiter":500,"ftol":1e-6})
        s.att={t:res.x[i] for i,t in enumerate(s.teams)}
        s.dfn={t:res.x[n+i] for i,t in enumerate(s.teams)}
        s.gamma=res.x[2*n];s.rho=res.x[2*n+1]
        return s
    def predict_one(s,t1,t2):
        lh=np.exp(s.att.get(t1,0)-s.dfn.get(t2,0))
        la=np.exp(s.att.get(t2,0)-s.dfn.get(t1,0))
        m=11;mat=np.zeros((m,m))
        for i in range(m):
            for j in range(m):
                mat[i,j]=poisson.pmf(i,lh)*poisson.pmf(j,la)*_tau(i,j,lh,la,s.rho)
        mat/=mat.sum()
        pw=sum(mat[i,j] for i in range(m) for j in range(m) if i>j)
        pd_=sum(mat[i,i] for i in range(m))
        pl=sum(mat[i,j] for i in range(m) for j in range(m) if i<j)
        return np.array([pw,pd_,pl])
    def predict_batch(s,df):
        out=[]
        for _,r in df.iterrows():
            try: out.append(s.predict_one(r["team_1"],r["team_2"]))
            except: out.append(np.array([.33,.34,.33]))
        return np.array(out)
    def get_rates(s,t1,t2):
        adj1=PEDIGREE_BOOST.get(t1,0)+CONF_DISCOUNT.get(TEAM_CONF.get(t1,"UEFA"),0)
        adj2=PEDIGREE_BOOST.get(t2,0)+CONF_DISCOUNT.get(TEAM_CONF.get(t2,"UEFA"),0)
        lh=np.exp(s.att.get(t1,0)-s.dfn.get(t2,0))*(1+adj1/1000)
        la=np.exp(s.att.get(t2,0)-s.dfn.get(t1,0))*(1+adj2/1000)
        if t1 in HOSTS: lh*=1.15
        if t2 in HOSTS: la*=1.15
        return max(lh,0.15),max(la,0.15)

class XGBModel:
    def __init__(s,features): s.features=features
    def fit(s,tr,val):
        s.f_=resolve(tr,s.features)
        w_tr=tr["sample_weight"].values if "sample_weight" in tr.columns else None
        w_val=val["sample_weight"].values if "sample_weight" in val.columns else None
        dt=xgb.DMatrix(tr[s.f_],label=tr["target_result"],weight=w_tr,feature_names=s.f_)
        dv=xgb.DMatrix(val[s.f_],label=val["target_result"],weight=w_val,feature_names=s.f_)
        p={"objective":"multi:softprob","num_class":3,"eval_metric":"mlogloss",
           "max_depth":5,"learning_rate":0.05,"subsample":0.8,
           "colsample_bytree":0.7,"min_child_weight":10,"seed":42,"verbosity":0}
        s.model=xgb.train(p,dt,num_boost_round=500,evals=[(dv,"val")],
                          early_stopping_rounds=50,verbose_eval=False)
        return s
    def predict(s,df): return s.model.predict(xgb.DMatrix(df[s.f_],feature_names=s.f_))

class LGBModel:
    def __init__(s,features): s.features=features
    def fit(s,tr,val):
        s.f_=resolve(tr,s.features)
        w_tr=tr["sample_weight"].values if "sample_weight" in tr.columns else None
        w_val=val["sample_weight"].values if "sample_weight" in val.columns else None
        dt=lgb.Dataset(tr[s.f_].values,label=tr["target_result"].values,weight=w_tr,
                       feature_name=s.f_,free_raw_data=False)
        dv=lgb.Dataset(val[s.f_].values,label=val["target_result"].values,weight=w_val,
                       feature_name=s.f_,reference=dt,free_raw_data=False)
        p={"objective":"multiclass","num_class":3,"metric":"multi_logloss",
           "num_leaves":31,"learning_rate":0.05,"subsample":0.8,
           "colsample_bytree":0.7,"min_child_samples":25,"seed":42,"verbose":-1}
        s.model=lgb.train(p,dt,num_boost_round=500,valid_sets=[dv],
                          callbacks=[lgb.early_stopping(50,verbose=False),lgb.log_evaluation(0)])
        return s
    def predict(s,df): return s.model.predict(df[s.f_].values)

class MLPModel:
    def __init__(s,features): s.features=features
    def fit(s,tr,val=None):
        s.f_=resolve(tr,s.features)
        X=np.nan_to_num(tr[s.f_].values.astype(np.float32))
        s.scaler=StandardScaler().fit(X)
        w=tr["sample_weight"].values if "sample_weight" in tr.columns else np.ones(len(tr))
        comp=w>FRIENDLY_WEIGHT
        X_full=np.vstack([X,X[comp]]);y_full=np.concatenate([tr["target_result"].values,tr["target_result"].values[comp]])
        s.model=MLPClassifier(hidden_layer_sizes=(128,64,32),max_iter=500,
                              early_stopping=True,validation_fraction=0.15,random_state=42,verbose=False)
        s.model.fit(s.scaler.transform(X_full),y_full)
        return s
    def predict(s,df):
        X=np.nan_to_num(df[s.f_].values.astype(np.float32))
        return s.model.predict_proba(s.scaler.transform(X))

# Feature row builder for knockout matchups 

def build_ko_feature_rows(team_snap, pairs):
    snap={r["team"]:r.drop("team").to_dict() for _,r in team_snap.iterrows()}
    rows=[]
    for t1,t2 in pairs:
        s1,s2=snap.get(t1,{}),snap.get(t2,{})
        row={"team_1":t1,"team_2":t2}
        for col,val in s1.items(): row[f"t1_{col}"]=val
        for col,val in s2.items(): row[f"t2_{col}"]=val
        e1=s1.get("elo",1500)+(HOST_ELO_BOOST if t1 in HOSTS else 0)+PEDIGREE_BOOST.get(t1,0)+CONF_DISCOUNT.get(TEAM_CONF.get(t1,"UEFA"),0)
        e2=s2.get("elo",1500)+(HOST_ELO_BOOST if t2 in HOSTS else 0)+PEDIGREE_BOOST.get(t2,0)+CONF_DISCOUNT.get(TEAM_CONF.get(t2,"UEFA"),0)
        row["t1_elo"]=e1;row["t2_elo"]=e2;row["elo_diff"]=e1-e2
        for col in ["avg_gf_10","avg_ga_10","avg_gd_10","win_rate_10","win_rate_10_comp",
                     "squad_avg_rating","squad_total_goals","qualifying_ppg","unbeaten_run","winning_streak"]:
            v1=s1.get(col,0);v2=s2.get(col,0)
            v1=0 if pd.isna(v1) else v1;v2=0 if pd.isna(v2) else v2
            row[f"diff_{col}"]=v1-v2
        row.update({"h2h_t1_win_pct":0.5,"h2h_draw_pct":0,"h2h_avg_gd":0,"h2h_recent_win_pct":0.5,"h2h_total":0,
                     "is_world_cup":1,"is_wc_qualifier":0,"is_continental":0,"is_friendly":0,
                     "t1_is_host":int(t1 in HOSTS),"t2_is_host":int(t2 in HOSTS)})
        rows.append(row)
    return pd.DataFrame(rows)

# Simulation functions 

def lookup(precomputed,t1,t2):
    pc=precomputed.get((t1,t2))
    if pc is not None: return pc
    pc=precomputed.get((t2,t1))
    if pc is not None: return np.array([pc[2],pc[1],pc[0]])
    return np.array([.33,.34,.33])

def sample_group(pre,dc,t1,t2):
    p=lookup(pre,t1,t2);p=p/p.sum()
    outcome=np.random.choice(["t1","draw","t2"],p=p)
    lh,la=dc.get_rates(t1,t2)
    for _ in range(100):
        g1,g2=np.random.poisson(lh),np.random.poisson(la)
        if outcome=="t1" and g1>g2: return g1,g2
        if outcome=="draw" and g1==g2: return g1,g2
        if outcome=="t2" and g2>g1: return g1,g2
    if outcome=="t1": return max(1,int(round(lh))),0
    if outcome=="t2": return 0,max(1,int(round(la)))
    return 1,1

def sample_knockout(pre,dc,t1,t2):
    p=lookup(pre,t1,t2)
    p1=p[0]+p[1]*0.5;p2=p[2]+p[1]*0.5;tot=p1+p2;p1/=tot;p2/=tot
    lh,la=dc.get_rates(t1,t2)
    g1,g2=np.random.poisson(lh),np.random.poisson(la)
    if g1!=g2: return t1 if g1>g2 else t2
    et1,et2=np.random.poisson(lh/3),np.random.poisson(la/3)
    if et1!=et2: return t1 if g1+et1>g2+et2 else t2
    return t1 if np.random.random()<np.clip(p1,0.35,0.65) else t2

def simulate(pre,dc,elo_map):
    gs={};thirds=[]
    for g in sorted(GROUPS.keys()):
        teams=GROUPS[g]
        st={t:{"team":t,"Pts":0,"GF":0,"GA":0,"GD":0} for t in teams}
        for i in range(4):
            for j in range(i+1,4):
                t1,t2=teams[i],teams[j]
                g1,g2=sample_group(pre,dc,t1,t2)
                st[t1]["GF"]+=g1;st[t1]["GA"]+=g2;st[t2]["GF"]+=g2;st[t2]["GA"]+=g1
                if g1>g2: st[t1]["Pts"]+=3
                elif g1==g2: st[t1]["Pts"]+=1;st[t2]["Pts"]+=1
                else: st[t2]["Pts"]+=3
        for t in teams: st[t]["GD"]=st[t]["GF"]-st[t]["GA"]
        stds=sorted(st.values(),key=lambda x:(-x["Pts"],-x["GD"],-x["GF"],-elo_map.get(x["team"],1500)))
        gs[g]=stds;th=stds[2].copy();th["group"]=g;thirds.append(th)
    w={g:gs[g][0]["team"] for g in GROUPS};r={g:gs[g][1]["team"] for g in GROUPS}
    ts=sorted(thirds,key=lambda x:(-x["Pts"],-x["GD"],-x["GF"],-elo_map.get(x["team"],1500)))
    qt={t["group"]:t["team"] for t in ts[:8]}
    stage=defaultdict(lambda:"Group Stage")
    for g in GROUPS: stage[w[g]]="R32";stage[r[g]]="R32"
    for t in ts[:8]: stage[t["team"]]="R32"
    avail=dict(qt)
    def res_t(src):
        if src[0]=="1": return w.get(src[1:])
        if src[0]=="2": return r.get(src[1:])
    def get_3(pool):
        for g in pool.split(","):
            if g in avail: return avail.pop(g)
        if avail: g,t=next(iter(avail.items()));del avail[g];return t
    ko={}
    for m,s1,s2 in sorted([(m,a,b) for m,(a,b) in R32_BRACKET.items() if "3rd:" in b]):
        t1=res_t(s1);t2=get_3(s2.replace("3rd:",""))
        if t1 and t2: ko[m]=sample_knockout(pre,dc,t1,t2)
    for m,(s1,s2) in R32_BRACKET.items():
        if m in ko: continue
        t1,t2=res_t(s1),res_t(s2)
        if t1 and t2: ko[m]=sample_knockout(pre,dc,t1,t2)
    for m in ko: stage[ko[m]]="R16"
    for m,(m1,m2) in R16.items():
        t1,t2=ko.get(m1),ko.get(m2)
        if t1 and t2: ko[m]=sample_knockout(pre,dc,t1,t2)
    for m in [89,90,91,92,93,94,95,96]:
        if m in ko: stage[ko[m]]="QF"
    for m,(m1,m2) in QF.items():
        t1,t2=ko.get(m1),ko.get(m2)
        if t1 and t2: ko[m]=sample_knockout(pre,dc,t1,t2)
    for m in [97,98,99,100]:
        if m in ko: stage[ko[m]]="SF"
    for m,(m1,m2) in SF.items():
        t1,t2=ko.get(m1),ko.get(m2)
        if t1 and t2: ko[m]=sample_knockout(pre,dc,t1,t2)
    for m in [101,102]:
        if m in ko: stage[ko[m]]="Final"
    for m,(m1,m2) in FINAL.items():
        t1,t2=ko.get(m1),ko.get(m2)
        if t1 and t2: ko[m]=sample_knockout(pre,dc,t1,t2)
    if 104 in ko: stage[ko[104]]="Champion"
    return ko.get(104),dict(stage)


# MAIN


print("="*70)
print("  Phase 3 Full Ensemble (DC+XGB+LGB+MLP) for ALL matches")
print(f"  {N_SIMS:,} simulations | Pedigree + Conf discount")
print("="*70)

tr=pd.read_csv(DATA_DIR/"training_features.csv");ts=pd.read_csv(DATA_DIR/"team_snapshot.csv")
tr["date"]=pd.to_datetime(tr["date"])
tr=tr.sort_values("date").reset_index(drop=True)

elo_map=dict(zip(ts["team"],ts["elo"]))
for t in elo_map:
    elo_map[t]+=CONF_DISCOUNT.get(TEAM_CONF.get(t,"UEFA"),0)+PEDIGREE_BOOST.get(t,0)
for t in HOSTS:
    if t in elo_map: elo_map[t]+=HOST_ELO_BOOST

tr=add_host_features(tr,is_wc=False);tr["sample_weight"]=compute_sample_weights(tr)
n=int(len(tr)*0.80)
train_raw,val_raw=tr.iloc[:n].copy(),tr.iloc[n:].copy()
train_aug=augment_swap(train_raw);val_aug=augment_swap(val_raw)

print("\n▸ Training 4 base models …")
dc=DixonColes(365).fit(train_raw);print("  Dixon-Coles done")
xm=XGBModel(CORE_FEATURES).fit(train_aug,val_aug);print("  XGBoost done")
lm=LGBModel(CORE_FEATURES).fit(train_aug,val_aug);print("  LightGBM done")
mm=MLPModel(ALL_FEATURES).fit(train_aug);print("  MLP done")

y_val=val_raw["target_result"].values
pv={"dc":dc.predict_batch(val_raw),"xgb":xm.predict(val_raw),"lgb":lm.predict(val_raw),"mlp":mm.predict(val_raw)}
weights={}
for name,p in pv.items():
    brier=np.mean(np.sum((p-np.eye(3)[y_val])**2,axis=1))
    weights[name]=1.0/(brier+1e-8)
tw=sum(weights.values());weights={k:v/tw for k,v in weights.items()}
print(f"\n▸ Ensemble weights: { {k:f'{v:.3f}' for k,v in sorted(weights.items(),key=lambda x:-x[1])} }")

# Pre-compute ALL 1128 pairwise ensemble predictions
print("\n▸ Pre-computing ensemble predictions for all 1,128 team pairings …")
all_t=sorted(set(t for g in GROUPS.values() for t in g))
precomputed={}

# Build feature rows for ALL pairs from team_snapshot (works regardless of WC CSV columns)
all_pairs=[(t1,t2) for i,t1 in enumerate(all_t) for t2 in all_t[i+1:]]
all_feat_df=build_ko_feature_rows(ts,all_pairs)

all_dc=np.array([dc.predict_one(t1,t2) if t1 in dc.att and t2 in dc.att
                 else np.array([.33,.34,.33]) for t1,t2 in all_pairs])
all_xgb=xm.predict(all_feat_df);all_lgb=lm.predict(all_feat_df);all_mlp=mm.predict(all_feat_df)

for j,(t1,t2) in enumerate(all_pairs):
    b=weights["dc"]*all_dc[j]+weights["xgb"]*all_xgb[j]+weights["lgb"]*all_lgb[j]+weights["mlp"]*all_mlp[j]
    precomputed[(t1,t2)]=b/b.sum()

print(f"  {len(precomputed)} total pairwise predictions cached")

# Adjusted Elo display
print(f"\n▸ Adjusted Elo (top 15):")
for t,e in sorted(elo_map.items(),key=lambda x:-x[1])[:15]:
    tags=[]
    if t in HOSTS: tags.append(f"host +{HOST_ELO_BOOST}")
    if t in PEDIGREE_BOOST: tags.append(f"ped +{PEDIGREE_BOOST[t]}")
    c=TEAM_CONF.get(t,"UEFA");d=CONF_DISCOUNT.get(c,0)
    if d: tags.append(f"{c} {d}")
    tag=f"  ({', '.join(tags)})" if tags else ""
    print(f"  {t:<22s} {e:>7.0f}{tag}")

# Run simulations
print(f"\n▸ Running {N_SIMS:,} simulations …")
champ=Counter();sc=defaultdict(Counter)
for sim in range(N_SIMS):
    if (sim+1)%10000==0: print(f"  … {sim+1:,} / {N_SIMS:,}")
    champion,stages=simulate(precomputed,dc,elo_map)
    if champion: champ[champion]+=1
    for t,s in stages.items(): sc[t][s]+=1

# Results
print(f"\n{'='*70}")
print(f"  WORLD CUP 2026 WIN PROBABILITIES ({N_SIMS:,} sims)")
print(f"  Ensemble: DC({weights['dc']:.0%}) + XGB({weights['xgb']:.0%}) + LGB({weights['lgb']:.0%}) + MLP({weights['mlp']:.0%})")
print(f"{'='*70}")

so=["Champion","Final","SF","QF","R16","R32"]
rows=[]
for t in all_t:
    row={"Team":t,"Group":TEAM_TO_GROUP[t],"Elo":elo_map.get(t,1500)}
    for s in so:
        idx=so.index(s);row[f"P({s})"]=sum(sc[t][x] for x in so[:idx+1])/N_SIMS
    row["P(Exit Group)"]=1-row["P(R32)"]
    rows.append(row)

results=pd.DataFrame(rows).sort_values("P(Champion)",ascending=False).reset_index(drop=True)

print(f"\n  {'#':<3} {'Team':<25} {'Grp':>3} {'Elo':>5}  {'Win':>6} {'Final':>6} {'SF':>6} {'QF':>6} {'R16':>6} {'R32':>6} {'Exit':>6}")
print("  "+"─"*92)
for i,r in results.iterrows():
    print(f"  {i+1:<3} {r['Team']:<25} {r['Group']:>3} {r['Elo']:>5.0f}  "
          f"{r['P(Champion)']:>5.1%} {r['P(Final)']:>5.1%} {r['P(SF)']:>5.1%} "
          f"{r['P(QF)']:>5.1%} {r['P(R16)']:>5.1%} {r['P(R32)']:>5.1%} "
          f"{r['P(Exit Group)']:>5.1%}")

print(f"\n▸ Top 10 with 95% CI:")
for i,r in results.head(10).iterrows():
    p=r["P(Champion)"];se=np.sqrt(p*(1-p)/N_SIMS)
    print(f"  {r['Team']:<25s} {p:>5.1%}  [{max(0,p-1.96*se):.1%} – {min(1,p+1.96*se):.1%}]")

results.to_csv(OUT_DIR/"try_probs.csv",index=False,float_format="%.4f")
print(f"\n Saved → {OUT_DIR/'try_probs.csv'}")

