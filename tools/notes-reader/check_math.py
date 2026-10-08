"""Numerical checks for the worked examples and identities in these notes."""
import json, math
checks=[]
def check(name, cond, evidence):
    if not cond: raise AssertionError((name,evidence))
    checks.append({'name':name,'passed':True,'evidence':evidence})
def close(a,b,tol=1e-9):return abs(a-b)<tol

def softmax(x):
    m=max(x);e=[math.exp(v-m) for v in x];return [v/sum(e) for v in e]
def dot(a,b):return sum(x*y for x,y in zip(a,b))
H,I,L,V,nq,nkv,d=3584,18944,28,152064,28,4,128
block=2*H*H+2*H*nkv*d+(H+2*nkv*d)+3*H*I+2*H
params=L*block+2*V*H+H
check('Qwen2.5-7B parameter shapes',params==7615616512 and block==233057792,{'block':block,'total':params})
kv=2*1*L*2048*nkv*d*2;kv_mha=2*L*2048*nq*d*2
check('GQA versus MHA KV cache',kv==112*2**20 and kv_mha==784*2**20,{'GQA_MiB':kv/2**20,'MHA_MiB':kv_mha/2**20})
act=L*(34*2048*H+5*nq*2048**2)
check('Teaching activation estimate',act==23429382144,{'bytes':act,'GB':act/1e9,'GiB':act/2**30,'microbatch128_GB':act*128/1e9})
p,n=7.5e9,64
zero=[16*p,4*p+12*p/n,2*p+14*p/n,16*p/n]
check('ZeRO state memory',all(close(a,b) for a,b in zip([v/1e9 for v in zero],[120,31.40625,16.640625,1.875])),{'GB': [v/1e9 for v in zero],'ideal_reduction':zero[0]/zero[-1]})
sec=2*(300e6*4)/(10e9/8)
check('Gb/s conversion',close(sec,1.92),{'large_N_seconds':sec})
days=8*7e9*300e9/(64*312e12*.45)/86400
check('Executed FLOPs time estimate',abs(days-21.64)<.01,{'days':days})
# Compare stable blocked online softmax and its output to an independent direct form.
scores=[1000.,998.,1002.,990.,1001.]; values=[[1.,2.],[-2.,3.],[4.,-1.],[2.,4.],[0.,5.]]
w=softmax(scores);expected=[sum(a*v[j] for a,v in zip(w,values)) for j in range(2)]
m=-math.inf;den=0.;u=[0.,0.]
for start,end in [(0,2),(2,4),(4,5)]:
 mb=max(scores[start:end]); eb=[math.exp(s-mb) for s in scores[start:end]]
 lb=sum(eb);ub=[sum(a*v[j] for a,v in zip(eb,values[start:end])) for j in range(2)]
 new_m=max(m,mb);alpha=math.exp(m-new_m);gamma=math.exp(mb-new_m)
 den=alpha*den+gamma*lb;u=[alpha*a+gamma*b for a,b in zip(u,ub)];m=new_m
actual=[v/den for v in u]
check('FlashAttention blocked normalization',all(close(a,b) for a,b in zip(actual,expected)),{'max_error':max(abs(a-b) for a,b in zip(actual,expected))})
# Check the normalized-output recurrence used by the original algorithm figure.
m=-math.inf;den=0.;out=[0.,0.]
for start,end in [(0,1),(1,4),(4,5)]:
 mb=max(scores[start:end]);eb=[math.exp(s-mb) for s in scores[start:end]];lb=sum(eb)
 block_out=[sum(a*v[j] for a,v in zip(eb,values[start:end]))/lb for j in range(2)]
 new_m=max(m,mb);alpha=math.exp(m-new_m);gamma=math.exp(mb-new_m);new_den=alpha*den+gamma*lb
 out=[(alpha*den*a+gamma*lb*b)/new_den for a,b in zip(out,block_out)];m,den=new_m,new_den
check('FlashAttention normalized output recurrence',all(close(a,b) for a,b in zip(out,expected)),{'max_error':max(abs(a-b) for a,b in zip(out,expected))})
# Compare sharded vocabulary cross entropy to direct softmax and finite differences.
logits=[1000.,1002.,998.,1001.,999.];target_id=3;shards=[(0,2),(2,5)]
global_max=max(max(logits[a:b]) for a,b in shards)
global_sum=sum(sum(math.exp(v-global_max) for v in logits[a:b]) for a,b in shards)
sharded_ce=-logits[target_id]+global_max+math.log(global_sum)
probs=softmax(logits);grad=[p-(j==target_id) for j,p in enumerate(probs)]
def ce(z):return -math.log(softmax(z)[target_id])
fd=[]
for j in range(len(logits)):
 zp=logits.copy();zm=logits.copy();zp[j]+=1e-4;zm[j]-=1e-4;fd.append((ce(zp)-ce(zm))/2e-4)
grad_error=max(abs(a-b) for a,b in zip(grad,fd))
check('Vocabulary TP cross entropy and gradients',close(sharded_ce,ce(logits)) and grad_error<1e-8,{'loss_error':abs(sharded_ce-ce(logits)),'gradient_max_error':grad_error})
# DPO partition identity and reward offset invariance.
ref=[.2,.3,.5];pi=[.4,.1,.5];r=[1.2,-.4,2.3];beta=.7
z=sum(q*math.exp(v/beta) for q,v in zip(ref,r));star=[q*math.exp(v/beta)/z for q,v in zip(ref,r)]
kl=lambda a,b:sum(p*math.log(p/q) for p,q in zip(a,b))
j=dot(pi,r)-beta*kl(pi,ref);rhs=beta*math.log(z)-beta*kl(pi,star)
implicit=[beta*math.log(p/q) for p,q in zip(star,ref)];offsets=[a-b for a,b in zip(r,implicit)]
check('DPO objective identity and offset',close(j,rhs) and max(offsets)-min(offsets)<1e-12,{'identity_error':abs(j-rhs),'reward_offset':offsets[0]})
# RoPE complex expression versus real rotations.
q=[.2,-.8];k=[1.3,.7];a,b=.31,1.26
rotate=lambda v,t:[math.cos(t)*v[0]-math.sin(t)*v[1],math.sin(t)*v[0]+math.cos(t)*v[1]]
left=dot(rotate(q,a),rotate(k,b));right=(complex(*q)*complex(*k).conjugate()*complex(math.cos(a-b),math.sin(a-b))).real
check('RoPE dot identity',close(left,right),{'error':abs(left-right)})
# Independent finite differences of a two-output linear layer with squared-error loss.
x=[.3,-.4,1.2];w=[[.2,.4],[-.7,.5],[.8,-.3]];target=[-.2,.6]
y=[sum(x[i]*w[i][j] for i in range(3)) for j in range(2)];g=[a-b for a,b in zip(y,target)]
analytical=[sum(g[j]*w[i][j] for j in range(2)) for i in range(3)]
def loss(xx):
 yy=[sum(xx[i]*w[i][j] for i in range(3)) for j in range(2)]
 return .5*sum((a-b)**2 for a,b in zip(yy,target))
fd=[]
for i in range(3):
 xp=x.copy();xm=x.copy();xp[i]+=1e-6;xm[i]-=1e-6;fd.append((loss(xp)-loss(xm))/2e-6)
check('Column TP input gradients',max(abs(a-b) for a,b in zip(analytical,fd))<1e-8,{'max_error':max(abs(a-b) for a,b in zip(analytical,fd))})
# The missing common term, and a shared-parameter probability counterexample.
gp,gr=[1.,2.],[3.,-1.];rp,rr,mu,sigma=4.,1.,1.,2.
exact=[((rp-mu)*a+(rr-mu)*b)/sigma for a,b in zip(gp,gr)]
expanded=[(rp-rr)/(2*sigma)*(a-b)+((rp+rr)/2-mu)/sigma*(a+b) for a,b in zip(gp,gr)]
check('Pair-GRPO common term',all(close(a,b) for a,b in zip(exact,expanded)),{'gradient':exact,'pair_difference_only':[(rp-rr)/(2*sigma)*(a-b) for a,b in zip(gp,gr)]})
pair=softmax([-.001,-.002,.003]);logratio=math.log(pair[0]/pair[1])
check('Soft-Pair shared-parameter counterexample',pair[0]<1/3 and pair[1]<1/3 and close(logratio,.001),{'preferred':pair[0],'rejected':pair[1],'logratio_increase':logratio})
# Exact differentiation of KL(softmax(z)||q) also changes irrelevant logits.
current=[.2,.3,.5];target=[.22,.28,.5];D=kl(current,target)
other_grad=current[2]*(math.log(current[2]/target[2])-D)
check('Hard-Pair irrelevant logit is generally nonzero',abs(other_grad)>1e-7,{'irrelevant_logit_gradient':other_grad})
check('Pipeline ideal bubble',all((k-1)/(5*k-1)<.2 for k in range(1,100)),{'K4_M16_bubble':3/19})
print(json.dumps({'checks':checks,'passed':len(checks)},ensure_ascii=False))
