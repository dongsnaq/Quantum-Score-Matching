"""IBM four-qubit TFIM scores, randomized estimators, and parameter updates."""
from __future__ import annotations
from functools import lru_cache
import numpy as np
from scipy.integrate import quad, cumulative_trapezoid
from scipy.special import zeta
from qiskit.quantum_info import Pauli

N = 4
C_U = 7.0 * zeta(3.0, 1.0) / np.pi**3

def word(sites):
    text = ["I"] * N
    for i, letter in sites.items():
        text[N-1-i] = letter
    return "".join(text)

TERMS = [word({i: 'Z', i+1: 'Z'}) for i in range(3)] + [word({i: 'X'}) for i in range(4)]
FRAME = [word({i: c}) for i in range(4) for c in ('X', 'Z')]
MATS = np.array([Pauli(p).to_matrix() for p in TERMS])
FMATS = np.array([Pauli(p).to_matrix() for p in FRAME])
GENERATORS = np.array([-MATS[:3].sum(0), -MATS[3:].sum(0)])
TARGET = np.array([1., 1.5])

def absolute_scaled_time_density(x: float) -> float:
    if x == 0.0:
        return np.inf
    return float(-(4.0 / np.pi) * np.log(np.tanh(0.5 * np.pi * x)))


@lru_cache(maxsize=512)
def scaled_tail_moments(x: float) -> tuple[float, float, float]:
    probability = quad(absolute_scaled_time_density, x, np.inf, epsabs=1e-12, epsrel=1e-10)[0]
    first = quad(lambda y: y * absolute_scaled_time_density(y), x, np.inf, epsabs=1e-12, epsrel=1e-10)[0]
    second = quad(lambda y: y * y * absolute_scaled_time_density(y), x, np.inf, epsabs=1e-12, epsrel=1e-10)[0]
    return float(probability), float(first), float(second)


def _coordinate_insertion_masses(q: np.ndarray, insertion_masses: np.ndarray | None) -> np.ndarray:
    if insertion_masses is None:
        return np.ones(q.shape[1], dtype=float)
    masses = np.asarray(insertion_masses, dtype=float)
    if masses.shape != (q.shape[1],):
        raise ValueError("insertion masses must have one entry per parameter coordinate")
    if np.any(masses < 0.0):
        raise ValueError("insertion masses must be nonnegative")
    return masses


def tail_bias_bound(
    beta: float,
    scaled_cutoff: float,
    b: np.ndarray,
    q: np.ndarray,
    insertion_masses: np.ndarray | None = None,
) -> np.ndarray:
    coordinate_masses = _coordinate_insertion_masses(q, insertion_masses)
    p_tail, scaled_first_tail, _ = scaled_tail_moments(scaled_cutoff)
    first_tail = beta * scaled_first_tail
    score_tail = beta * b * p_tail
    response_tail = beta * q * p_tail + 2.0 * beta * b[:, None] * first_tail * coordinate_masses[None, :]
    full_response = beta * q + 2.0 * C_U * beta**2 * b[:, None] * coordinate_masses[None, :]
    retained_score_plus_frame = beta * b * (1.0 - p_tail) + 2.0
    return np.sum(score_tail[:, None] * full_response + retained_score_plus_frame[:, None] * response_tail, axis=0)


def uniform_cutoff(
    beta: float,
    target_bias: float,
    b: np.ndarray,
    q: np.ndarray,
    insertion_masses: np.ndarray | None = None,
) -> float:
    lower, upper = 1.0, 2.0
    while float(np.max(tail_bias_bound(beta, upper, b, q, insertion_masses))) > target_bias:
        lower, upper = upper, 2.0 * upper
        if upper > 256.0:
            raise RuntimeError("failed to certify a finite time cutoff")
    for _ in range(55):
        midpoint = 0.5 * (lower + upper)
        if float(np.max(tail_bias_bound(beta, midpoint, b, q, insertion_masses))) <= target_bias:
            upper = midpoint
        else:
            lower = midpoint
    return beta * upper

def commutator_terms(a, theta=None, coordinate=None):
    ans=[]
    for k,p in enumerate(TERMS):
        if coordinate is not None and (k>=3)!=bool(coordinate): continue
        if Pauli(FRAME[a]).commutes(Pauli(p)): continue
        product=Pauli(FRAME[a]).dot(Pauli(p))
        coefficient=float(np.real(2j*(-1j)**product.phase))
        label=Pauli((product.z,product.x)).to_label()
        if theta is not None: coefficient*=theta[int(k>=3)]
        ans.append((coefficient,label))
    return ans


def loss(theta,beta,rho):
    energies,vectors=np.linalg.eigh(np.einsum('j,jab->ab',theta,GENERATORS))
    a=vectors.conj().T@FMATS@vectors
    score=-2j*np.tanh(beta*(energies[:,None]-energies[None,:])/2)*a
    observable=np.sum(.5*score@score-1j*(a@score-score@a),axis=0)
    return float(np.trace((vectors.conj().T@rho@vectors)@observable).real)


def gradient(theta,beta,rho):
    eye=np.eye(2)*1e-5
    return np.array([(loss(theta+e,beta,rho)-loss(theta-e,beta,rho))/2e-5 for e in eye])


def masses(theta,beta,cutoff):
    b=np.array([sum(abs(c) for c,_ in commutator_terms(a,theta)) for a in range(8)])
    q=np.array([[sum(abs(c) for c,_ in commutator_terms(a,coordinate=j)) for j in range(2)] for a in range(8)])
    tail,first,_=scaled_tail_moments(cutoff/beta)
    prob=1-tail; moment=beta*(C_U-first)
    sm=beta*b*prob
    t1=beta*q*prob
    t2=2*beta*b[:,None]*moment*np.array([3,4])[None,:]
    return sm,t1,t2


def time_tables(beta,cutoff):
    x=np.r_[0.,np.geomspace(1e-12,cutoff,32768)]
    density=np.zeros_like(x)
    density[1:]=-4/(np.pi*beta)*np.log(np.tanh(np.pi*x[1:]/(2*beta)))
    density[0]=density[1]
    tables=[]
    for power in (0,1):
        cumulative=np.r_[0.,cumulative_trapezoid(density*x**power,x)]
        tables.append((x,cumulative/cumulative[-1]))
    return tables


def choose(items,rng):
    weights=np.array([abs(c) for c,_ in items]); weights/=weights.sum()
    c,label=items[rng.choice(len(items),p=weights)]
    return int(np.sign(c)),label


def draw(theta,beta,cutoff,j,count,rng):
    sm,t1,t2=masses(theta,beta,cutoff)
    aw=(sm+2)*(t1[:,j]+t2[:,j]); total=aw.sum()
    tables=time_tables(beta,cutoff)
    def time_sample(tilt):
        grid,cdf=tables[tilt]
        return float(np.interp(rng.random(),cdf,grid)*rng.choice([-1,1]))
    specs=[]
    for _ in range(count):
        a=int(rng.choice(8,p=aw/total))
        product=bool(rng.random()<sm[a]/(sm[a]+2))
        second=bool(rng.random()<t2[a,j]/(t1[a,j]+t2[a,j]))
        if second:
            sign,q=choose(commutator_terms(a,theta),rng)
            xi=time_sample(1); eta=int(rng.choice([-1,1])); s=float(rng.random())
            insertion=TERMS[int(rng.integers(0,3) if j==0 else rng.integers(3,7))]
                                                           
            sign*=int(np.sign(xi))*eta
        else:
            sign,q=choose(commutator_terms(a,coordinate=j),rng)
            sign*=-1; xi=time_sample(0); eta=1; s=None; insertion=None
        if product:
            ss,qs=choose(commutator_terms(a,theta),rng)
            sign*=-ss; xs=time_sample(0)
        else: qs=FRAME[a]; xs=None
        specs.append(dict(a=a,product=product,sign=sign,q=q,xi=xi,s=s,insertion=insertion,eta=eta,qs=qs,xs=xs))
    return total,specs


@lru_cache(maxsize=32)
def eigensystem(theta):
    return np.linalg.eigh(np.einsum('j,jab->ab',theta,GENERATORS))


def exact_branch(spec,theta,rho):
    energies,vectors=eigensystem(tuple(theta))
    def evolve(t): return (vectors*np.exp(1j*t*energies))@vectors.conj().T
    xi=spec['xi']; s=spec['s']
    if s is None: w=evolve(xi)
    else:
        rotation=(np.eye(16)+1j*spec['eta']*Pauli(spec['insertion']).to_matrix())/np.sqrt(2)
        w=evolve((1-s)*xi)@rotation@evolve(s*xi)
    response=w@Pauli(spec['q']).to_matrix()@w.conj().T
    a=Pauli(spec['qs']).to_matrix()
    if spec['product']:
        u=evolve(spec['xs']); a=u@a@u.conj().T
    z=np.trace(rho@a@response)
    return spec['sign']*float(z.real if spec['product'] else z.imag)


def update(theta,g,beta,t):
                                                                     
    step=-(.5/(1+t/10))*g/(beta**2*np.array([24.,16.]))
    norm=np.linalg.norm(step); cap=.05*np.linalg.norm(TARGET)
    if norm>cap: step*=cap/norm
    return np.clip(theta+step,0.,2.)
