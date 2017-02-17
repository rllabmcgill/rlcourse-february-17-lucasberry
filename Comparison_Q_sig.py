import numpy as np
import matplotlib.pyplot as plt

# Rewards
rewards={'-':-1,'St':-1,'C':-100,'G':0,'P':-1}

# change from state to state
# n=0 R, n=1 L, n=2 U, n=3 D
def take_action(n,S,G):
    Sg=convert_st(S)
    if G[Sg] == 'C':
        new_state = [3,0]
    elif n == 'R':
        if Sg[1]==11:
            new_state = np.array(Sg)
        else:
            new_state = np.array(Sg) + np.array([0,1])
    elif n == 'L':
        if Sg[1]==0:
            new_state = np.array(Sg)
        else:
            new_state = np.array(Sg) + np.array([0,-1])
    elif n == 'U':
        if Sg[0]==0:
            new_state = np.array(Sg)
        else:
            new_state = np.array(Sg) + np.array([-1,0])
    else:
        if Sg[0]==3:
            new_state = np.array(Sg)
        else:
            new_state = np.array(Sg) + np.array([1,0])
    new_state = (3-new_state[0])*12+new_state[1]
    return new_state

# epsilon greedy
# S is current state, e is epsilon
def select_action(S,e,A,Q):
    a=np.random.uniform(0,1)
    W = np.argwhere(Q[S,:] == np.amax(Q[S,:]))
    m = list(range(len(A)))
    if 0< S < 11:
        return 'C'
    elif len(W) == len(A):
        #print('all equal')
        c = np.random.choice(len(W))
        act = W[c][0]
    elif a < (1-e):
        #print('majority')
        c = np.random.choice(len(W))
        act = W[c][0]
    else:
        nm = [ss for ss in m if ss not in list(W[:,0])]
        #print('minority')
        c = np.random.choice(len(nm))
        act = nm[c]
    return A[act]

def random_action(A):
    u=np.random.choice(len(A))
    return u

# Create pi
def policy(Q,e):
    pol = list(range(4))
    W = np.argwhere(Q == np.amax(Q))
    ind = [i for i in pol if i not in list(W[:,0])]
    if ind == []:
         pol = [1/len(pol)]*len(pol)
    else:
        for inde in ind:
            pol[inde]=e/len(ind)
        for i in W:
            pol[i[0]]=(1-e)/len(W)
    return np.array(pol)

# takes state and produces corresponding array in Grid
def convert_st(S):
    col = S % 12
    g = (int(np.absolute((S-col)/12-3)),col)
    return g

def conver_act(A):
    if A == 'C':
        a=0
    elif A == 'R':
        a=0
    elif A == 'L':
        a=1
    elif A == 'U':
        a=2
    else:
        a=3
    return a

# show path creates path on grid
def show_path(S,CG,RG):
    Sg=convert_st(S)
    if Sg[0] == 3 and Sg[1]>0:
        return np.copy(RG)
    else:
        G=np.copy(CG)
        G[Sg]='P'
        return G

def select_sig(t):
    if t%2==1:
        sig=1
    else:
        sig=0
    return sig

# Expected Sarsa
def exp_sarsa(cQ , eps, no, alpa, S, CG, RG , act):
    L_R = []
    L_S = [S]
    Q = np.copy(cQ)
    A=select_action(S,eps,act,Q)
    Ac = conver_act(A)
    L_A =[Ac]
    T = 10**15
    for t in list(range(10000000)):
        #CG = show_path(S,CG,RG)
        #print(CG)
        if t<T:
            S = take_action(A,S,RG)
            L_S.append(S)
            L_R.append(rewards[RG[convert_st(S)]])
            if RG[convert_st(S)] == 'G':
                T = t+1
            else:
                A=select_action(S,eps,act,Q)
                Ac = conver_act(A)
                L_A.append(Ac)
        tau = t - no + 1
        if tau >= 0:
            minc = min(tau+no,T)
            GG = sum(L_R[tau:minc])
            if tau+no<T:
                S_taun = L_S[-1]
                GG = GG + np.dot(policy(Q[S_taun,:],eps), Q[S_taun,:])
            S_tau = L_S[tau]
            A_tau = L_A[tau]
            Q[S_tau,A_tau] += alpa*(GG-Q[S_tau,A_tau])
        if tau == T - 1:
            break
    return Q , sum(L_R)

def tree_backup(cQ , eps, no, alpa, S, CG, RG , act):
    delta=[]
    L_S = [S]
    Q = np.copy(cQ)
    A=select_action(S,eps,act,Q)
    Ac = conver_act(A)
    L_A =[Ac]
    L_Q=[Q[S,Ac]]
    T = 10**15
    pp=policy(Q[S,:],eps)
    L_pi=[pp[Ac]]
    L_R = []
    for t in list(range(10000000)):
        if t<T:
            S = take_action(A,S,RG)
            L_S.append(S)
            R=rewards[RG[convert_st(S)]]
            L_R.append(R)
            if RG[convert_st(S)] == 'G':
                T = t+1
                delta.append(R-L_Q[t])
            else:
                delta.append(R+np.dot(policy(Q[S,:],eps), Q[S,:])-L_Q[t])
                if RG[convert_st(S)] == 'C':
                    A='C'
                else:
                    A=act[random_action(act)]
                A=select_action(S,eps,act,Q)
                Ac = conver_act(A)
                L_A.append(Ac)
                L_Q.append(Q[S,Ac])
                pp=policy(Q[S,:],eps)
                L_pi.append(pp[Ac])
        tau=t-no+1
        if tau >= 0:
            E=1
            G=L_Q[tau]
            for k in range(tau,min(tau+no-1,T-1)):
                G += E*delta[k]
                E = E*L_pi[k+1]
            S_tau = L_S[tau]
            A_tau = L_A[tau]
            Q[S_tau,A_tau] += alpa*(G-Q[S_tau,A_tau])
        if tau == T - 1:
            break
    return Q , sum(L_R)

def q_sig(cQ , eps, no, alpa, S, CG, RG , act):
    delta=[]
    L_S = [S]
    Q = np.copy(cQ)
    A=select_action(S,eps,act,Q)
    Ac = conver_act(A)
    L_A =[Ac]
    L_Q=[Q[S,Ac]]
    T = 10**15
    pp=policy(Q[S,:],eps)
    L_pi=[pp[Ac]]
    L_sig=[1]
    L_R=[]
    for t in list(range(10000000)):
        if t<T:
            S = take_action(A,S,RG)
            L_S.append(S)
            R=rewards[RG[convert_st(S)]]
            L_R.append(R)
            if RG[convert_st(S)] == 'G':
                T = t+1
                delta.append(R-L_Q[t])
            else:
                A=select_action(S,eps,act,Q)
                Ac = conver_act(A)
                L_A.append(Ac)
                sig=select_sig(t)
                L_sig.append(sig)
                L_Q.append(Q[S,Ac])
                delta.append(R+sig*L_Q[t+1]+(1-sig)*np.dot(policy(Q[S,:],eps),Q[S,:])-L_Q[t])
                pp=policy(Q[S,:],eps)
                L_pi.append(pp[Ac])
        tau=t-no+1
        if tau >= 0:
            E=1
            G=L_Q[tau]
            for k in range(tau,min(tau+no-1,T-1)):
                G += E*delta[k]
                E = E*((1-L_sig[k+1])*L_pi[k+1]+L_sig[k+1])
            S_tau = L_S[tau]
            A_tau = L_A[tau]
            Q[S_tau,A_tau] += alpa*(G-Q[S_tau,A_tau])
        if tau == T - 1:
            break
    return Q , sum(L_R)
