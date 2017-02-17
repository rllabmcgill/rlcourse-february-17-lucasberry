from Comparison_Q_sig import *
import time


# N is the number of episodes
# n how far you look forward
# e is epsilon
# S_0 is the start
# M is the number of runs
N = 300
n = 2
e = .1
alpha = .6
S_0 = 0
M = 5

# Cliff Grid World
Grid=np.array([
    ['-','-','-','-','-','-','-','-','-','-','-','-'],
    ['-','-','-','-','-','-','-','-','-','-','-','-'],
    ['-','-','-','-','-','-','-','-','-','-','-','-'],
    ['St','C','C','C','C','C','C','C','C','C','C','G']
])

# S is the number of states
# A is the set of actions
S=np.size(Grid)
actions=['R','L','U','D']

# Rewards
rewards={'-':-1,'St':-1,'C':-100,'G':0,'P':-1}

# Q matrix is SxA
Q=np.zeros((S,len(actions)))

np.random.seed(6)

Qn=np.copy(Q)
Qn1=np.copy(Q)
Qn2=np.copy(Q)
Qn3=np.copy(Q)
Qn4=np.copy(Q)
Qn5=np.copy(Q)
Qn6=np.copy(Q)
Qn7=np.copy(Q)
Qn8=np.copy(Q)
Qn9=np.copy(Q)
Qn10=np.copy(Q)
Qn11=np.copy(Q)
Qn12=np.copy(Q)
Qn13=np.copy(Q)
Qn14=np.copy(Q)
Qn15=np.copy(Q)
Qn16=np.copy(Q)
Qn17=np.copy(Q)
Qn18=np.copy(Q)
Qn19=np.copy(Q)
Qn20=np.copy(Q)
Qn21=np.copy(Q)
Qn22=np.copy(Q)
Qn23=np.copy(Q)
Qn24=np.copy(Q)
Qn25=np.copy(Q)
Qn26=np.copy(Q)
Qn27=np.copy(Q)
Qn28=np.copy(Q)
Qn29=np.copy(Q)
L_exp=[]
L_tree=[]
L_sig=[]
start = time.time()
for i in range(0,N+1,1):
    L=exp_sarsa(Qn , e, n, alpha, S_0, Grid, Grid , actions)
    L1=exp_sarsa(Qn1 , e, n, alpha, S_0, Grid, Grid , actions)
    L2=exp_sarsa(Qn2 , e, n, alpha, S_0, Grid, Grid , actions)
    L3=exp_sarsa(Qn3 , e, n, alpha, S_0, Grid, Grid , actions)
    L4=exp_sarsa(Qn4 , e, n, alpha, S_0, Grid, Grid , actions)
    L5=exp_sarsa(Qn5 , e, n, alpha, S_0, Grid, Grid , actions)
    L6=exp_sarsa(Qn6 , e, n, alpha, S_0, Grid, Grid , actions)
    L7=exp_sarsa(Qn7 , e, n, alpha, S_0, Grid, Grid , actions)
    L8=exp_sarsa(Qn8 , e, n, alpha, S_0, Grid, Grid , actions)
    L9=exp_sarsa(Qn9 , e, n, alpha, S_0, Grid, Grid , actions)
    L10=tree_backup(Qn10 , e, n, alpha, S_0, Grid, Grid , actions)
    L11=tree_backup(Qn11 , e, n, alpha, S_0, Grid, Grid , actions)
    L12=tree_backup(Qn12 , e, n, alpha, S_0, Grid, Grid , actions)
    L13=tree_backup(Qn13 , e, n, alpha, S_0, Grid, Grid , actions)
    L14=tree_backup(Qn14 , e, n, alpha, S_0, Grid, Grid , actions)
    L15=tree_backup(Qn15 , e, n, alpha, S_0, Grid, Grid , actions)
    L16=tree_backup(Qn16 , e, n, alpha, S_0, Grid, Grid , actions)
    L17=tree_backup(Qn17 , e, n, alpha, S_0, Grid, Grid , actions)
    L18=tree_backup(Qn18 , e, n, alpha, S_0, Grid, Grid , actions)
    L19=tree_backup(Qn19 , e, n, alpha, S_0, Grid, Grid , actions)
    L20=q_sig(Qn20 , e, n, alpha, S_0, Grid, Grid , actions)
    L21=q_sig(Qn21 , e, n, alpha, S_0, Grid, Grid , actions)
    L22=q_sig(Qn22 , e, n, alpha, S_0, Grid, Grid , actions)
    L23=q_sig(Qn23 , e, n, alpha, S_0, Grid, Grid , actions)
    L24=q_sig(Qn24 , e, n, alpha, S_0, Grid, Grid , actions)
    L25=q_sig(Qn25 , e, n, alpha, S_0, Grid, Grid , actions)
    L26=q_sig(Qn26 , e, n, alpha, S_0, Grid, Grid , actions)
    L27=q_sig(Qn27 , e, n, alpha, S_0, Grid, Grid , actions)
    L28=q_sig(Qn28 , e, n, alpha, S_0, Grid, Grid , actions)
    L29=q_sig(Qn29 , e, n, alpha, S_0, Grid, Grid , actions)
    Qn=L[0]
    Qn1=L1[0]
    Qn2=L2[0]
    Qn3=L3[0]
    Qn4=L4[0]
    Qn5=L5[0]
    Qn6=L6[0]
    Qn7=L7[0]
    Qn8=L8[0]
    Qn9=L9[0]
    Qn10=L10[0]
    Qn11=L11[0]
    Qn12=L12[0]
    Qn13=L13[0]
    Qn14=L14[0]
    Qn15=L15[0]
    Qn16=L16[0]
    Qn17=L17[0]
    Qn18=L18[0]
    Qn19=L19[0]
    Qn20=L20[0]
    Qn21=L21[0]
    Qn22=L22[0]
    Qn23=L23[0]
    Qn24=L24[0]
    Qn25=L25[0]
    Qn26=L26[0]
    Qn27=L27[0]
    Qn28=L28[0]
    Qn29=L29[0]
    exp=np.mean([L[1],L1[1],L2[1],L3[1],L4[1],L5[1],L6[1],L7[1],L8[1],L9[1]])
    tree=np.mean([L10[1],L11[1],L12[1],L13[1],L14[1],L15[1],L16[1],L17[1],L18[1],L19[1]])
    sig=np.mean([L20[1],L21[1],L22[1],L23[1],L24[1],L25[1],L26[1],L27[1],L28[1],L29[1]])
    print(exp)
    print(tree)
    print(sig)
    end = time.time()
    print('Approx run time in secs' , N*(end - start))
    print(i)
    L_exp.append(exp)
    L_tree.append(tree)
    L_sig.append(sig)


line1=plt.plot(list(range(2,N+1,1)),L_exp[2:301],'b-', label='Exp_Sarsa')
line2 = plt.plot(list(range(2,N+1,1)),L_tree[2:301],'r-', label='Tree')
line3= plt.plot(list(range(2,N+1,1)),L_sig[2:301],'g-',label='Q_sig')
plt.xlabel('Episodes')
plt.ylabel('Sum_of_Rewards_during_Episode')
plt.title('n=2')
plt.legend()
plt.show()
