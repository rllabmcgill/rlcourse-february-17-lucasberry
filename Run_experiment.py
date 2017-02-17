from Comparison_Q_sig import *
import time


# N is the number of episodes
# n how far you look forward
# e is epsilon
# S_0 is the start
# M is the number of runs
N = 100
n = 4
e = .1
alpha = [x / 100 for x in range(20, 120, 20)]
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

np.random.seed(3)
L_exp_avgs=[]
L_tree_avgs=[]
L_sig_avgs=[]
for a in alpha:
    print('Current alpha' , a)
    L_exp_avgs_r=[]
    L_tree_avgs_r=[]
    L_sig_avgs_r=[]
    for m in range(M):
        run_tot_exp=0
        run_tot_tree=0
        run_tot_sig=0
        print('Current m' , m)
        start = time.time()
        Qn=np.copy(Q)
        Qn1=np.copy(Q)
        Qn2=np.copy(Q)
        for u in range(N):
            L=exp_sarsa(Qn , e, n, a, S_0, Grid, Grid , actions)
            L1=tree_backup(Qn1 , e, n, a, S_0, Grid, Grid , actions)
            L2=q_sig(Qn2 , e, n, a, S_0, Grid, Grid , actions)
            Qn=L[0]
            run_tot_exp += L[1]
            Qn1=L1[0]
            run_tot_tree += L1[1]
            Qn2=L2[0]
            run_tot_sig += L2[1]
        L_exp_avgs_r.append(run_tot_exp/N)
        L_tree_avgs_r.append(run_tot_tree/N)
        L_sig_avgs_r.append(run_tot_sig/N)
        end = time.time()
        print('Approx run time in secs' , len(alpha)*M*(end - start))
        print(run_tot_exp/N)
        print(run_tot_tree/N)
        print(run_tot_sig/N)
    L_exp_avgs.append(np.mean(L_exp_avgs_r))
    L_tree_avgs.append(np.mean(L_tree_avgs_r))
    L_sig_avgs.append(np.mean(L_sig_avgs_r))

print(L_exp_avgs)
print(L_tree_avgs)
print(L_sig_avgs)


line1=plt.plot(alpha,L_exp_avgs,'b-', label='Exp_Sarsa')
line2 = plt.plot(alpha,L_tree_avgs,'r-', label='Tree')
line3= plt.plot(alpha,L_sig_avgs,'g-',label='Q_sig')
plt.xlabel('alpha')
plt.ylabel('Average_Return')
plt.title('n=4')
plt.legend()
plt.show()
