from ANNarchy import *
import pylab as plt
from BGM_22 import izhikevich2007_standard, izhikevich2007_fsi
setup(dt=0.1)


"""
README:

1. set a,b,c,d
if b < 0.267:
1. calculate v1,v2,u1,u2 = steady-state points
2. initialize u,v with steady-state points
3. check if its a stable steady-state (small stimulus --> goes back to steady-state?)
4. if yes ... great
5. if no... init v and u within phase plot
6. init v and u at minimum v
else:
1. init v and u with 0
2. check phase plot
3. init v and u within phase plot
4. init v and u at minimum v

"""

C=80
k=1
v_r=-70
v_t=-50
a=0.2
b=0.025
c=-60
d=0
v_peak=25
v_b=-55

### INITIALIZE HERE
v_init=-30
u_init=0


pop=Population(1,neuron=izhikevich2007_fsi)
pop.C=C
pop.k=k
pop.v_r=v_r
pop.v_t=v_t
pop.a=a
pop.b=b
pop.c=c
pop.d=d
pop.v_peak=v_peak
#pop.v_b=v_b

m=Monitor(pop,['v','u'])

compile()

pop.v=v_init
pop.u=u_init

simulate(1000)
pop.I_add=10
simulate(1)
pop.I_add=0
simulate(3000)

v=m.get('v')[:,0]
u=m.get('u')[:,0]

plt.figure()
plt.plot(v,u)
plt.plot(v[0],u[0],'k.',markersize=20)
plt.savefig('phase_plot.svg')

plt.figure()
plt.subplot(211)
plt.plot(v,'k')
plt.subplot(212)
plt.plot(u,'k')
plt.savefig('time_plot.svg')

print(v[np.argmin(v)])
print(u[np.argmin(v)])




