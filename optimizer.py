# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 17:03:44 2019

@author: Morimoto Lab
"""
import openmdao.api as om
import numpy as np
from model import CTR
import h5py
import time
import scipy.io
from stiffness import Stiffness


t0=time.time()
# Load the path point
f = h5py.File('path.mat','r')
data_pa = f.get('Pa')
data_pb = f.get('Pb')
data_pa = np.array(data_pa)
data_pb = np.array(data_pb)
# Adjust the base position of the concentric tube robot
data_pa[0] = data_pa[0] + 30
data_pa[2] = data_pa[2] - 100

# We'll use the component that was defined in the last tutorial

dim = np.shape(data_pa)
N = dim[1]
E=80
# build the model
prob = om.Problem()
indeps = prob.model.add_subsystem('indeps', om.IndepVarComp())


indeps.add_output('length1', np.ones(N)*(-65))
indeps.add_output('length2', np.ones(N)*(-40))
indeps.add_output('length4', np.ones(N)*(-55))
indeps.add_output('kappa2', .04)
indeps.add_output('d1', 1.5)
indeps.add_output('d2', 1.7)
indeps.add_output('d3', 2.1)
indeps.add_output('d4', 2.6)
indeps.add_output('d5', 2.8)
indeps.add_output('d6', 3.3)
indeps.add_output('l22', np.ones(N)*(-45))
indeps.add_output('psi2', np.ones(N))


prob.model.add_subsystem('ctr', CTR())
prob.model.add_subsystem('Stiffness',Stiffness())
# define the component whose output will be constrained
prob.model.add_subsystem('const1', om.ExecComp('x = length1*(cos((kappa2*kb2*(l22 - length2))/(kb1 + kb2 + kb3))*cos(psi2)*sin((kappa2*kb2*(length2))/(kb1 + kb2)) + sin((kappa2*kb2*(l22 - length2))/(kb1 + kb2 + kb3))*cos((kappa2*kb2*(length2))/(kb1 + kb2))*cos(psi2)) - (cos(psi2)*(cos((kappa2*kb2*(l22 - length2))/(kb1 + kb2 + kb3)) - 1)*(kb1 + kb2 + kb3))/(kappa2*kb2) - (cos((kappa2*kb2*(l22 - length2))/(kb1 + kb2 + kb3))*cos(psi2)*(cos((kappa2*kb2*(length2))/(kb1 + kb2)) - 1)*(kb1 + kb2))/(kappa2*kb2) + (sin((kappa2*kb2*(l22 - length2))/(kb1 + kb2 + kb3))*cos(psi2)*sin((kappa2*kb2*(length2))/(kb1 + kb2))*(kb1 + kb2))/(kappa2*kb2)',x=np.zeros(N),l22=np.zeros(N),length2=np.zeros(N),length1=np.zeros(N),psi2=np.zeros(N)))
prob.model.add_subsystem('const2', om.ExecComp('y = length1*(cos((kappa2*kb2*(l22 - length2))/(kb1 + kb2 + kb3))*sin((kappa2*kb2*(length2))/(kb1 + kb2))*sin(psi2) + sin((kappa2*kb2*(l22 - length2))/(kb1 + kb2 + kb3))*cos((kappa2*kb2*(length2))/(kb1 + kb2))*sin(psi2)) - (sin(psi2)*(cos((kappa2*kb2*(l22 - length2))/(kb1 + kb2 + kb3)) - 1)*(kb1 + kb2 + kb3))/(kappa2*kb2) - (cos((kappa2*kb2*(l22 - length2))/(kb1 + kb2 + kb3))*sin(psi2)*(cos((kappa2*kb2*(length2))/(kb1 + kb2)) - 1)*(kb1 + kb2))/(kappa2*kb2) + (sin((kappa2*kb2*(l22 - length2))/(kb1 + kb2 + kb3))*sin((kappa2*kb2*(length2))/(kb1 + kb2))*sin(psi2)*(kb1 + kb2))/(kappa2*kb2)',y=np.zeros(N),l22=np.zeros(N),length2=np.zeros(N),length1=np.zeros(N),psi2=np.zeros(N)))
prob.model.add_subsystem('const3', om.ExecComp('z = length4 + length1*(cos((kappa2*kb2*(l22 - length2))/(kb1 + kb2 + kb3))*cos((kappa2*kb2*(length2))/(kb1 + kb2)) - sin((kappa2*kb2*(l22 - length2))/(kb1 + kb2 + kb3))*sin((kappa2*kb2*(length2))/(kb1 + kb2))) + (sin((kappa2*kb2*(l22 - length2))/(kb1 + kb2 + kb3))*(kb1 + kb2 + kb3))/(kappa2*kb2) + (cos((kappa2*kb2*(l22 - length2))/(kb1 + kb2 + kb3))*sin((kappa2*kb2*(length2))/(kb1 + kb2))*(kb1 + kb2))/(kappa2*kb2) + (sin((kappa2*kb2*(l22 - length2))/(kb1 + kb2 + kb3))*(cos((kappa2*kb2*(length2))/(kb1 + kb2)) - 1)*(kb1 + kb2))/(kappa2*kb2)',z=np.zeros(N),l22=np.zeros(N),length4=np.zeros(N),length1=np.zeros(N),length2=np.zeros(N)))
prob.model.add_subsystem('const4', om.ExecComp('g=l22-length2', g=np.zeros(N),l22=np.zeros(N),length2=np.zeros(N)))
prob.model.add_subsystem('const5', om.ExecComp('k1=kb1'))
prob.model.add_subsystem('const6', om.ExecComp('k2=kb2'))
prob.model.add_subsystem('const12', om.ExecComp('k3=kb3'))
"""prob.model.add_subsystem('const5', om.ExecComp('k1=kb2-kb1'))
prob.model.add_subsystem('const6', om.ExecComp('k2=kb3-kb2'))"""
prob.model.add_subsystem('const7', om.ExecComp('t1=d2-d1'))
prob.model.add_subsystem('const8', om.ExecComp('t2=d4-d3'))
prob.model.add_subsystem('const9', om.ExecComp('t3=d6-d5'))
prob.model.add_subsystem('const10', om.ExecComp('t12=d3-d2')) 
prob.model.add_subsystem('const11', om.ExecComp('t23=d5-d4'))

# Connect the variables
prob.model.connect('indeps.d1', ['Stiffness.d1','const7.d1'])
prob.model.connect('indeps.d2', ['Stiffness.d2','const10.d2','const7.d2'])
prob.model.connect('indeps.d3', ['Stiffness.d3','const10.d3','const8.d3'])
prob.model.connect('indeps.d4', ['Stiffness.d4','const11.d4','const8.d4'])
prob.model.connect('indeps.d5', ['Stiffness.d5','const11.d5','const9.d5'])
prob.model.connect('indeps.d6', ['Stiffness.d6','const9.d6'])
prob.model.connect('indeps.l22', ['ctr.l22','const1.l22','const2.l22','const3.l22','const4.l22'])
prob.model.connect('indeps.length1', ['ctr.length1','const1.length1','const2.length1','const3.length1'])
prob.model.connect('indeps.length2', ['ctr.length2','const1.length2','const2.length2','const3.length2','const4.length2'])
prob.model.connect('indeps.length4', ['ctr.length4','const3.length4'])
prob.model.connect('indeps.kappa2', ['ctr.kappa2','const1.kappa2','const2.kappa2','const3.kappa2'])
prob.model.connect('Stiffness.kb1', ['ctr.kb1','const1.kb1','const2.kb1','const3.kb1','const5.kb1'])
prob.model.connect('Stiffness.kb2', ['ctr.kb2','const1.kb2','const2.kb2','const3.kb2','const6.kb2'])
prob.model.connect('Stiffness.kb3', ['ctr.kb3','const1.kb3','const2.kb3','const3.kb3','const12.kb3'])
prob.model.connect('indeps.psi2', ['ctr.psi2','const1.psi2','const2.psi2'])


# setup the optimization
prob.driver = om.ScipyOptimizeDriver()
prob.driver.options['optimizer'] = 'SLSQP'
prob.driver.options['maxiter'] = 1500



# Add design variables
prob.model.add_design_var('indeps.length1', lower=-140, upper=0)
prob.model.add_design_var('indeps.length2', lower=-140, upper=0)
prob.model.add_design_var('indeps.length4', lower=-140, upper=0)
prob.model.add_design_var('indeps.kappa2', lower=0, upper=.08)
"""prob.model.add_design_var('indeps.kb1', lower=0, upper=60)
prob.model.add_design_var('indeps.kb2', lower=60, upper=100)
prob.model.add_design_var('indeps.kb3', lower=100, upper=200)"""
prob.model.add_design_var('indeps.l22', lower=-140, upper=0)
prob.model.add_design_var('indeps.psi2', lower=-2, upper=2)
prob.model.add_design_var('indeps.d1', lower=1, upper=3.5)
prob.model.add_design_var('indeps.d2', lower=1,upper=3.5)
prob.model.add_design_var('indeps.d3', lower=1,upper=3.5)
prob.model.add_design_var('indeps.d4', lower=1,upper=3.5)
prob.model.add_design_var('indeps.d5', lower=1,upper=3.5)
prob.model.add_design_var('indeps.d6', lower=1,upper=3.5)
prob.model.add_objective('ctr.f')

# Add constraints
prob.model.add_constraint('const1.x', equals=data_pa[0])
prob.model.add_constraint('const2.y', equals=data_pa[1])
prob.model.add_constraint('const3.z', equals=data_pa[2])
prob.model.add_constraint('const4.g', upper=0)
prob.model.add_constraint('const5.k1', lower=0,upper=50)
prob.model.add_constraint('const6.k2', lower=50,upper=90)
prob.model.add_constraint('const12.k3', lower=100,upper=200)
"""prob.model.add_constraint('const5.k1', lower=0)
prob.model.add_constraint('const6.k2', lower=0)"""
prob.model.add_constraint('const7.t1', lower=0.1,upper=1)
prob.model.add_constraint('const8.t2', lower=0.1,upper=1)
prob.model.add_constraint('const9.t3', lower=0.1,upper=1)
prob.model.add_constraint('const10.t12', lower=0.12, upper=0.15)
prob.model.add_constraint('const11.t23', lower=0.12, upper=0.15)
#prob.model.add_constraint('const7.l23', lower=0)
prob.setup()
prob.run_driver()

t1=time.time()
# print the optimal results
print('time:', t1-t0)
print('Length1 =',prob['indeps.length1'])
print('Length3 =',prob['indeps.l22']-prob['indeps.length2'])
print('Length2 =',prob['indeps.length2'])
print('Length4 =',prob['indeps.length4'])
print('kappa2 = ',prob['indeps.kappa2'])
print('kb1 = ',prob['Stiffness.kb1'])
print('kb2 = ',prob['Stiffness.kb2'])
print('kb3 = ',prob['Stiffness.kb3'])
print('d1 = ',prob['indeps.d1'])
print('d2 = ',prob['indeps.d2'])
print('d3 = ',prob['indeps.d3'])
print('d4 = ',prob['indeps.d4'])
print('d5 = ',prob['indeps.d5'])
print('d6 = ',prob['indeps.d6'])
print('l22 = ',prob['indeps.l22']) 
print('psi2 = ',prob['indeps.psi2'])

# Save joint values and tube parameters to .mat file
mdict = {'Length1':prob['indeps.length1'],'Length2':prob['indeps.length2'],'Length3':prob['indeps.l22']-prob['indeps.length2'],
        'Length4':prob['indeps.length4'],'l22':prob['indeps.l22'],'kb1':prob['Stiffness.kb1'],'kb2':prob['Stiffness.kb2'],
        'kb3':prob['Stiffness.kb3'],'kappa2':prob['indeps.kappa2'],'d1':prob['indeps.d1'],'d2':prob['indeps.d2'],'d3':prob['indeps.d3'],
        'd4':prob['indeps.d4'],'d5':prob['indeps.d5'],'d6':prob['indeps.d6'],'psi2':prob['indeps.psi2']}
# scipy.io.savemat('D:/Desktop/Fred/CTR/CTR optimization/CTR/Inverse_kinematics/jointvalues/jointvalue_distance_001.mat',mdict)
scipy.io.savemat('/Users/fredlin/Desktop/Morimoto Lab Research/Concentric robot suturing/CTR_tubeselection/CTR_tubeselection/jointvalues/jointvalue_distance_004_b30.mat',mdict)
