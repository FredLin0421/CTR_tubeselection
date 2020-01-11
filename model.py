# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""



import openmdao.api as om
import numpy as np
from math import cos, sin, sqrt
from stiffness import Stiffness

class CTR(om.ExplicitComponent):
    
    def setup(self):
        
        self.add_input('length1', val=np.zeros(95))
        self.add_input('length2', val=np.zeros(95))
        self.add_input('length4', val=np.zeros(95))
        self.add_input('kappa2', val=0.0)
        self.add_input('kb1')
        self.add_input('kb2')
        self.add_input('kb3')
        self.add_input('l22', val=np.zeros(95))
        self.add_input('psi2',val=np.zeros(95))
        self.add_output('f', val=0.0)
        

        # Finite difference all partials.
        self.declare_partials('*', '*', method='fd')

    def compute(self, inputs, outputs):
    
        length1 = inputs['length1']
        length2 = inputs['length2']
        length4 = inputs['length4']
        kappa2 = inputs['kappa2']
        kb1 = inputs['kb1']
        kb2 = inputs['kb2']
        kb3 = inputs['kb3']
        l22 = inputs['l22']
        psi2 = inputs['psi2']
        
        # Derive the backbone points
        def backbone(inputs):
            
            backbone_point = [[np.zeros(len(length4)),np.zeros(len(length4)),length4],
                              [ -(np.cos(psi2)*(np.cos((kappa2*kb2*(l22 - length2)/2)/(kb1 + kb2 + kb3)) - 1)*(kb1 + kb2 + kb3))/(kappa2*kb2), -(np.sin(psi2)*(np.cos((kappa2*kb2*(l22 - length2)/2)/(kb1 + kb2 + kb3)) - 1)*(kb1 + kb2 + kb3))/(kappa2*kb2), length4 + (np.sin((kappa2*kb2*(l22 - length2)/2)/(kb1 + kb2 + kb3))*(kb1 + kb2 + kb3))/(kappa2*kb2)],
                              [ -(np.cos(psi2)*(np.cos((kappa2*kb2*(l22 - length2))/(kb1 + kb2 + kb3)) - 1)*(kb1 + kb2 + kb3))/(kappa2*kb2), -(np.sin(psi2)*(np.cos((kappa2*kb2*(l22 - length2))/(kb1 + kb2 + kb3)) - 1)*(kb1 + kb2 + kb3))/(kappa2*kb2), length4 + (np.sin((kappa2*kb2*(l22 - length2))/(kb1 + kb2 + kb3))*(kb1 + kb2 + kb3))/(kappa2*kb2)],
                              [ (np.sin((kappa2*kb2*(l22 - length2/2))/(kb1 + kb2 + kb3))*np.cos(psi2)*np.sin((kappa2*kb2*(length2/2))/(kb1 + kb2))*(kb1 + kb2))/(kappa2*kb2) - (np.cos((kappa2*kb2*(l22 - length2/2))/(kb1 + kb2 + kb3))*np.cos(psi2)*(np.cos((kappa2*kb2*(length2/2))/(kb1 + kb2)) - 1)*(kb1 + kb2))/(kappa2*kb2) - (np.cos(psi2)*(np.cos((kappa2*kb2*(l22 - length2/2))/(kb1 + kb2 + kb3)) - 1)*(kb1 + kb2 + kb3))/(kappa2*kb2), (np.sin((kappa2*kb2*(l22 - length2/2))/(kb1 + kb2 + kb3))*np.sin((kappa2*kb2*(length2/2))/(kb1 + kb2))*np.sin(psi2)*(kb1 + kb2))/(kappa2*kb2) - (np.cos((kappa2*kb2*(l22 - length2/2))/(kb1 + kb2 + kb3))*np.sin(psi2)*(np.cos((kappa2*kb2*(length2/2))/(kb1 + kb2)) - 1)*(kb1 + kb2))/(kappa2*kb2) - (np.sin(psi2)*(np.cos((kappa2*kb2*(l22 - length2/2))/(kb1 + kb2 + kb3)) - 1)*(kb1 + kb2 + kb3))/(kappa2*kb2), length4 + (np.sin((kappa2*kb2*(l22 - length2/2))/(kb1 + kb2 + kb3))*(kb1 + kb2 + kb3))/(kappa2*kb2) + (np.cos((kappa2*kb2*(l22 - length2/2))/(kb1 + kb2 + kb3))*np.sin((kappa2*kb2*(length2/2))/(kb1 + kb2))*(kb1 + kb2))/(kappa2*kb2) + (np.sin((kappa2*kb2*(l22 - length2/2))/(kb1 + kb2 + kb3))*(np.cos((kappa2*kb2*(length2/2))/(kb1 + kb2)) - 1)*(kb1 + kb2))/(kappa2*kb2)],
                              [ (np.sin((kappa2*kb2*(l22 - length2))/(kb1 + kb2 + kb3))*np.cos(psi2)*np.sin((kappa2*kb2*(length2))/(kb1 + kb2))*(kb1 + kb2))/(kappa2*kb2) - (np.cos((kappa2*kb2*(l22 - length2))/(kb1 + kb2 + kb3))*np.cos(psi2)*(np.cos((kappa2*kb2*(length2))/(kb1 + kb2)) - 1)*(kb1 + kb2))/(kappa2*kb2) - (np.cos(psi2)*(np.cos((kappa2*kb2*(l22 - length2))/(kb1 + kb2 + kb3)) - 1)*(kb1 + kb2 + kb3))/(kappa2*kb2), (np.sin((kappa2*kb2*(l22 - length2))/(kb1 + kb2 + kb3))*np.sin((kappa2*kb2*(length2))/(kb1 + kb2))*np.sin(psi2)*(kb1 + kb2))/(kappa2*kb2) - (np.cos((kappa2*kb2*(l22 - length2))/(kb1 + kb2 + kb3))*np.sin(psi2)*(np.cos((kappa2*kb2*(length2))/(kb1 + kb2)) - 1)*(kb1 + kb2))/(kappa2*kb2) - (np.sin(psi2)*(np.cos((kappa2*kb2*(l22 - length2))/(kb1 + kb2 + kb3)) - 1)*(kb1 + kb2 + kb3))/(kappa2*kb2), length4 + (np.sin((kappa2*kb2*(l22 - length2))/(kb1 + kb2 + kb3))*(kb1 + kb2 + kb3))/(kappa2*kb2) + (np.cos((kappa2*kb2*(l22 - length2))/(kb1 + kb2 + kb3))*np.sin((kappa2*kb2*(length2))/(kb1 + kb2))*(kb1 + kb2))/(kappa2*kb2) + (np.sin((kappa2*kb2*(l22 - length2))/(kb1 + kb2 + kb3))*(np.cos((kappa2*kb2*(length2))/(kb1 + kb2)) - 1)*(kb1 + kb2))/(kappa2*kb2)],
                              [np.zeros(len(length4)),np.zeros(len(length4)),length1/2]]
            return backbone_point
        
        def vec_distant(x,y,z):
            x = np.concatenate((x), axis=None)
            y = np.concatenate((y), axis=None)
            z = np.concatenate((z), axis=None)
            X = np.subtract(x[:93],x[1:94])
            Y = np.subtract(y[:93],y[1:94])
            Z = np.subtract(z[:93],z[1:94])
            dist = np.sum(np.sqrt((np.square(X)+np.square(Y)+np.square(Z))))
            return dist
        
        """def vec_surface(x,y,z):
            x = np.array(x)
            y = np.array(y)
            z = np.array(z)
            X = x[:][np.newaxis,:] - x[:][:, np.newaxis]
            Y = y[:][np.newaxis,:] - y[:][:, np.newaxis]
            Z = z[:][np.newaxis,:] - z[:][:, np.newaxis]
            dist = np.sum(np.sqrt((np.square(X.flatten())+np.square(Y.flatten())+np.square(Z.flatten()))))
            return dist"""
        
        backbone_point = np.array(backbone(inputs))
        x = [backbone_point[0][0],backbone_point[1][0],backbone_point[2][0],backbone_point[3][0],backbone_point[4][0],backbone_point[5][0]]
        y = [backbone_point[0][1],backbone_point[1][1],backbone_point[2][1],backbone_point[3][1],backbone_point[4][1],backbone_point[5][1]]
        z = [backbone_point[0][2],backbone_point[1][2],backbone_point[2][2],backbone_point[3][2],backbone_point[4][2],backbone_point[5][2]]
        
        # Objective function 1
        outputs['f'] = vec_distant(x,y,z)
        # Objective function 2
        """outputs['f'] = vec_surface(x,y,z) * 0.1"""
        # Objective function 3
        #backbone_point = np.array(backbone(inputs))
        """outputs['f'] = np.linalg.norm(backbone_point)"""
        
        

if __name__ == "__main__":
    N = 95
    model = om.Group()
    ivc = om.IndepVarComp()
    ivc.add_output('length1', np.ones(N))
    ivc.add_output('length2', np.ones(N))
    ivc.add_output('kappa2', 1.0)
    ivc.add_output('kb1', 1.0)
    ivc.add_output('kb2', 1.0)
    ivc.add_output('kb3', 1.0)
    ivc.add_output('l22', np.ones(N))
    ivc.add_output('psi2', np.ones(N))
    model.add_subsystem('des_vars', ivc)
    model.add_subsystem('parab_comp', CTR())

    model.connect('des_vars.length1', 'parab_comp.length1')
    model.connect('des_vars.length2', 'parab_comp.length2')
    model.connect('des_vars.kappa2', 'parab_comp.kappa2')
    model.connect('des_vars.kb1', 'parab_comp.kb1')
    model.connect('des_vars.kb2', 'parab_comp.kb2')
    model.connect('des_vars.kb3', 'parab_comp.kb3')
    model.connect('des_vars.l22', 'parab_comp.l22')
    model.connect('des_vars.psi2', 'parab_comp.psi2')
 

    prob = om.Problem(model)
    prob.setup()
    prob.run_model()
    print((prob['parab_comp.length1']))
    print(prob['parab_comp.f'])
