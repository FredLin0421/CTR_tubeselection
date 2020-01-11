import numpy as np
from openmdao.api import ExplicitComponent
E = 80

class Stiffness(ExplicitComponent):
    def setup(self):
        self.add_input('d1')
        self.add_input('d2')
        self.add_input('d3')
        self.add_input('d4')
        self.add_input('d5')
        self.add_input('d6')
        self.add_output('kb1')
        self.add_output('kb2')
        self.add_output('kb3')
        self.declare_partials('kb1','d1')
        self.declare_partials('kb1','d2')
        self.declare_partials('kb2','d3')
        self.declare_partials('kb2','d4')
        self.declare_partials('kb3','d5')
        self.declare_partials('kb3','d6')
        

    def compute(self,inputs,outputs):
        E=80
        d1 = inputs['d1']
        d2 = inputs['d2']
        d3 = inputs['d3']
        d4 = inputs['d4']
        d5 = inputs['d5']
        d6 = inputs['d6']
        outputs['kb1'] = E*np.pi*(d2**4-d1**4)/64
        outputs['kb2'] = E*np.pi*(d4**4-d3**4)/64
        outputs['kb3'] = E*np.pi*(d6**4-d5**4)/64

    def compute_partials(self, inputs, partials):
        E=80
        d1 = inputs['d1']
        d2 = inputs['d2']
        d3 = inputs['d3']
        d4 = inputs['d4']
        d5 = inputs['d5']
        d6 = inputs['d6']
        
        partials['kb1','d1'] = E*np.pi*(-4*d1**3)/64
        partials['kb1','d2'] = E*np.pi*(4*d2**3)/64
        partials['kb2','d3'] = E*np.pi*(-4*d3**3)/64
        partials['kb2','d4'] = E*np.pi*(4*d4**3)/64
        partials['kb3','d5'] = E*np.pi*(-4*d5**3)/64
        partials['kb3','d6'] = E*np.pi*(4*d6**3)/64 



