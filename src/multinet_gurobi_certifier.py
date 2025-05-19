import torch
import gurobipy as grb
import numpy as np

class MultiNetMILPTransformer:
    def __init__(self, eps, input, lAs, lbiases, batch_size):
        self.device = 'cpu'
        self.eps = eps.cpu().reshape(-1).detach().numpy()
        self.input = input.cpu().reshape(-1).detach().numpy()
        self.lAs = lAs
        self.lbiases = lbiases
        self.refined_lAs = None
        self.refined_lbiases = None
        self.batch_size = batch_size
        print("Batch size:", batch_size)
        assert len(lAs) == batch_size
        assert len(lbiases) == batch_size
        # input var and output vars for each network in 
        # the ensemble of networks.
        self.input_var = None
        # Gurobi model
        self.gmdl = grb.Model()
        self.gmdl.setParam('OutputFlag', False)
        self.gmdl.setParam('TimeLimit', 600)
        self.gmdl.Params.MIPFocus = 3
        self.gmdl.Params.ConcurrentMIP = 3

    def input_constraints(self):
        self.input_var = self.gmdl.addMVar(self.input.shape[0], lb = -self.eps + self.input, 
                                  ub = self.eps + self.input, vtype=grb.GRB.CONTINUOUS, name='all_x')
    
    def output_variables(self):
        output_length = 9
        self.output_vars = [self.gmdl.addMVar(output_length, 
                        lb=-float('inf'), ub=float('inf'),
                        vtype=grb.GRB.CONTINUOUS, name=f'output_{i}')
                        for i in range(self.batch_size)]

    def formulate_constriants(self):
        self.input_constraints()
        self.output_variables()

        for i in range(self.batch_size):
            la_coef = self.lAs[i][0].cpu().detach().numpy()
            lbias = self.lbiases[i][0].cpu().detach().numpy()
            self.gmdl.addConstr(self.output_vars[i] >= la_coef @ self.input_var + lbias)
        # Add constraints from refined linear coefficients if available
        if self.refined_lAs is not None and self.refined_lbiases is not None:
            for i in range(self.batch_size):
                la_coef = self.refined_lAs[i][0].cpu().detach().numpy()
                lbias = self.refined_lbiases[i][0].cpu().detach().numpy()
                self.gmdl.addConstr(self.output_vars[i] >= la_coef @ self.input_var + lbias)
        return self


    def handle_optimization_res(self):
        if self.gmdl.status in [2, 6, 10]:
            # print("Final MIP gap value: %f" % self.gmdl.MIPGap)
            # try:
            #     print("Final MIP best value: %f" % self.final_ans.X)
            # except:
            #     print("No solution obtained")
            # print("Final ObjBound: %f" % self.gmdl.ObjBound)
            return self.gmdl.ObjBound
        else:
            if self.gmdl.status == 4:
                return 0.0
            elif self.gmdl.status in [9, 11, 13]:
                print("Suboptimal solution")

                print("Final MIP gap value: %f" % self.gmdl.MIPGap)
                try:
                    print("Final MIP best value: %f" % self.final_ans.X)
                except:
                    print("No solution obtained")
                print("Final ObjBound: %f" % self.gmdl.ObjBound)
                if self.gmdl.SolCount > 0:
                    return self.gmdl.ObjBound
                else:
                    return 0.0    
            print("Gurobi model status", self.gmdl.status)
            print("The optimization failed\n")            
            if self.gmdl.status == 3:
                pass

            return 0.0

            
    def solv_MILP(self):
        bs = [] # indicator for each network
        for i, final_var in enumerate(self.output_vars):
            final_var_min = self.gmdl.addVar(lb=-float('inf'), ub=float('inf'), 
                                                vtype=grb.GRB.CONTINUOUS, 
                                                name=f'final_var_min_{i}')
            self.gmdl.addGenConstrMin(final_var_min, final_var.tolist())
            bs.append(self.gmdl.addVar(vtype=grb.GRB.BINARY, name=f'b{i}'))
            self.gmdl.addGenConstrIndicator(bs[-1], True, final_var_min >= -1e-10)
            self.gmdl.addGenConstrIndicator(bs[-1], False, final_var_min <= -1e-10)
        
        
        self.final_ans = self.gmdl.addVar(vtype=grb.GRB.CONTINUOUS, name=f'p')
        self.gmdl.addConstr(self.final_ans == grb.quicksum(bs[i] for i in range(self.batch_size)))
        self.gmdl.update()
        self.gmdl.setObjective(self.final_ans, grb.GRB.MINIMIZE)
        self.gmdl.optimize()
        return self.handle_optimization_res()