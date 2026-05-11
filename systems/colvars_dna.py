import math
import torch
import numpy as np
from boltz.model.potentials import potentials, schedules 
##################################################
# Routines for user to define colvar for TDS
# steering of FKPF
##################################################
def fkpf_options(steering_args):
    """
    Produce the parameters passed to the BayesianPotential under get_potentials.

    Returns
        - parameters: dictionary with kwargs
            -- guidance_interval: how often the gradient is recomputed for gradient
                                  perturbation of the denoised prediction under
                                  conditoning
            -- guidance_weight: the tempering coefficients used to scale the
                                energy gradient perturbation of the denoiser
            -- resampling_weight: the tempering powers used on the FKPF potentials
                                  in intermediate computations of the likelihood.
                                  At time 0, this should compute to 1 for consistency
                                  with desired Bayesian conditioning.
            -- write_diff_traj_ids: which diff trajectory ids (before reordering like
                                    minimal energy) to write to disk if diffusion
                                    trajectories are being recorded.
    """

    nsmps = steering_args["num_diffsamples"]
    nparticles = steering_args["num_particles"]
    parameters = {
            "guidance_interval": 1,
            "guidance_weight": PowerSchedule(0.,125.,2,flag_guidance_weight=True),
            "resampling_weight": PowerSchedule(0.,125.,2),
            "write_diff_traj_ids": [k*nparticles for k in range(nsmps)],
        }

    return parameters

class BayesianPotential(potentials.Potential):
    """
    Class that defines potentials for Bayesian conditioning of diffusion
    trajectories. The user is expected to modify compute_variable and 
    compute_function for defining their own colvars. They may optionally
    program compute_args to be used inside of compute_variable or 
    compute_args.
    """
    def compute_args(self, feats, **parameters):
        """
        Optional method to be used inside compute_variable or compute_function, 
        which is included because it's an abstract method of the Potential class.
        """
        
        # Boltz2 standard compute function will just return 0's if index = None
        index = None
        args = None
        com_args = None
        ref_args = None
        operator_args = None

        return index, args, com_args, ref_args, operator_args

    def compute_variable(self,coords,index,compute_gradient=False):
        """
        Compute the relevant colvar from atomic coordinates.
        """
        device = coords.device
        a = torch.tensor(0.).to(device)
        b = torch.tensor(100.).to(device)
        d = torch.tensor(35.).to(device)

        # Atom selections for the dna ends
        # Top is Chain A 1DG and Chain B 12DC
        Nterm = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,
                 473,474,475,476,477,478,479,480,481,482,483,484,485,486,487,488,489,490,491]
        # Bottom is Chain A 12-DT and Chain B 1-DA
        Cterm = [229,230,231,232,233,234,235,236,237,238,239,240,241,242,243,244,245,246,247,248,
                 249,250,251,252,253,254,255,256,257,258,259,260,261,262,263,264,265,266,267,268,269]

        # Make the selections compatible with autograd tracking
        #Nterm = torch.tensor(Nterm, device=device)
        #Cterm = torch.tensor(Cterm, device=device)

        # grab xyz's of the selections
        xyzNterm = coords[:,Nterm,:] #torch.index_select(coords, dim=1, index=Nterm)
        xyzCterm = coords[:,Cterm,:] #torch.index_select(coords, dim=1, index=Cterm)

        # compute centers of mass
        cmNterm = torch.sum(xyzNterm,dim=1)/xyzNterm.shape[1]
        cmCterm = torch.sum(xyzCterm,dim=1)/xyzCterm.shape[1]

        # compute distances between centers of mass of helices that forms colvar
        zetatmp = torch.sqrt( torch.sum( torch.square( cmNterm - cmCterm  ),dim=1  ) )

        # compute colvar via a sigmoidal curve
        zetaclip1 = torch.clamp(zetatmp,a,d)
        zetaclip2 = torch.clamp(zetatmp,d,b)
        zeta1 = 3*((zetaclip1-a)/(d-a))**2-2*((zetaclip1-a)/(d-a))**3
        zeta2 = 3*((b-zetaclip2)/(b-d))**2-2*((b-zetaclip2)/(b-d))**3
        value = zeta1*zeta2

        return value

    def compute_function(self, value, *args, negation_mask=None, compute_derivative=False):
        """
        Compute the negative log-likelihood (energy) for the desired Bayesian 
        conditioning as a function of the colvars.
        """
        device = value.device
        xi = torch.tensor(1).to(device)
        d = torch.tensor(0.010).to(device)

        energy = 0.5*torch.square( (value-xi)/d  )

        return energy

    def compute(self, coords, feats, parameters):
        """
        Compute the negative log-likelihood (energy) for the desired Bayesian
        conditioning as a function of the atomic coordinates.
        """

        index, args, com_args, ref_args, operator_args = self.compute_args(feats, **parameters)

        value = self.compute_variable(coords,index)
        energy0 = self.compute_function(value,args)

        # Diffusion scales the log G by fk_lambda, which we remove from Bayes to be consistent
        # with conditioning Boltz's distribution by the intended likelihood
        energy = energy0/parameters["fk_lambda"]

        return energy

    def compute_gradient(self, coords, feats, parameters):
        """
        Compute the gradient of the negative log-likelihood (energy) for the desired Bayesian
        conditioning with respect to the atomic coordinates.
        """
        index, args, com_args, ref_args, operator_args = self.compute_args(feats, **parameters)

        with torch.set_grad_enabled(True):
            coords_wgrad = coords.detach().clone().requires_grad_(True)
            value = self.compute_variable(coords_wgrad,index)
            energy0 = self.compute_function(value,args)

            # Diffusion scales the log G by fk_lambda, which we remove from Bayes to be consistent
            # with conditioning Boltz's distribution by the intended likelihood
            energy = energy0/parameters["fk_lambda"]

            # the kth component of cv restraint only depends on Xk in the batch, so if you
            # sum over all components, the dXk gradient of the sum will coincide with 
            # dXk gradient of the kth component as all other terms will go to 0.
            agg_energy = torch.sum(energy)
            agg_energy.backward()

        return coords_wgrad.grad

class PowerSchedule(schedules.ParameterSchedule):
    """
    This is a simple tempering schedule that is modeled after x^p on [0,1]
    to define its schedule.
    """
    def __init__(self,start,end,power,
                 flag_guidance_weight=False):
        """
        Parameters
            - start: beneath start the schedule returns 1
            - end:   after end the schedule returns 0
            - power: the power for x^p model
            - flag_guidance_weight: whether this schedule is for guidance
                                    and the t**2 coefficient on the
                                    gradient perturbation should be
                                    included
        """
        self.start = start
        self.end = end
        self.power = power
        self.flag_guidance_weight = flag_guidance_weight

    def compute(self,t):
        if t <= self.start:
            return 1

        elif t >= self.end:    
            return 0
        
        else:
            u = 1-(t - self.start)/(self.end - self.start)
            v = u**self.power

        if not self.flag_guidance_weight:
            return v
        else:
            return (t**2)*v
