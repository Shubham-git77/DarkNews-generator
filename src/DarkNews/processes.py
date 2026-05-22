import numpy as np
import vegas as vg
import json
import os

from DarkNews import const
from DarkNews import pdg
from DarkNews import integrands
from DarkNews import MC
from DarkNews import model
from DarkNews import amplitudes as amps
from DarkNews import phase_space as ps
from DarkNews import decay_rates as dr

from . import Cfourvec as Cfv

import logging

logger = logging.getLogger("logger." + __name__)

# ─────────────────────────────────────────────────────────────────────────────
# Optional SIREN imports (needed only for ChiPrimeDecay / DarkPhotonDecay)
# ─────────────────────────────────────────────────────────────────────────────
try:
    from siren.interactions import DarkNewsDecay
    from siren import dataclasses
    from siren.dataclasses import Particle
    _SIREN_AVAILABLE = True
except ImportError:
    # DarkNews-side classes work without SIREN; Dutta-Kim SIREN classes will
    # raise a descriptive error only when actually instantiated.
    _SIREN_AVAILABLE = False
    DarkNewsDecay = object   # harmless fallback base class

import scipy.integrate as _sci_int
import scipy.special  as _sci_sp   # kept for completeness; used by flux builder


# ─────────────────────────────────────────────────────────────────────────────
# Private constants (Dutta-Kim / flux helpers, natural units)
# ─────────────────────────────────────────────────────────────────────────────
_ALPHA      = 1.0 / 137.036     # fine-structure constant
_GEV_TO_S_INV = 1.519e24        # 1 GeV / ℏ  in s⁻¹  (lifetime printing only)

_M_PI       = 0.13957           # GeV  charged pion mass
_M_MU       = 0.10566           # GeV  muon mass
_M_NU       = 0.0               # GeV  neutrino mass (massless)
_GF         = 1.1664e-5         # GeV⁻²  Fermi constant
_Vud        = 0.9740            # CKM element
_fpi        = 0.1307            # GeV  pion decay constant
_HBAR_C     = 0.197327e-13      # GeV·cm  (converts GeV⁻¹ to cm)


# =============================================================================
# DarkNews process classes
# =============================================================================

class UpscatteringProcess:
    def __init__(self, nu_projectile, nu_upscattered, nuclear_target, scattering_regime, TheoryModel, helicity):
        """
        A class to describe the process of neutrino upscattering, which involves a neutrino scattering off a target and gaining energy in the process.
        This class supports various scattering regimes (coherent, proton elastic, and neutron elastic), and allows for the calculation of total and differential cross sections for these processes.

        Attributes:
            nuclear_target (object): The nuclear target involved in the scattering process.
            scattering_regime (str): The regime of scattering, e.g., 'coherent', 'p-el' (proton elastic), 'n-el' (neutron elastic).
            target (object): The actual target of the scattering, which could be the whole nucleus, a constituent nucleon, or constituent quarks, depending on the scattering regime.
            target_multiplicity (int): The number of targets involved in the scattering process, relevant for calculating cross sections.
            nu_projectile (object): The incoming neutrino involved in the upscattering process.
            nu_upscattered (object): The upscattered neutrino resulting from the scattering process.
            TheoryModel (object): The theoretical model used to describe the interactions in the upscattering process.
            helicity (str): The helicity configuration of the upscattering process, either 'conserving' or 'flipping'.
            MA (float): The mass of the target involved in the scattering.
            mzprime (float): The mass of the Z' boson in the theory model, if applicable.
            mhprime (float): The mass of the H' boson in the theory model, if applicable.
            m_ups (float): The mass of the upscattered neutrino.
            Cij, Cji, Vij, Vji, Sij, Sji, Tij, Tji (float): Coupling constants for the interaction vertices involved in the upscattering process.
            Chad, Vhad, Shad (float): Hadronic coupling constants for the interaction vertices.
            Cprimehad (float): Mass-mixed vertex coupling constant for the hadronic interaction.
            Ethreshold (float): The minimum energy threshold for the upscattering process to occur.
            vectorized_total_xsec (function): A vectorized function to calculate the total cross section for the upscattering process.
            calculable_diagrams (list): A list of diagrams that can be calculated for the upscattering process.

        Methods:
            __init__(self, nu_projectile, nu_upscattered, nuclear_target, scattering_regime, TheoryModel, helicity): Initializes the upscattering process with specified parameters.
            scalar_total_xsec(self, Enu, diagram=\"total\", NINT=MC.NINT, NEVAL=MC.NEVAL, NINT_warmup=MC.NINT_warmup, NEVAL_warmup=MC.NEVAL_warmup, savefile_xsec=None, savefile_norm=None): Calculates the scalar total cross section for a given neutrino energy and diagram.
            total_xsec(self, Enu, diagrams=[\"total\"], NINT=MC.NINT, NEVAL=MC.NEVAL, NINT_warmup=MC.NINT_warmup, NEVAL_warmup=MC.NEVAL_warmup, seed=None, savestr=None): Calculates the total cross section for the upscattering process for a fixed neutrino energy.
            diff_xsec_Q2(self, Enu, Q2, diagrams=[\"total\"]): Calculates the differential cross section for the upscattering process as a function of the squared momentum transfer Q2.
        """

        self.nuclear_target = nuclear_target
        self.scattering_regime = scattering_regime
        if self.scattering_regime == "coherent":
            self.target = self.nuclear_target
        elif self.scattering_regime == "p-el":
            self.target = self.nuclear_target.get_constituent_nucleon("proton")
        elif self.scattering_regime == "n-el":
            self.target = self.nuclear_target.get_constituent_nucleon("neutron")
        elif self.scattering_regime == "DIS":
            self.target = self.nuclear_target.get_constituent_quarks()
        else:
            logger.error(f"Scattering regime {scattering_regime} not supported.")

        # How many constituent targets inside scattering regime?
        if self.scattering_regime == "coherent":
            self.target_multiplicity = 1
        elif self.scattering_regime == "p-el":
            self.target_multiplicity = self.nuclear_target.Z
        elif self.scattering_regime == "n-el":
            self.target_multiplicity = self.nuclear_target.N
        elif self.scattering_regime == "DIS":
            self.target_multiplicity = self.nuclear_target.N * self.target.in_neutron + self.nuclear_target.Z * self.target.in_proton
        else:
            logger.error(f"Scattering regime {self.scattering_regime} not supported.")

        self.nu_projectile = nu_projectile
        self.nu_upscattered = nu_upscattered
        self.TheoryModel = TheoryModel
        self.helicity = helicity

        self.MA = self.target.mass
        self.mzprime = TheoryModel.mzprime if TheoryModel.mzprime is not None else 1e10
        self.mhprime = TheoryModel.mhprime if TheoryModel.mhprime is not None else 1e10
        self.m_ups = self.nu_upscattered.mass

        if self.helicity == "conserving":
            self.h_upscattered = -1
        elif self.helicity == "flipping":
            self.h_upscattered = +1
        else:
            logger.error(f"Error! Could not find helicity case {self.helicity}")

        self.Cij = TheoryModel.c_aj[pdg.get_lepton_index(nu_projectile), pdg.get_HNL_index(nu_upscattered)]
        self.Cji = self.Cij
        self.Vij = TheoryModel.d_aj[pdg.get_lepton_index(nu_projectile), pdg.get_HNL_index(nu_upscattered)]
        self.Vji = self.Vij
        self.Sij = TheoryModel.s_aj[pdg.get_lepton_index(nu_projectile), pdg.get_HNL_index(nu_upscattered)]
        self.Sji = self.Sij
        self.Tij = TheoryModel.t_aj[pdg.get_lepton_index(nu_projectile), pdg.get_HNL_index(nu_upscattered)]
        self.Tji = self.Tij

        ###############
        # Hadronic vertices
        if self.target.is_nucleus:
            self.Chad = TheoryModel.cprotonV * self.target.Z + TheoryModel.cneutronV * self.target.N
            self.Vhad = TheoryModel.dprotonV * self.target.Z + TheoryModel.dneutronV * self.target.N
            self.Shad = TheoryModel.dprotonS * self.target.Z + TheoryModel.dneutronS * self.target.N
        elif self.target.is_proton:
            self.Chad = TheoryModel.cprotonV
            self.Vhad = TheoryModel.dprotonV
            self.Shad = TheoryModel.dprotonS
        elif self.target.is_neutron:
            self.Chad = TheoryModel.cneutronV
            self.Vhad = TheoryModel.dneutronV
            self.Shad = TheoryModel.dneutronS

        # If three portal model, set mass-mixed vertex
        if isinstance(TheoryModel, model.ThreePortalModel):
            # mass mixed vertex
            self.Cprimehad = self.Chad * TheoryModel.epsilonZ
        else:
            self.Cprimehad = 0.0

        # Neutrino energy threshold
        self.Ethreshold = self.m_ups**2 / 2.0 / self.MA + self.m_ups

        # vectorize total cross section calculator using vegas integration
        self.vectorized_total_xsec = np.vectorize(
            self.scalar_total_xsec, excluded=["self", "diagram", "NINT", "NEVAL", "NINT_warmup", "NEVAL_warmup", "savefile_xsec", "savefile_norm"]
        )

        self.calculable_diagrams = find_calculable_diagrams(TheoryModel)

    def scalar_total_xsec(
        self,
        Enu,
        diagram="total",
        NINT=MC.NINT,
        NEVAL=MC.NEVAL,
        NINT_warmup=MC.NINT_warmup,
        NEVAL_warmup=MC.NEVAL_warmup,
        savefile_xsec=None,
        savefile_norm=None,
    ):
        # below threshold
        if Enu < (self.Ethreshold):
            return 0.0
        else:
            DIM = 1
            batch_f = integrands.UpscatteringXsec(dim=DIM, Enu=Enu, ups_case=self, diagram=diagram)
            integ = vg.Integrator(DIM * [[0.0, 1.0]])  # unit hypercube

            if savefile_norm is not None:
                # Save normalization information
                with open(savefile_norm, "w") as f:
                    json.dump(batch_f.norm, f)
            integrals = MC.run_vegas(
                batch_f, integ, adapt_to_errors=True, NINT=NINT, NEVAL=NEVAL, NINT_warmup=NINT_warmup, NEVAL_warmup=NEVAL_warmup, savestr=savefile_xsec
            )
            logger.debug("Main VEGAS run completed.")

            return integrals["diff_xsec"].mean * batch_f.norm["diff_xsec"]

    def total_xsec(
        self, Enu, diagrams=["total"], NINT=MC.NINT, NEVAL=MC.NEVAL, NINT_warmup=MC.NINT_warmup, NEVAL_warmup=MC.NEVAL_warmup, seed=None, savestr=None
    ):
        """
        Returns the total upscattering xsec for a fixed neutrino energy in cm^2
        """

        if seed is not None:
            np.random.seed(seed)

        self.Enu = Enu
        all_xsecs = 0.0
        for diagram in diagrams:
            if diagram in self.calculable_diagrams or diagram == "total":
                tot_xsec = self.vectorized_total_xsec(
                    Enu, diagram=diagram, NINT=NINT, NEVAL=NEVAL, NINT_warmup=NINT_warmup, NEVAL_warmup=NEVAL_warmup, savefile_xsec=savestr
                )
            else:
                logger.warning(f"Warning: Diagram not found. Either not implemented or misspelled. Setting tot xsec it to zero: {diagram}")
                tot_xsec = 0.0 * Enu

            #############
            # integrated xsec coverted to cm^2
            all_xsecs += tot_xsec * self.target_multiplicity
            logger.debug(f"Total cross section for {diagram} calculated.")

        return all_xsecs

    def diff_xsec_Q2(self, Enu, Q2, diagrams=["total"]):
        """
        Returns the differential upscattering xsec for a fixed neutrino energy in cm^2
        """
        s = Enu * self.MA * 2 + self.MA**2
        physical = (Q2 > ps.upscattering_Q2min(Enu, self.m_ups, self.MA)) & (Q2 < ps.upscattering_Q2max(Enu, self.m_ups, self.MA))
        u = 2 * self.MA**2 + self.m_ups - s + Q2
        diff_xsecs = amps.upscattering_dxsec_dQ2([s, -Q2, u], process=self, diagrams=diagrams)
        if type(diff_xsecs) is dict:
            return {key: diff_xsecs[key] * physical for key in diff_xsecs.keys()}
        else:
            return diff_xsecs * physical * self.target_multiplicity


class FermionDileptonDecay:
    def __init__(self, nu_parent, nu_daughter, final_lepton1, final_lepton2, TheoryModel, h_parent=-1):

        self.TheoryModel = TheoryModel
        self.HNLtype = TheoryModel.HNLtype
        self.h_parent = h_parent

        self.nu_parent = nu_parent
        self.nu_daughter = nu_daughter
        self.secondaries = [final_lepton1, final_lepton2]

        # particle masses
        self.mzprime = TheoryModel.mzprime if TheoryModel.mzprime is not None else 1e10
        self.mhprime = TheoryModel.mhprime if TheoryModel.mhprime is not None else 1e10
        self.mm = final_lepton1.mass * const.MeV_to_GeV
        self.mp = final_lepton2.mass * const.MeV_to_GeV

        # Neutral lepton vertices
        if nu_daughter == pdg.nulight:
            self.Cih = np.sqrt(np.sum(np.abs(TheoryModel.c_aj[const.inds_active, pdg.get_HNL_index(nu_parent)]) ** 2))
            self.Dih = np.sqrt(np.sum(np.abs(TheoryModel.d_aj[const.inds_active, pdg.get_HNL_index(nu_parent)]) ** 2))
            self.Sih = np.sqrt(np.sum(np.abs(TheoryModel.s_aj[const.inds_active, pdg.get_HNL_index(nu_parent)]) ** 2))
            self.Tih = np.sqrt(np.sum(np.abs(TheoryModel.t_aj[const.inds_active, pdg.get_HNL_index(nu_parent)]) ** 2))
        else:
            self.Cih = TheoryModel.c_aj[pdg.get_HNL_index(nu_daughter), pdg.get_HNL_index(nu_parent)]
            self.Dih = TheoryModel.d_aj[pdg.get_HNL_index(nu_daughter), pdg.get_HNL_index(nu_parent)]
            self.Sih = TheoryModel.s_aj[pdg.get_HNL_index(nu_daughter), pdg.get_HNL_index(nu_parent)]
            self.Tih = TheoryModel.t_aj[pdg.get_HNL_index(nu_daughter), pdg.get_HNL_index(nu_parent)]

        # Charged lepton vertices
        self.Cv = TheoryModel.ceV
        self.Ca = TheoryModel.ceA
        self.Dv = TheoryModel.deV
        self.Da = TheoryModel.deA
        self.Ds = TheoryModel.deS
        self.Dp = TheoryModel.deP

        if nu_parent == pdg.neutrino4:
            self.m_parent = TheoryModel.m4
        elif nu_parent == pdg.neutrino5:
            self.m_parent = TheoryModel.m5
        elif nu_parent == pdg.neutrino6:
            self.m_parent = TheoryModel.m6
        else:
            self.m_parent = 0.0

        if nu_daughter == pdg.neutrino4:
            self.m_daughter = TheoryModel.m4
        elif nu_daughter == pdg.neutrino5:
            self.m_daughter = TheoryModel.m5
        elif nu_daughter == pdg.neutrino6:
            self.m_daughter = TheoryModel.m6
        else:
            self.m_daughter = 0.0

        # check if CC is allowed
        # CC_mixing1 = LNC, CC_mixing2 = LNV channel.
        if pdg.in_same_doublet(nu_daughter, final_lepton1):
            self.CC_mixing1 = TheoryModel.Ulep[pdg.get_lepton_index(final_lepton1), pdg.get_HNL_index(nu_parent)]
            self.CC_mixing2 = TheoryModel.Ulep[pdg.get_lepton_index(final_lepton2), pdg.get_HNL_index(nu_parent)]
        else:
            self.CC_mixing1 = 0
            self.CC_mixing2 = 0
        # Minus sign important for interference!
        self.CC_mixing2 *= -1

        if self.m_parent - self.m_daughter - self.mm - self.mp < 0:
            logger.error(f"Error! Final states are above the mass of parent particle: mass excess = {self.m_parent - self.m_daughter - self.mm - self.mp}.")
            raise ValueError("Energy not conserved.")

        # Is the mediator on shell?
        self.vector_on_shell = (
            (TheoryModel.mzprime is not None) and (self.m_parent - self.m_daughter > TheoryModel.mzprime) and (TheoryModel.mzprime > self.mm + self.mp)
        )
        self.vector_off_shell = not self.vector_on_shell

        self.scalar_on_shell = (
            (TheoryModel.mhprime is not None) and (self.m_parent - self.m_daughter > TheoryModel.mhprime) and (TheoryModel.mhprime > self.mm + self.mp)
        )
        self.scalar_off_shell = not self.scalar_on_shell

        # does it have transition magnetic moment?
        self.TMM = TheoryModel.has_TMM

    def SamplePS(
        self,
        NINT=MC.NINT,
        NEVAL=MC.NEVAL,
        NINT_warmup=MC.NINT_warmup,
        NEVAL_warmup=MC.NEVAL_warmup,
        NINT_sample=1,
        NEVAL_sample=10_000,
        savefile_norm=None,
        savefile_dec=None,
        existing_integrator=None,
    ):
        """
        Samples the phase space of the differential decay width in the rest frame of the HNL
        """
        if self.vector_on_shell and self.scalar_on_shell:
            logger.error("Vector and scalar simultaneously on shell is not implemented.")
            raise NotImplementedError("Feature not implemented.")
        elif (self.vector_off_shell and self.scalar_on_shell) or (self.vector_on_shell and self.scalar_off_shell):
            DIM = 1
        elif self.vector_off_shell and self.scalar_off_shell:
            DIM = 4
        batch_f = integrands.HNLDecay(dim=DIM, dec_case=self)
        if existing_integrator is None:
            # need to define a new integrator
            integ = vg.Integrator(DIM * [[0.0, 1.0]])  # unit hypercube

            if savefile_norm is not None:
                # Save normalization information
                with open(savefile_norm, "w") as f:
                    json.dump(batch_f.norm, f)
            # run the integrator
            MC.run_vegas(batch_f, integ, adapt_to_errors=True, NINT=NINT, NEVAL=NEVAL, NINT_warmup=NINT_warmup, NEVAL_warmup=NEVAL_warmup, savestr=savefile_dec)
            logger.debug("Main VEGAS run completed for decay sampler.")
            # Run one more time without adaptation to fix integration points to sample
            # Save the resulting integrator to a pickle file
            existing_integrator = integ
        existing_integrator(batch_f, adapt=False, nitn=NINT_sample, neval=NEVAL_sample)
        return MC.get_samples(existing_integrator, batch_f)

    def total_width(self, NINT=MC.NINT, NEVAL=MC.NEVAL, NINT_warmup=MC.NINT_warmup, NEVAL_warmup=MC.NEVAL_warmup, savefile_norm=None, savefile_dec=None):
        if self.vector_on_shell and self.scalar_on_shell:
            logger.error("Vector and scalar simultaneously on shell is not implemented.")
            raise NotImplementedError("Feature not implemented.")
        elif self.vector_on_shell and self.scalar_off_shell:
            return dr.gamma_Ni_to_Nj_V(vertex_ij=self.Dih, mi=self.m_parent, mj=self.m_daughter, mV=self.mzprime, HNLtype=self.HNLtype)
        # * dr.gamma_V_to_ell_ell(vertex=self.TheoryModel.deV, mV=self.mzprime, m_ell=self.mm)
        elif self.vector_off_shell and self.scalar_on_shell:
            return dr.gamma_Ni_to_Nj_S(vertex_ij=self.Sih, mi=self.m_parent, mj=self.m_daughter, mS=self.mhprime, HNLtype=self.HNLtype)
            # * dr.gamma_S_to_ell_ell(vertex=self.TheoryModel.deS, mS=self.mhprime, m_ell=self.mm)
        elif self.vector_off_shell and self.scalar_off_shell:

            # We need to integraate the differential cross section
            batch_f = integrands.HNLDecay(dim=4, dec_case=self)

            integ = vg.Integrator(4 * [[0.0, 1.0]])  # unit hypercube
            if savefile_norm is not None:
                # Save normalization information
                with open(savefile_norm, "w") as f:
                    json.dump(batch_f.norm, f)
            # run the integrator
            integrals = MC.run_vegas(batch_f, integ, adapt_to_errors=True, NINT=NINT, NEVAL=NEVAL, NINT_warmup=NINT_warmup, NEVAL_warmup=NEVAL_warmup)
            logger.debug("Main VEGAS run completed for decay total width calculation.")
            return integrals["diff_decay_rate_0"].mean * batch_f.norm["diff_decay_rate_0"]

    def differential_width(self, momenta):
        PN_LAB, Plepminus_LAB, Plepplus_LAB, Pnu_LAB = momenta
        # Calculate kinematics of HNL
        CosThetaPNLab = Cfv.get_cosTheta(PN_LAB)
        PhiPNLab = np.arctan2(PN_LAB.T[2], PN_LAB.T[1])
        pN_LAB = np.sqrt(PN_LAB.T[0] ** 2 - self.m_parent**2)
        if self.vector_on_shell and self.scalar_on_shell:
            logger.error("Vector and scalar simultaneously on shell is not implemented.")
            raise NotImplementedError("Feature not implemented.")
        elif self.vector_on_shell and self.scalar_off_shell:
            # Find vector boson four momentum in lab frame
            PV_LAB = Plepminus_LAB + Plepplus_LAB
            # Boost vector boson to the HNL rest frame
            PV_CM = Cfv.T(PV_LAB, -pN_LAB / PN_LAB.T[0], np.arccos(CosThetaPNLab), PhiPNLab)
            # CosTheta of vector boson in HNL rest frame
            PS = Cfv.get_cosTheta(PV_CM)
            return dr.diff_gamma_Ni_to_Nj_V(
                cost=PS, vertex_ij=self.Dih, mi=self.m_parent, mj=self.m_daughter, mV=self.mzprime, HNLtype=self.HNLtype, h=self.h_parent
            ) * dr.gamma_V_to_ell_ell(vertex=self.TheoryModel.deV, mV=self.mzprime, m_ell=self.mm)
        elif self.vector_off_shell and self.scalar_on_shell:
            # Find scalar boson four momentum in lab frame
            PS_LAB = Plepminus_LAB + Plepplus_LAB
            # Boost scalar boson to the HNL rest frame
            PS_CM = Cfv.T(PS_LAB, -pN_LAB / PN_LAB.T[0], np.arccos(CosThetaPNLab), PhiPNLab)
            # CosTheta of vector boson in HNL rest frame
            PS = Cfv.get_cosTheta(PS_CM)
            return dr.diff_gamma_Ni_to_Nj_S(
                cost=PS, vertex_ij=self.Sih, mi=self.m_parent, mj=self.m_daughter, mS=self.mhprime, HNLtype=self.HNLtype, h=self.h_parent
            ) * dr.gamma_S_to_ell_ell(vertex=self.TheoryModel.deS, mS=self.mhprime, m_ell=self.mm)
        elif self.vector_off_shell and self.scalar_off_shell:
            # Ni (k1) --> ell-(k2)  ell+(k3)  Nj(k4)

            # t = m23^2
            t = Cfv.dot4(Plepminus_LAB, Plepplus_LAB)

            # u = m24^2
            u = Cfv.dot4(Plepminus_LAB, Pnu_LAB)

            # c3 = cosine of polar angle of k3
            # Boost ell+ to HNL rest frame
            Plepplus_CM = Cfv.T(Plepplus_LAB, -pN_LAB / PN_LAB.T[0], np.arccos(CosThetaPNLab), PhiPNLab)
            c3 = Cfv.get_cosTheta(Plepplus_CM)

            # phi34 = azimuthal angle of k4 wrt k3
            # Boost Nj to HNL rest frame
            Pnu_CM = Cfv.T(Pnu_LAB, -pN_LAB / PN_LAB.T[0], np.arccos(CosThetaPNLab), PhiPNLab)
            PhiPnuCM = np.arctan2(Pnu_CM.T[2], Pnu_CM.T[1])
            PhiPlepplusCM = np.arctan2(Plepplus_CM.T[2], Plepplus_CM.T[1])
            phi34 = PhiPnuCM - PhiPlepplusCM

            m1 = self.m_parent
            m2 = self.mm
            m3 = self.mp
            m4 = self.m_daughter
            masses = np.array([m1, m2, m3, m4])

            v = np.sum(masses**2) - u - t

            return dr.diff_gamma_Ni_to_Nj_ell_ell([t, u, v, c3, phi34], self)


class MesonThreeBodyDecay:

 
    def __init__(
        self,
        meson_parent,
        final_lepton,
        final_neutrino,
        meson_scalar,
        TheoryModel,
        h_parent=-1,
    ):
        self.TheoryModel    = TheoryModel
        self.h_parent       = h_parent
 
        # particles
        self.meson_parent   = meson_parent
        self.final_lepton   = final_lepton
        self.final_neutrino = final_neutrino
        self.meson_scalar   = meson_scalar
 
        # For compatibility with SIREN/DarkNews interfaces that inspect
        self.nu_parent      = meson_parent
        self.nu_daughter    = final_neutrino
        self.secondaries    = [final_lepton, final_neutrino]
 
        # masses
        self.m_parent   = meson_parent.mass * const.MeV_to_GeV    # meson
        self.m_lepton   = final_lepton.mass * const.MeV_to_GeV    # ℓ
        self.m_nu       = 0.0                                      # massless ν
        self.m_mediator = meson_scalar.mass * const.MeV_to_GeV    # ϕ
 
        # Alias used by SIREN's PyDarkNewsDecay for the "parent" mass
        self.m_ups = self.m_parent
 
        # couplings
        self.gmu = getattr(TheoryModel, "gmu", 0.0)
 
        # Mediator type: "scalar" or "pseudoscalar"
        self.mediator_type = getattr(TheoryModel, "meson_scalar_type", "scalar")
 
        # meson decay constant and CKM mixing
        if self.meson_parent == pdg.piplus or self.meson_parent == pdg.piminus:
            self.f_M  = getattr(const, "fpi",   0.1307)   # GeV (PDG 2020)
            self.V_Mq = getattr(const, "Vud",   0.9737)   # CKM Vud
        elif self.meson_parent == pdg.Kplus or self.meson_parent == pdg.Kminus:
            self.f_M  = getattr(const, "fkaon", 0.1598)   # GeV (PDG 2020)
            self.V_Mq = getattr(const, "Vus",   0.2245)   # CKM Vus
        else:
            logger.warning(f"Unknown meson parent {meson_parent}; assuming f_M = f_π.")
            self.f_M  = getattr(const, "fpi",   0.1307)
            self.V_Mq = getattr(const, "Vud",   0.9737)
 
        self.G_F = getattr(const, "G_F", 1.16638e-5)   # GeV^{-2}
 
        # kinematic threshold
        # Minimum meson energy required for the decay to be allowed (rest frame):
        if self.m_parent < self.m_lepton + self.m_mediator:
            logger.error(
                f"MesonThreeBodyDecay: {meson_parent.name} too light to decay to "
                f"{final_lepton.name} + {meson_scalar.name}: "
                f"m_M={self.m_parent:.4f} < m_ℓ+m_ϕ={self.m_lepton+self.m_mediator:.4f} GeV"
            )
            raise ValueError("Meson lighter than final state: kinematically forbidden.")
 
        # Maximum neutrino energy in the meson rest frame (ν kinematic limit):
        self.E_nu_max = (
            self.m_parent**2 - (self.m_lepton + self.m_mediator)**2
        ) / (2.0 * self.m_parent)
 
        # Maximum mediator energy in the meson rest frame (ϕ kinematic limit):
        self.E_phi_max = (
            self.m_parent**2 + self.m_mediator**2 - self.m_lepton**2
        ) / (2.0 * self.m_parent)
        self._C2 = (self.G_F * self.f_M * self.V_Mq * self.gmu)**2 / 2.0
 
        logger.debug(
            f"MesonThreeBodyDecay: {meson_parent.name} → "
            f"{final_lepton.name} {final_neutrino.name} {meson_scalar.name}, "
            f"m_ϕ={self.m_mediator*1e3:.1f} MeV, g_μ={self.gmu:.2e}, "
            f"type={self.mediator_type}, E_ν^max={self.E_nu_max*1e3:.1f} MeV"
        )
 
    def _E_phi_limits(self, E_nu):
        """
        Kinematically allowed range of the mediator energy E_ϕ for a given
        neutrino energy E_ν  (all in meson rest frame, units GeV).
 
 
        Returns (E_phi_min, E_phi_max) or (None, None) if outside phase space.
        """
        m_M = self.m_parent
        m_l = self.m_lepton
        m_phi = self.m_mediator
 
        if E_nu < 0.0 or E_nu > self.E_nu_max:
            return None, None
 
        # Invariant mass² of the (ℓ + ϕ) system
        M_lph2 = m_M**2 - 2.0 * m_M * E_nu        # = t
        if M_lph2 < (m_l + m_phi)**2:
            return None, None
 
        M_lph = np.sqrt(M_lph2)
 
        # Kinematics of ϕ in the (ℓϕ) CM frame
        lam = (
            M_lph2 - (m_l + m_phi)**2
        ) * (
            M_lph2 - (m_l - m_phi)**2
        )
        if lam < 0.0:
            return None, None
 
        pstar_phi = np.sqrt(lam) / (2.0 * M_lph)
        E_star_phi = (M_lph2 - m_l**2 + m_phi**2) / (2.0 * M_lph)
 
        # Boost factors  (ℓϕ system moving with momentum |p_12| = E_ν in M frame)
        gamma = (m_M - E_nu) / M_lph       # Lorentz γ of the (ℓϕ) system
        beta_gamma = E_nu / M_lph           # β·γ  (|p_12| / M_lph)
 
        E_phi_max = gamma * E_star_phi + beta_gamma * pstar_phi
        E_phi_min = gamma * E_star_phi - beta_gamma * pstar_phi
 
        # Enforce E_phi ≥ m_phi and energy conservation E_l ≥ m_l
        E_phi_min = max(E_phi_min, m_phi)
        E_l = m_M - E_nu - E_phi_min
        if E_l < m_l:
            return None, None
 
        return E_phi_min, E_phi_max
 
    def _t_inv(self, E_nu):
        """t = (p_l + p_ϕ)² = m_M² - 2 m_M E_ν  [GeV²]"""
        return self.m_parent**2 - 2.0 * self.m_parent * E_nu
 
    def _u_inv(self, E_phi):
        """u = (p_ν + p_l)² = m_M² + m_ϕ² - 2 m_M E_ϕ  [GeV²]"""
        return self.m_parent**2 + self.m_mediator**2 - 2.0 * self.m_parent * E_phi
 
    def _matel_sq_scalar(self, E_nu, E_phi):
        """Spin-summed |M|² for SCALAR coupling ([Carlson-Rislow] Eq. 25)."""
        m_M = self.m_parent
        m_l = self.m_lepton
        m_phi = self.m_mediator
        t = self._t_inv(E_nu)
        u = self._u_inv(E_phi)
        D = t - m_l**2
        if D <= 0.0:
            return 0.0
        common = (t + u - m_phi**2) * t * D \
            - (t**2 - m_l**2 * m_M**2) * (t + m_l**2 - m_phi**2)
        mass_term = m_l**2 * t * (m_M**2 - t)
        T_S = 8.0 * (common + 2.0 * mass_term)
        return max(self._C2 * T_S / D**2, 0.0)
 
    def _matel_sq_pseudo(self, E_nu, E_phi):
        """Spin-summed |M|² for PSEUDOSCALAR ([Carlson-Rislow] Eq. 25, sign flip)."""
        m_M = self.m_parent
        m_l = self.m_lepton
        m_phi = self.m_mediator
        t = self._t_inv(E_nu)
        u = self._u_inv(E_phi)
        D = t - m_l**2
        if D <= 0.0:
            return 0.0
        common = (t + u - m_phi**2) * t * D \
            - (t**2 - m_l**2 * m_M**2) * (t + m_l**2 - m_phi**2)
        mass_term = m_l**2 * t * (m_M**2 - t)
        T_P = 8.0 * (common - 2.0 * mass_term)  # note: MINUS
        return max(self._C2 * T_P / D**2, 0.0)
 
    def _matel_sq(self, E_nu, E_phi):
        """Dispatch to scalar or pseudoscalar matrix element."""
        if self.mediator_type == "scalar":
            return self._matel_sq_scalar(E_nu, E_phi)
        elif self.mediator_type == "pseudoscalar":
            return self._matel_sq_pseudo(E_nu, E_phi)
        else:
            raise ValueError(f"Unknown mediator_type '{self.mediator_type}'. "
                             "Choose 'scalar' or 'pseudoscalar'.")

    def total_width(self):
        from scipy.integrate import dblquad, quad
 
        prefactor = 1.0 / (64.0 * np.pi**3 * self.m_parent)
        m_M = self.m_parent
 
        E_phi_lo = self.m_mediator
        E_phi_hi = self.E_phi_max

        def integrand_2d(E_phi, E_nu):
            lims = self._E_phi_limits(E_nu)
            if lims[0] is None:
                return 0.0
            if E_phi < lims[0] or E_phi > lims[1]:
                return 0.0
            E_l = m_M - E_nu - E_phi
            if E_l < self.m_lepton:
                return 0.0
            return self._matel_sq(E_nu, E_phi)

        result, _ = dblquad(
            integrand_2d,
            0.0, self.E_nu_max,          # outer: E_nu limits
            lambda E_nu: self._E_phi_limits(E_nu)[0] or 0.0,  # inner lower
            lambda E_nu: self._E_phi_limits(E_nu)[1] or 0.0,  # inner upper
            epsrel=1e-3
        )
        return max(result * prefactor, 0.0)
 
    def differential_width(self, momenta):
        P_meson_LAB, P_lepton_LAB, P_nu_LAB, P_phi_LAB = momenta
 
        # kinematics of the parent meson 
        CosThetaP = Cfv.get_cosTheta(P_meson_LAB)
        PhiP      = np.arctan2(P_meson_LAB.T[2], P_meson_LAB.T[1])
        p_meson   = np.sqrt(P_meson_LAB.T[0]**2 - self.m_parent**2)
 
        # boost all momenta to the meson rest frame
        beta   = p_meson / P_meson_LAB.T[0]
        P_nu_CM  = Cfv.T(P_nu_LAB,  -beta, np.arccos(CosThetaP), PhiP)
        P_phi_CM = Cfv.T(P_phi_LAB, -beta, np.arccos(CosThetaP), PhiP)
 
        E_nu  = P_nu_CM.T[0]
        E_phi = P_phi_CM.T[0]
 
        matel2 = self._matel_sq(E_nu, E_phi)
 
        return matel2 / (64.0 * np.pi**3 * self.m_parent)
 
    def SamplePS(
        self,
        NINT=MC.NINT,
        NEVAL=MC.NEVAL,
        NINT_warmup=MC.NINT_warmup,
        NEVAL_warmup=MC.NEVAL_warmup,
        NINT_sample=1,
        NEVAL_sample=10_000,
        savefile_norm=None,
        savefile_dec=None,
        existing_integrator=None,
    ):
        
        DIM = 2
        batch_f = integrands.MesonThreeBodyDecayIntegrands(dim=DIM, dec_case=self)
 
        if existing_integrator is None:
            integ = vg.Integrator(DIM * [[0.0, 1.0]])
 
            if savefile_norm is not None:
                with open(savefile_norm, "w") as f:
                    json.dump(batch_f.norm, f)
 
            MC.run_vegas(
                batch_f, integ,
                adapt_to_errors=True,
                NINT=NINT, NEVAL=NEVAL,
                NINT_warmup=NINT_warmup, NEVAL_warmup=NEVAL_warmup,
                savestr=savefile_dec,
            )
            logger.debug("Main VEGAS run completed for MesonThreeBodyDecay sampler.")
            existing_integrator = integ
 
        existing_integrator(batch_f, adapt=False, nitn=NINT_sample, neval=NEVAL_sample)
        return MC.get_samples(existing_integrator, batch_f)


class ThreePortalDecay(FermionDileptonDecay):
    """
    Decay class for HNL → ν + ℓ⁺ + ℓ⁻ using the Three-Portal model.

    This class inherits the phase space sampling, differential width,
    and total width machinery from FermionDileptonDecay but uses
    the couplings defined in ThreePortalModel.
    """

    def __init__(self, nu_parent, nu_daughter, final_lepton1, final_lepton2, TheoryModel, h_parent=-1):

        # initialize base class
        super().__init__(nu_parent, nu_daughter, final_lepton1, final_lepton2, TheoryModel, h_parent)

        # store reference to the ThreePortal model
        self.TheoryModel = TheoryModel

        # mediator masses
        self.mzprime = TheoryModel.mzprime if TheoryModel.mzprime is not None else 1e10
        self.mhprime = TheoryModel.mhprime if TheoryModel.mhprime is not None else 1e10

        # couplings from ThreePortalModel
        self.Cv = TheoryModel.ceV
        self.Ca = TheoryModel.ceA
        self.Dv = TheoryModel.deV
        self.Da = TheoryModel.deA
        self.Ds = TheoryModel.deS
        self.Dp = TheoryModel.deP

        # neutrino interaction vertices
        if nu_daughter == pdg.nulight:

            self.Cih = np.sqrt(
                np.sum(
                    np.abs(
                        TheoryModel.c_aj[const.inds_active,
                        pdg.get_HNL_index(nu_parent)]
                    ) ** 2
                )
            )

            self.Dih = np.sqrt(
                np.sum(
                    np.abs(
                        TheoryModel.d_aj[const.inds_active,
                        pdg.get_HNL_index(nu_parent)]
                    ) ** 2
                )
            )

            self.Sih = np.sqrt(
                np.sum(
                    np.abs(
                        TheoryModel.s_aj[const.inds_active,
                        pdg.get_HNL_index(nu_parent)]
                    ) ** 2
                )
            )

            self.Tih = np.sqrt(
                np.sum(
                    np.abs(
                        TheoryModel.t_aj[const.inds_active,
                        pdg.get_HNL_index(nu_parent)]
                    ) ** 2
                )
            )

        else:

            self.Cih = TheoryModel.c_aj[
                pdg.get_HNL_index(nu_daughter),
                pdg.get_HNL_index(nu_parent),
            ]

            self.Dih = TheoryModel.d_aj[
                pdg.get_HNL_index(nu_daughter),
                pdg.get_HNL_index(nu_parent),
            ]

            self.Sih = TheoryModel.s_aj[
                pdg.get_HNL_index(nu_daughter),
                pdg.get_HNL_index(nu_parent),
            ]

            self.Tih = TheoryModel.t_aj[
                pdg.get_HNL_index(nu_daughter),
                pdg.get_HNL_index(nu_parent),
            ]

        # scalar mixing
        self.theta = getattr(TheoryModel, "theta", 0.0)

        # kinetic mixing
        self.epsilon = getattr(TheoryModel, "epsilon", 0.0)

        # Z-Z' mass mixing
        self.epsilonZ = getattr(TheoryModel, "epsilonZ", 0.0)

        # dark gauge coupling
        self.gD = getattr(TheoryModel, "gD", 0.0)

        logger.debug(
            f"Initialized ThreePortalDecay with couplings: "
            f"Cih={self.Cih}, Dih={self.Dih}, Sih={self.Sih}"
        )


class FermionSinglePhotonDecay:

    def __init__(self, nu_parent, nu_daughter, TheoryModel, h_parent=-1):

        self.TheoryModel = TheoryModel
        self.HNLtype = TheoryModel.HNLtype
        self.h_parent = h_parent

        self.nu_parent = nu_parent
        self.nu_daughter = nu_daughter
        self.secondaries = [pdg.photon]

        # mass of the HNLs
        if nu_daughter == pdg.neutrino4:
            self.m_daughter = TheoryModel.m4
        elif nu_daughter == pdg.neutrino5:
            self.m_daughter = TheoryModel.m5
        elif nu_daughter == pdg.neutrino6:
            self.m_daughter = TheoryModel.m6
        else:
            self.m_daughter = 0.0

        if nu_parent == pdg.neutrino4:
            self.m_parent = TheoryModel.m4
        elif nu_parent == pdg.neutrino5:
            self.m_parent = TheoryModel.m5
        elif nu_parent == pdg.neutrino6:
            self.m_parent = TheoryModel.m6
        else:
            self.m_parent = 0.0

        # transition magnetic moment parameter
        if nu_daughter == pdg.nulight:
            # |T| = sqrt(|T_ei|^2 + |T_mui|^2 + |T_taui|^2)
            self.Tih = np.sqrt(np.sum(np.abs(self.TheoryModel.t_aj[const.inds_active, pdg.get_HNL_index(nu_parent)]) ** 2))
            self.m_daughter = 0.0
        else:
            self.Tih = self.TheoryModel.t_aj[pdg.get_HNL_index(nu_daughter), pdg.get_HNL_index(nu_parent)]

    def SamplePS(
        self,
        NINT=MC.NINT,
        NEVAL=MC.NEVAL,
        NINT_warmup=MC.NINT_warmup,
        NEVAL_warmup=MC.NEVAL_warmup,
        NINT_sample=1,
        NEVAL_sample=10_000,
        savefile_norm=None,
        savefile_dec=None,
        existing_integrator=None,
    ):
        """
        Samples the phase space of the differential decay width in the rest frame of the HNL
        """
        DIM = 1
        batch_f = integrands.HNLDecay(dim=DIM, dec_case=self)
        if existing_integrator is None:
            # need to define a new integrator
            integ = vg.Integrator(DIM * [[0.0, 1.0]])  # unit hypercube

            if savefile_norm is not None:
                # Save normalization information
                with open(savefile_norm, "w") as f:
                    json.dump(batch_f.norm, f)
            # run the integrator
            MC.run_vegas(batch_f, integ, adapt_to_errors=True, NINT=NINT, NEVAL=NEVAL, NINT_warmup=NINT_warmup, NEVAL_warmup=NEVAL_warmup)
            logger.debug("Main VEGAS run completed for decay sampler.")
            # Run one more time without adaptation to fix integration points to sample
            # Save the resulting integrator to a pickle file
            integ(batch_f, adapt=False, nitn=NINT_sample, neval=NEVAL_sample, saveall=savefile_dec)
            existing_integrator = integ
        return MC.get_samples(existing_integrator, batch_f)

    def total_width(self):
        return dr.gamma_Ni_to_Nj_gamma(vertex_ij=self.Tih, mi=self.m_parent, mj=self.m_daughter, HNLtype=self.HNLtype)

    def differential_width(self, momenta):
        PN_LAB, Pgamma_LAB = momenta
        # Calculate kinematics of HNL
        CosThetaPNLab = Cfv.get_cosTheta(PN_LAB)
        PhiPNLab = np.arctan2(PN_LAB.T[2], PN_LAB.T[1])
        pN_LAB = np.sqrt(PN_LAB.T[0] ** 2 - self.m_parent**2)
        # Boost gamma to the HNL rest frame
        Pgamma_CM = Cfv.T(Pgamma_LAB, -pN_LAB / PN_LAB.T[0], np.arccos(CosThetaPNLab), PhiPNLab)
        # PN_CM = Cfv.T(PN_LAB, -pN_LAB/PN_LAB.T[0], np.arccos(CosThetaPNLab), PhiPNLab)
        # CosTheta of gamma in HNL rest frame
        PS = Cfv.get_cosTheta(Pgamma_CM)
        return dr.diff_gamma_Ni_to_Nj_gamma(cost=PS, vertex_ij=self.Tih, mi=self.m_parent, mj=self.m_daughter, HNLtype=self.HNLtype, h=self.h_parent)


def find_calculable_diagrams(bsm_model):
    calculable_diagrams = []
    calculable_diagrams.append("NC_SQR")
    if bsm_model.has_vector_coupling:
        calculable_diagrams.append("KinMix_SQR")
        calculable_diagrams.append("KinMix_NC_inter")
    if bsm_model.is_mass_mixed:
        calculable_diagrams.append("MassMix_SQR")
        calculable_diagrams.append("MassMix_NC_inter")
        if bsm_model.has_vector_coupling:
            calculable_diagrams.append("KinMix_MassMix_inter")
    if bsm_model.has_TMM:
        calculable_diagrams.append("TMM_SQR")
    if bsm_model.has_scalar_coupling:
        calculable_diagrams.append("Scalar_SQR")
        calculable_diagrams.append("Scalar_NC_inter")
        if bsm_model.has_vector_coupling:
            calculable_diagrams.append("Scalar_KinMix_inter")
        if bsm_model.is_mass_mixed:
            calculable_diagrams.append("Scalar_MassMix_inter")
    return calculable_diagrams


def _require_siren(cls_name):
    if not _SIREN_AVAILABLE:
        raise ImportError(
            f"{cls_name} requires SIREN.  Install it with: pip install siren"
        )

def _two_body_p_cm(M, m1, m2):
    """Centre-of-mass momentum for M → m1 m2."""
    arg = (M**2 - (m1 + m2)**2) * (M**2 - (m1 - m2)**2)
    if arg <= 0.0:
        return 0.0
    return np.sqrt(arg) / (2.0 * M)


def _boost_to_lab(P_parent, p_cm, cos_theta, phi, m_daughter):

    E_parent = P_parent[0]
    p_parent = P_parent[1:]
    M_parent = np.sqrt(max(E_parent**2 - np.dot(p_parent, p_parent), 0.0))

    # Daughter in parent rest frame
    sin_theta = np.sqrt(max(1.0 - cos_theta**2, 0.0))
    E_rf      = np.sqrt(p_cm**2 + m_daughter**2)
    p_rf      = np.array([
        p_cm * sin_theta * np.cos(phi),
        p_cm * sin_theta * np.sin(phi),
        p_cm * cos_theta,
    ])

    # Boost parameters
    beta   = np.linalg.norm(p_parent) / E_parent if E_parent > 0 else 0.0
    gamma  = E_parent / M_parent if M_parent > 0 else 1.0

    if beta < 1e-12:
        return np.array([E_rf, *p_rf])

    beta_vec = p_parent / np.linalg.norm(p_parent)

    # Standard Lorentz boost
    p_parallel  = np.dot(p_rf, beta_vec)
    p_perp_vec  = p_rf - p_parallel * beta_vec

    E_lab       = gamma * (E_rf  + beta * p_parallel)
    p_par_lab   = gamma * (p_parallel + beta * E_rf)
    p_lab_vec   = p_par_lab * beta_vec + p_perp_vec

    return np.array([E_lab, *p_lab_vec])


class ChiPrimeDecay(DarkNewsDecay):
    """
    Two-body decay χ' → χ + V₁.

    Decay width (tree-level, g_D coupling):
        Γ = (g_D² / 48π) × m_χ' × λ^{3/2}(1, r_χ², r_V²)
    where r_i = m_i / m_χ' and λ is the Källén function.

    SIREN interface attributes accessed externally:
        .table_dir       — required by SaveDecayTables / pickle
    """

    def __init__(
        self,
        m_chi: float,
        m_chi_prime: float,
        m_V1: float,
        g_D: float,
        *,
        pdgid_chi_prime: int = 5918,
        pdgid_chi:       int = 5917,
        pdgid_V1:        int = 5922,
        table_dir: str       = None,
    ):
        _require_siren("ChiPrimeDecay")
        DarkNewsDecay.__init__(self)   # C++ constructor

        self.m_chi        = m_chi
        self.m_chi_prime  = m_chi_prime
        self.m_V1         = m_V1
        self.g_D          = g_D

        self.pdgid_chi_prime = pdgid_chi_prime
        self.pdgid_chi       = pdgid_chi
        self.pdgid_V1        = pdgid_V1

        self.table_dir = table_dir or "."
        os.makedirs(self.table_dir, exist_ok=True)

        # Cache analytic width
        self._total_width = self._compute_width()

    # analytic width 

    def _compute_width(self) -> float:
        """Γ(χ' → χ V₁) in GeV. (g_D²/48π) M λ^{3/2}."""
        M  = self.m_chi_prime
        m1 = self.m_chi
        m2 = self.m_V1
        p  = _two_body_p_cm(M, m1, m2)
        if p <= 0.0:
            return 0.0
        return self.g_D**2 * p**3 / (6.0 * np.pi * M**2)

    # SIREN interface 

    def GetPossibleSignatures(self):
        sig = dataclasses.InteractionSignature()
        sig.primary_type    = Particle.ParticleType(self.pdgid_chi_prime)
        sig.target_type     = Particle.ParticleType.Decay  # SIREN decay sentinel
        sig.secondary_types = [
            Particle.ParticleType(self.pdgid_chi),
            Particle.ParticleType(self.pdgid_V1),
        ]
        return [sig]

    def GetPossibleSignaturesFromParent(self, primary_type):
        """
        Called by SIREN's C++ injector at event-generation time.
        Returns the signature list only when the queried parent matches χ'.
        """
        if int(primary_type) == self.pdgid_chi_prime:
            return self.GetPossibleSignatures()
        return []

    def TotalDecayWidth(self, arg1) -> float:
        if isinstance(arg1, dataclasses.InteractionRecord):
            primary = arg1.signature.primary_type
        else:
            primary = arg1
        if int(primary) != self.pdgid_chi_prime:
            return 0.0
        return self._total_width

    def TotalDecayWidthForFinalState(self, record) -> float:
        """
        Called by SIREN's C++ injector during event generation to weight by
        branching ratio. For a single final state, equals the total width.
        """
        if int(record.signature.primary_type) != self.pdgid_chi_prime:
            return 0.0
        return self._total_width

    def DifferentialDecayWidth(self, record) -> float:
        if int(record.signature.primary_type) != self.pdgid_chi_prime:
            return 0.0
        return self._total_width / (4.0 * np.pi)

    def save_to_table(self, table_subdir=None):
        pass

    def SampleFinalState(self, record, random):
        P_parent = np.array(record.primary_momentum)
        p_cm     = _two_body_p_cm(self.m_chi_prime, self.m_chi, self.m_V1)

        cos_theta = random.Uniform(-1.0, 1.0)
        phi       = random.Uniform(0.0, 2.0 * np.pi)

        P_chi = _boost_to_lab(P_parent, p_cm,  cos_theta,        phi, self.m_chi)
        P_V1  = _boost_to_lab(P_parent, p_cm, -cos_theta, phi + np.pi, self.m_V1)

        secondaries = record.get_secondary_particle_records()
        # secondary_types order: [chi, V1]  (matches GetPossibleSignatures)
        for sec in secondaries:
            if int(sec.type) == self.pdgid_chi:
                sec.four_momentum = P_chi
                sec.mass          = self.m_chi
            elif int(sec.type) == self.pdgid_V1:
                sec.four_momentum = P_V1
                sec.mass          = self.m_V1
        return record




class DarkPhotonDecay(DarkNewsDecay):
    """
    Two-body decay V₁ → e⁻ + e⁺.

    Decay width (kinetically-mixed dark photon, QED):
        Γ = (α ε² m_V) / 3  ×  √(1 − 4m_e²/m_V²)  ×  (1 + 2m_e²/m_V²)
    For m_V ≫ m_e this reduces to  Γ ≈ α ε² m_V / 3.
    
    """

    _M_ELECTRON = 0.000511   # GeV

    def __init__(
        self,
        m_V1: float,
        epsilon: float,
        *,
        pdgid_V1: int  = 5922,
        table_dir: str = None,
    ):
        _require_siren("DarkPhotonDecay")
        DarkNewsDecay.__init__(self)

        self.m_V1     = m_V1
        self.epsilon  = epsilon
        self.pdgid_V1 = pdgid_V1

        self.table_dir = table_dir or "."
        os.makedirs(self.table_dir, exist_ok=True)

        self._total_width = self._compute_width()

    # analytic width

    def _compute_width(self) -> float:
        """Γ(V₁ → e⁻ e⁺) in GeV."""
        mV  = self.m_V1
        me  = self._M_ELECTRON
        eps = self.epsilon
        if mV < 2.0 * me:
            return 0.0
        beta = np.sqrt(max(1.0 - (2.0 * me / mV)**2, 0.0))
        return (_ALPHA * eps**2 * mV / 3.0) * beta * (1.0 + 2.0 * me**2 / mV**2)

    # SIREN interface

    def GetPossibleSignatures(self):
        sig = dataclasses.InteractionSignature()
        sig.primary_type    = Particle.ParticleType(self.pdgid_V1)
        sig.target_type     = Particle.ParticleType.Decay  # SIREN decay sentinel
        sig.secondary_types = [
            Particle.ParticleType.EMinus,
            Particle.ParticleType.EPlus,
        ]
        return [sig]

    def GetPossibleSignaturesFromParent(self, primary_type):

        if int(primary_type) == self.pdgid_V1:
            return self.GetPossibleSignatures()
        return []

    def TotalDecayWidth(self, arg1) -> float:
        if isinstance(arg1, dataclasses.InteractionRecord):
            primary = arg1.signature.primary_type
        else:
            primary = arg1
        if int(primary) != self.pdgid_V1:
            return 0.0
        return self._total_width

    def TotalDecayWidthForFinalState(self, record) -> float:

        if int(record.signature.primary_type) != self.pdgid_V1:
            return 0.0
        return self._total_width

    def DifferentialDecayWidth(self, record) -> float:
        if int(record.signature.primary_type) != self.pdgid_V1:
            return 0.0
        # Exact two-body angular distribution for V→ff̄:
        # dΓ/d(cosθ) ∝ 1 + β² cos²θ   (in the V rest frame)
        # Without access to the actual cosθ from the record, return isotropic.
        return self._total_width / (4.0 * np.pi)

    def save_to_table(self, table_subdir=None):
        """No interpolation tables for analytic two-body decay — no-op."""
        pass

    def SampleFinalState(self, record, random):
        """
        Sample V₁ → e⁻ e⁺ with exact angular distribution
        dΓ/d(cosθ) ∝ 1 + β² cos²θ  using rejection sampling.
        """
        me  = self._M_ELECTRON
        mV  = self.m_V1
        p_cm = _two_body_p_cm(mV, me, me)
        beta = p_cm / np.sqrt(p_cm**2 + me**2) if p_cm > 0 else 0.0

        # Rejection sample cosθ ~ 1 + β²cos²θ (normalised max = 1 + β²)
        while True:
            cos_theta = random.Uniform(-1.0, 1.0)
            u         = random.Uniform(0.0, 1.0 + beta**2)
            if u <= 1.0 + beta**2 * cos_theta**2:
                break

        phi = random.Uniform(0.0, 2.0 * np.pi)

        P_parent = np.array(record.primary_momentum)
        P_eminus = _boost_to_lab(P_parent, p_cm,  cos_theta,        phi, me)
        P_eplus  = _boost_to_lab(P_parent, p_cm, -cos_theta, phi + np.pi, me)

        secondaries = record.get_secondary_particle_records()
        for sec in secondaries:
            if sec.type == Particle.ParticleType.EMinus:
                sec.four_momentum = P_eminus
                sec.mass          = me
            elif sec.type == Particle.ParticleType.EPlus:
                sec.four_momentum = P_eplus
                sec.mass          = me
        return record
        
def _three_body_decay_rate_phi(m_phi: float, g_mu: float,
                                E_phi_vals: np.ndarray) -> np.ndarray:
    M   = _M_PI
    mm  = _M_MU
    mphi = m_phi

    # Overall coupling prefactor: _C2 = (G_F f_pi V_ud g_mu)^2 / 2
    _C2 = (_GF * _fpi * _Vud * g_mu)**2 / 2.0

    results = np.zeros_like(E_phi_vals, dtype=float)
    E_phi_min_global = mphi
    E_phi_max_global = (M**2 + mphi**2 - mm**2) / (2.0 * M)

    if E_phi_max_global <= E_phi_min_global:
        return results   # kinematically forbidden

    for i, Ep in enumerate(E_phi_vals):
        if Ep <= E_phi_min_global or Ep >= E_phi_max_global:
            results[i] = 0.0
            continue

        # u invariant for this E_phi: u = M^2 + m_phi^2 - 2 M E_phi
        u_val = M**2 + mphi**2 - 2.0 * M * Ep

        # Invariant mass squared of the (mu + nu) system
        # q^2 = (P - k_phi)^2 = M^2 + m_phi^2 - 2 M E_phi = u_val
        q2 = u_val
        if q2 <= mm**2:
            results[i] = 0.0
            continue

        mq = np.sqrt(q2)   # invariant mass of (mu + nu) system

        # E_mu* in the (mu+nu) CM frame
        E_mu_star = (q2 + mm**2) / (2.0 * mq)
        p_mu_star = np.sqrt(max(E_mu_star**2 - mm**2, 0.0))

        # Lorentz boost of the (mu+nu) system in the pion rest frame
        # E_{mu+nu} = M - Ep,  p_{mu+nu} = sqrt((M-Ep)^2 - q2)
        E_q_pion = M - Ep
        p_q_pion = np.sqrt(max(E_q_pion**2 - q2, 0.0))
        gamma_boost = E_q_pion / mq
        beta_gamma  = p_q_pion / mq

        E_mu_max = gamma_boost * E_mu_star + beta_gamma * p_mu_star
        E_mu_min = gamma_boost * E_mu_star - beta_gamma * p_mu_star
        E_mu_min = max(E_mu_min, mm)

        if E_mu_max <= E_mu_min:
            results[i] = 0.0
            continue

        def integrand(Emu):
            Enu = M - Ep - Emu
            if Enu < 0.0:
                return 0.0
            # Dalitz invariants
            t = M**2 - 2.0 * M * Enu          # t = (k_mu + k_phi)^2
            D = t - mm**2
            if D <= 0.0:
                return 0.0
            # u is fixed by E_phi for this outer loop, but recompute for clarity
            u = M**2 + mphi**2 - 2.0 * M * Ep
            # Carlson-Rislow Eq. 25, scalar trace (T_S)
            common = (t + u - mphi**2) * t * D \
                - (t**2 - mm**2 * M**2) * (t + mm**2 - mphi**2)
            mass_term = mm**2 * t * (M**2 - t)
            T_S = 8.0 * (common + 2.0 * mass_term)
            val = _C2 * T_S / D**2
            return max(val, 0.0)

        try:
            val, _ = _sci_int.quad(integrand, E_mu_min, E_mu_max,
                                   limit=40, epsrel=1e-3)
        except Exception:
            val = 0.0

        results[i] = max(val, 0.0)

    # PDG three-body prefactor: 1 / (64 pi^3 M)
    prefactor = 1.0 / (64.0 * np.pi**3 * M)
    return prefactor * results


def build_phi_flux(
    m_phi: float,
    g_mu: float,
    E_thresh: float,
    E_max: float = 3.0,
    n_bins: int  = 50,
    physically_normalized: bool = True,
    dat_dir: str = None,
):

    import siren

    if dat_dir is None:
        _flux_py = siren.utilities.get_resource_path(
            "fluxes/PionKaon/PionKaon-v1.0/flux.py"
        )
        dat_dir = os.path.dirname(_flux_py)

    dat_path  = os.path.join(dat_dir, "PionKaon_FHC_pion.dat")
    all_lines = open(dat_path).readlines()
    headers   = all_lines[0].strip().split()
    data      = [line.strip().split() for line in all_lines[1:] if line.strip()]

    numu_col = headers.index("numu")

    pi_E   = np.array([(float(r[0]) + float(r[1])) / 2.0 for r in data])
    pi_phi = np.array([float(r[numu_col]) / (50 * 1000 * 1e4) for r in data])
    # (50 bins × 1000 cm² area × 1e4 unit conversion — same as flux.py)


    E_phi_out = np.linspace(E_thresh, E_max, n_bins)
    phi_flux  = np.zeros(n_bins)

    E_phi_rf_max = (_M_PI**2 + m_phi**2 - _M_MU**2) / (2.0 * _M_PI)
    if E_phi_rf_max <= m_phi:
        energies = list(E_phi_out)
        flux_arr = list(phi_flux)
        return siren.distributions.TabulatedFluxDistribution(
            E_thresh, E_max, energies, flux_arr, physically_normalized
        )

    E_phi_rf = np.linspace(m_phi, E_phi_rf_max, 200)
    dGamma   = _three_body_decay_rate_phi(m_phi, g_mu, E_phi_rf)

    for E_pi, phi_pi in zip(pi_E, pi_phi):
        if phi_pi <= 0.0 or E_pi < _M_PI:
            continue
        p_pi  = np.sqrt(max(E_pi**2 - _M_PI**2, 0.0))
        gamma = E_pi / _M_PI
        beta  = p_pi / E_pi if E_pi > 0 else 0.0

        for j, (Erf, dG) in enumerate(zip(E_phi_rf, dGamma)):
            if dG <= 0.0:
                continue
            p_rf   = np.sqrt(max(Erf**2 - m_phi**2, 0.0))
            E_lo   = gamma * (Erf - beta * p_rf)
            E_hi   = gamma * (Erf + beta * p_rf)
            if E_hi <= E_thresh or E_lo >= E_max:
                continue
            if E_lo >= E_hi:
                E_lo = E_hi = gamma * Erf

            # Width of the rest-frame bin
            dErf = (E_phi_rf_max - m_phi) / 199.0
            dE_lab = max(E_hi - E_lo, 1e-9)
            density = dG * dErf / dE_lab * phi_pi   # [GeV⁻¹ cm⁻² POT⁻¹]

            for k, Eout in enumerate(E_phi_out):
                dEout = (E_max - E_thresh) / (n_bins - 1)
                if E_lo <= Eout + 0.5*dEout and Eout - 0.5*dEout <= E_hi:
                    overlap = min(Eout + 0.5*dEout, E_hi) - max(Eout - 0.5*dEout, E_lo)
                    phi_flux[k] += density * overlap

    return siren.distributions.TabulatedFluxDistribution(
        E_thresh, E_max, list(E_phi_out), list(phi_flux), physically_normalized
    )
    
    
#Meson Decay Process


# Physical constants
_GF   = 1.16638e-5   # GeV^{-2}
_Vud  = 0.9737
_fpi  = 0.1307       # GeV
_M_PI = 0.13957      # GeV
_M_MU = 0.10566      # GeV
_M_NU = 0.0
 
 

class MesonSimpleDecay(DarkNewsDecay):
    """
    Two-body SM decay  π⁺ → μ⁺ + ν_μ.

    Decay width (SM, tree-level):
        Γ = G_F² f_π² |V_ud|² m_π m_μ² (1 − m_μ²/m_π²)² / (8π)

    Phase space is isotropic in the pion rest frame, so no VEGAS
    integrator is needed.  SampleFinalState samples cos θ and φ
    uniformly and boosts the daughters to the lab frame.

    Follows the ChiPrimeDecay pattern — inherits DarkNewsDecay (C++)
    and implements the SIREN interface directly.
    """

    # SM constants
    G_F  = 1.1663788e-5   # GeV^-2
    f_pi = 0.1307          # GeV  (charged pion decay constant)
    V_ud = 0.97373         # CKM |V_ud|

    # Particle masses (GeV)
    M_PI = 0.13957039
    M_MU = 0.10565837
    M_NU = 0.0

    def __init__(self, nu_parent, nu_daughter):
        _require_siren("MesonSimpleDecay")
        DarkNewsDecay.__init__(self)      # C++ base constructor

        self.nu_parent   = nu_parent
        self.nu_daughter = nu_daughter

        # Hard-code PDG IDs — safe regardless of whether nu_parent is an
        # integer or a DarkNews particle object.
        self.pdgid_pion = 211    # π⁺
        self.pdgid_mu   = -13   # μ⁺
        self.pdgid_nu   = 14    # ν_μ

        self.secondaries = [nu_daughter]   # kept for DarkNews compatibility

        # Cache the analytic total width
        self._total_width_val = self._compute_width()

    # ------------------------------------------------------------------ #
    #  Physics                                                             #
    # ------------------------------------------------------------------ #

    def _compute_width(self):
        """Γ(π⁺ → μ⁺ ν_μ) in GeV."""
        r = (self.M_MU / self.M_PI) ** 2
        return (self.G_F**2 * self.f_pi**2 * self.V_ud**2
                * self.M_PI * self.M_MU**2 * (1.0 - r)**2
                / (8.0 * np.pi))

    def total_width(self):
        return self._total_width_val

    def differential_width(self, momenta):
        # Isotropic 2-body: dΓ/dΩ = Γ / (4π)
        return self._total_width_val / (4.0 * np.pi)

    # ------------------------------------------------------------------ #
    #  SIREN interface                                                     #
    # ------------------------------------------------------------------ #

    def GetPossibleSignatures(self):
        sig = dataclasses.InteractionSignature()
        sig.primary_type    = Particle.ParticleType(self.pdgid_pion)
        sig.target_type     = Particle.ParticleType.Decay
        sig.secondary_types = [
            Particle.ParticleType(self.pdgid_mu),
            Particle.ParticleType(self.pdgid_nu),
        ]
        return [sig]

    def GetPossibleSignaturesFromParent(self, primary_type):
        if int(primary_type) == self.pdgid_pion:
            return self.GetPossibleSignatures()
        return []

    def TotalDecayWidth(self, arg1):
        if isinstance(arg1, dataclasses.InteractionRecord):
            primary = arg1.signature.primary_type
        else:
            primary = arg1
        return self._total_width_val if int(primary) == self.pdgid_pion else 0.0

    def TotalDecayWidthForFinalState(self, record):
        if int(record.signature.primary_type) != self.pdgid_pion:
            return 0.0
        return self._total_width_val

    def DifferentialDecayWidth(self, record):
        if int(record.signature.primary_type) != self.pdgid_pion:
            return 0.0
        return self._total_width_val / (4.0 * np.pi)

    def save_to_table(self, table_subdir=None):
        pass   # no tables needed for analytic sampling

    def SampleFinalState(self, record, random):
        """
        Sample π⁺ → μ⁺ + ν_μ isotropically in the pion rest frame,
        then boost both daughters to the lab frame.
        """
        P_parent = np.array(record.primary_momentum)
        p_cm     = _two_body_p_cm(self.M_PI, self.M_MU, self.M_NU)

        cos_theta = random.Uniform(-1.0, 1.0)
        phi       = random.Uniform(0.0, 2.0 * np.pi)

        # μ⁺ and ν_μ are back-to-back in the rest frame
        P_mu = _boost_to_lab(P_parent, p_cm,  cos_theta,        phi,          self.M_MU)
        P_nu = _boost_to_lab(P_parent, p_cm, -cos_theta, phi + np.pi,         self.M_NU)

        for sec in record.get_secondary_particle_records():
            if int(sec.type) == self.pdgid_mu:
                sec.four_momentum = P_mu
                sec.mass          = self.M_MU
            elif int(sec.type) == self.pdgid_nu:
                sec.four_momentum = P_nu
                sec.mass          = self.M_NU
        return record
