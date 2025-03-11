import correctionlib
import numpy as np
import awkward as ak
from typing import Tuple
from wprime_plus_b.corrections.utils import get_pog_json
from wprime_plus_b.corrections.jetvetomaps import jetvetomaps_mask


def apply_met_phi_corrections(
    events: ak.Array,
    is_mc: bool,
    year: str,
) -> Tuple[ak.Array, ak.Array]:
    """
    Apply MET phi modulation corrections

    Parameters:
    -----------
        events:
            Events array
        is_mc:
            True if dataset is MC
        year:
            Year of the dataset {'2016', '2016APV', '2017', '2018'}

    Returns:
    --------
        corrected MET pt and phi
    """
    cset = correctionlib.CorrectionSet.from_file(
        get_pog_json(json_name="met", year=year)
    )
    events["MET", "pt_raw"] = ak.ones_like(events.MET.pt) * events.MET.pt
    events["MET", "phi_raw"] = ak.ones_like(events.MET.phi) * events.MET.phi
    
    # make sure to not cross the maximum allowed value for uncorrected met
    met_pt = events.MET.pt_raw
    met_pt = np.clip(met_pt, 0.0, 6499.0)
    met_phi = events.MET.phi_raw
    met_phi = np.clip(met_phi, -3.5, 3.5)

    # use correct run ranges when working with data, otherwise use uniform run numbers in an arbitrary large window
    run_ranges = {
        "2016APV": [272007, 278771],
        "2016": [278769, 284045],
        "2017": [297020, 306463],
        "2018": [315252, 325274],
    }
    data_kind = "mc" if is_mc else "data"
    if data_kind == "mc":
        run = np.random.randint(
            run_ranges[year][0], run_ranges[year][1], size=len(met_pt)
        )
    else:
        run = events.run
    try:
        events["MET", "pt"] = cset[f"pt_metphicorr_pfmet_{data_kind}"].evaluate(
            met_pt.to_numpy(), met_phi.to_numpy(), events.PV.npvsGood.to_numpy(), run
        )
        events["MET", "phi"] = cset[f"phi_metphicorr_pfmet_{data_kind}"].evaluate(
            met_pt.to_numpy(), met_phi.to_numpy(), events.PV.npvsGood.to_numpy(), run
        )
    except:
        pass

    
def update_met(events: ak.Array, lepton: str = "Muon") -> None:
    """
    helper function to compute new MET after lepton pT correction. 
    It uses the 'pt_raw' and 'pt' fields from 'leptons' to update MET 'pt' and 'phi' fields
    
    Parameters:
        - events:
            Events array
        - lepton:
            Lepton name {'Muon', 'Tau'}

    https://github.com/columnflow/columnflow/blob/16d35bb2f25f62f9110a8f1089e8dc5c62b29825/columnflow/calibration/util.py#L42
    https://github.com/Katsch21/hh2bbtautau/blob/e268752454a0ce0089ff08cc6c373a353be77679/hbt/calibration/tau.py#L117
    """
    assert lepton in ["Muon", "Electron", "Tau"], "Lepton not provided"
    
    # get needed lepton and MET fields
    lepton_pt_raw = events[lepton, "pt_raw"]
    lepton_pt = events[lepton, "pt"]
    lepton_phi = events[lepton, "phi"]
    met_pt = events.MET.pt
    met_phi = events.MET.phi
    
    # build px and py sums before and after: we sum the time at x and the time at y of each event    
    old_px = ak.sum(lepton_pt_raw * np.cos(lepton_phi), axis=1)
    old_py = ak.sum(lepton_pt_raw * np.sin(lepton_phi), axis=1)
    new_px = ak.sum(lepton_pt * np.cos(lepton_phi), axis=1)
    new_py = ak.sum(lepton_pt * np.sin(lepton_phi), axis=1)

    # get x and y changes
    delta_x = new_px - old_px
    delta_y = new_py - old_py
    
    # propagate changes to MET (x, y) components
    met_px = met_pt * np.cos(met_phi) + delta_x
    met_py = met_pt * np.sin(met_phi) + delta_y
    
    # propagate changes to MET (pT, phi) components
    met_pt = np.sqrt((met_px ** 2.0 + met_py ** 2.0))
    met_phi = np.arctan2(met_py, met_px)
    
    # update MET fields
    events["MET", "pt"] = met_pt
    events["MET", "phi"] = met_phi
    
    
def update_met_jet_veto(events, year) -> None:
    """
    helper function to compute new MET after lepton pT correction.
    It uses the 'pt_raw' and 'pt' fields from 'leptons' to update MET 'pt' and 'phi' fields

    Parameters:
        - events:
            Events array
        - lepton:
            Lepton name {'Muon', 'Tau'}

    https://github.com/columnflow/columnflow/blob/16d35bb2f25f62f9110a8f1089e8dc5c62b29825/columnflow/calibration/util.py#L42
    https://github.com/Katsch21/hh2bbtautau/blob/e268752454a0ce0089ff08cc6c373a353be77679/hbt/calibration/tau.py#L117
    """
    # get jets for the veto
    jets_veto = events.Jet[jetvetomaps_mask(events.Jet, year=year)]

    # Raw MET
    met_pt = events.MET.pt
    met_phi = events.MET.phi

    # Jet veto pt(x,y) per event
    jet_veto_pt_x = jets_veto.pt * np.cos(jets_veto.phi)
    jet_veto_pt_y = jets_veto.pt * np.sin(jets_veto.phi)

    # events.Jet.pt
    jet_pt_x = events.Jet.pt * np.cos(events.Jet.phi)
    jet_pt_y = events.Jet.pt * np.sin(events.Jet.phi)

    # get x and y changes
    delta_x = ak.sum(jet_pt_x, axis=-1) - ak.sum(jet_veto_pt_x, axis=-1)
    delta_y = ak.sum(jet_pt_y, axis=-1) - ak.sum(jet_veto_pt_y, axis=-1)

    # propagate changes to MET (x, y) components:Problematic samples y TTToSemiLeptonic and SingleMuon
    met_px = met_pt * np.cos(met_phi) - delta_x
    met_py = met_pt * np.sin(met_phi) - delta_y

    # propagate changes to MET (pT, phi) components
    new_met_pt = np.sqrt((met_px**2.0 + met_py**2.0))
    new_met_phi = np.arctan2(met_py, met_px)

    # update MET fields
    events["MET", "pt"] = new_met_pt
    events["MET", "phi"] = new_met_phi
    
    
    
def update_met_tt_estimation(events, bjets):
    # all bjets pT (x,y) components
    jet_pt_x = bjets.pt * np.cos(bjets.phi)
    jet_pt_y = bjets.pt * np.sin(bjets.phi)
    
    # all bjets pT - subleading bjet pT (x,y) components
    subleading_bjets = ak.pad_none(bjets, 2)[:, 1]
    jet_minus_subjet_pt_x = jet_pt_x - subleading_bjets.pt * np.cos(subleading_bjets.phi)
    jet_minus_subjet_pt_y = jet_pt_y - subleading_bjets.pt * np.sin(subleading_bjets.phi)
    
    # compute changes from original pT (all jets pT) to new pT (all jets pT - subleading jet pT) for (x,y) components
    delta_x = ak.sum(jet_pt_x, axis=-1) - ak.sum(jet_minus_subjet_pt_x, axis=-1)
    delta_y = ak.sum(jet_pt_y, axis=-1) - ak.sum(jet_minus_subjet_pt_y, axis=-1)
    
    # propagate changes to MET (x, y) components
    met = events.MET
    met_pt = met.pt
    met_phi = met.phi
    met_px = met_pt * np.cos(met_phi) - delta_x
    met_py = met_pt * np.sin(met_phi) - delta_y

    # update MET pT and phi fields
    events["MET", "pt"] = np.sqrt((met_px ** 2.0 + met_py ** 2.0))
    events["MET", "phi"] = np.arctan2(met_py, met_px)