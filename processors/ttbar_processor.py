import json
import pickle
import numpy as np
import pandas as pd
import awkward as ak
from typing import List
from coffea import processor
from coffea.analysis_tools import Weights, PackedSelection
from .corrections import add_pileup_weight, BTagCorrector, add_lepton_weights
from .utils import normalize, pad_val, build_p4, ak_to_pandas, save_output


class TTBarControlRegionProcessor(processor.ProcessorABC):
    def __init__(
        self,
        year: str = "2017",
        yearmod: str = "",
        channel: str = "ele",
        output_location: str = "",
        dir_name: str = "",
    ):
        self._year = year
        self._yearmod = yearmod
        self._channel = channel
        self._output_location = output_location
        self._dir_name = dir_name

        # open triggers
        with open("/home/cms-jovyan/wprime_plus_b/data/triggers.json", "r") as f:
            self._triggers = json.load(f)[self._year]
            
        # open btagDeepFlavB
        with open("/home/cms-jovyan/wprime_plus_b/data/btagDeepFlavB.json", "r") as f:
            self._btagDeepFlavB = json.load(f)[self._year]
            
        # open met filters
        # https://twiki.cern.ch/twiki/bin/view/CMS/MissingETOptionalFiltersRun2
        with open(
            "/home/cms-jovyan/wprime_plus_b/data/metfilters.json", "rb"
        ) as handle:
            self._metfilters = json.load(handle)[self._year]
            
        # open lumi masks
        with open("/home/cms-jovyan/wprime_plus_b/data/lumi_masks.pkl", "rb") as handle:
            self._lumi_mask = pickle.load(handle)
        
        # dataset names per year
        if year == "2018":
            self.dataset_per_ch = {
                "ele": "EGamma",
                "mu": "SingleMuon",
            }
        else:
            self.dataset_per_ch = {
                "ele": "SingleElectron",
                "mu": "SingleMuon",
            }
        
        # lepton and common weights
        self.lepton_weights = {
            "ele": ["electronID_wp80noiso", "electronRecoAbove20"],
            "mu": ["muonId", "muonIso", "muonTriggerIso"]
        }
        self.common_weights = ["genweight", "L1Prefiring", "pileup", "btagSF"]

    def add_selection(
        self,
        name: str,
        sel: ak.Array,
    ) -> None:
        """
        Adds selection to PackedSelection object and the cutflow dictionary

        taken from: github.com/cmantill/boostedhiggs/blob/main/boostedhiggs/hwwprocessor.py
        """
        self.selections.add(name, sel)
        selection = self.selections.all(*self.selections.names)
        if self.isMC:
            weight = self.weights.partial_weight(
                self.lepton_weights[self._channel] + self.common_weights
            )
            self.cutflows[name] = float(weight[selection].sum())
        else:
            self.cutflows[name] = np.sum(selection)

    @property
    def accumulator(self):
        return self._accumulator

    def process(self, events):
        dataset = events.metadata["dataset"]
        nevents = len(events)

        self.isMC = hasattr(events, "genWeight")
        self.weights = Weights(nevents, storeIndividual=True)
        self.selections = PackedSelection()
        self.cutflows = {}

        # luminosity
        if not self.isMC:
            lumi_mask = self._lumi_mask[self._year](events.run, events.luminosityBlock)
        else:
            lumi_mask = np.ones(len(events), dtype="bool")
            
        # MET filters
        metfilters = np.ones(nevents, dtype="bool")
        metfilterkey = "mc" if self.isMC else "data"
        for mf in self._metfilters[metfilterkey]:
            if mf in events.Flag.fields:
                metfilters = metfilters & events.Flag[mf]
                
        # triggers
        trigger = {}
        trigger[self._channel] = np.zeros(nevents, dtype="bool")
        for t in self._triggers[self._channel]:
            if t in events.HLT.fields:
                trigger[self._channel] = trigger[self._channel] | events.HLT[t]
                
        # taus
        deep_tau_ele = (
            (events.Tau.idDeepTau2017v2p1VSjet > 8)
            & (events.Tau.idDeepTau2017v2p1VSe > 1)
            & (np.abs(events.Tau.eta) < 2.3)
            & (events.Tau.pt > 20)
            & (events.Tau.dz < 0.2)
        )
        deep_tau_mu = (
            (events.Tau.idDeepTau2017v2p1VSjet > 8)
            & (events.Tau.idDeepTau2017v2p1VSmu > 1)
            & (np.abs(events.Tau.eta) < 2.3)
            & (events.Tau.pt > 20)
            & (events.Tau.dz < 0.2)
        )
        n_deep_tau_ele = ak.sum(deep_tau_ele, axis=1)
        n_deep_tau_mu = ak.sum(deep_tau_mu, axis=1)

        # electrons
        good_electrons = (
            (events.Electron.pt > 30)
            & (np.abs(events.Electron.eta) < 2.4)
            & (
                (np.abs(events.Electron.eta) < 1.44)
                | (np.abs(events.Electron.eta) > 1.57)
            )
            & (
                events.Electron.cutBased_HEEP
                if self._channel == "ele"
                else events.Electron.cutBased_HEEP
            )
        )
        n_good_electrons = ak.sum(good_electrons, axis=1)

        # muons
        # mediumId OR tightId?
        # DO WE NEED LOOSE MUONS?
        good_muons = (
            (events.Muon.pt > 30)
            & (np.abs(events.Muon.eta) < 2.4)
            & (events.Muon.mediumId if self._channel == "ele" else events.Muon.tightId)
        )
        n_good_muons = ak.sum(good_muons, axis=1)

        # get candidate lepton
        if self._channel == "ele":
            candidatelep = events.Electron[good_electrons]
        if self._channel == "mu":
            candidatelep = events.Muon[good_muons]
        candidatelep = ak.firsts(candidatelep)
        candidatelep_p4 = build_p4(candidatelep)

        # reliso for candidate lepton
        lep_reliso = (
            candidatelep.pfRelIso04_all
            if hasattr(candidatelep, "pfRelIso04_all")
            else candidatelep.pfRelIso03_all
        )
        # miniso for candidate lepton
        lep_miso = candidatelep.miniPFRelIso_all

        # b-jets
        good_bjets = (
            (events.Jet.pt > 30)
            & (events.Jet.jetId == 6)
            & (events.Jet.puId == 7)
            & (events.Jet.btagDeepFlavB > self._btagDeepFlavB)
        )
        n_good_bjets = ak.sum(good_bjets, axis=1)
        candidatebjet = ak.firsts(events.Jet[good_bjets])

        # lepton-bjet delta R
        lep_bjet_dr = candidatebjet.delta_r(candidatelep_p4)

        # MET
        met = events.MET

        # lepton-MET transverse mass
        mt_lep_met = np.sqrt(
            2.0
            * candidatelep_p4.pt
            * met.pt
            * (ak.ones_like(met.pt) - np.cos(candidatelep_p4.delta_phi(met)))
        )

        # output tuple variables
        variables = {
            "lep": {
                "lep_pt": candidatelep.pt,
                "lep_reliso": lep_reliso,
                "lep_miniso": lep_miso,
                "lep_met_mt": mt_lep_met,
                "lep_bjet_dr": lep_bjet_dr,
            },
            "common": {
                "met": met.pt,
                "met_phi": met.phi,
                "jet_pt": candidatebjet.pt,
                "jet_eta": candidatebjet.eta,
                "jet_phi": candidatebjet.phi,
            },
            self._channel: {},
        }

        # weights
        weigths = {}
        if self.isMC:
            # genweight
            self.weights.add("genweight", events.genWeight)
            # L1prefiring
            if self._year in ("2016", "2017"):
                self.weights.add(
                    "L1Prefiring",
                    weight=events.L1PreFiringWeight.Nom,
                    weightUp=events.L1PreFiringWeight.Up,
                    weightDown=events.L1PreFiringWeight.Dn,
                )
            # pileup
            add_pileup_weight(
                weights=self.weights,
                year=self._year,
                mod=self._yearmod,
                nPU=ak.to_numpy(events.Pileup.nPU),
            )
            # b-tagging
            self._btagSF = BTagCorrector(
                wp="M", tagger="deepJet", year=self._year, mod=self._yearmod
            )
            self._btagSF.add_btag_weight(
                jets=events.Jet[good_bjets], weights=self.weights
            )
            # lepton weights
            add_lepton_weights(
                weights=self.weights,
                candidatelep=candidatelep,
                year=self._year,
                mod=self._yearmod,
                channel=self._channel,
            )
            # store common weights
            for weight in self.common_weights:
                variables["common"][f"{weight}_weight"] = self.weights.partial_weight(
                    [weight]
                )
            variables["common"]["common_weight"] = self.weights.partial_weight(
                self.common_weights
            )
            # store lepton weights
            for weight in self.lepton_weights[self._channel]:
                variables[self._channel][
                    f"{weight}_weight"
                ] = self.weights.partial_weight([weight])
            variables[self._channel]["lepton_weigth"] = self.weights.partial_weight(
                self.lepton_weights[self._channel]
            )
        # selections
        one_lepton = {
            "ele": (n_good_muons == 0) & (n_good_electrons == 1),
            "mu": (n_good_muons == 1) & (n_good_electrons == 0),
        }
        self.add_selection("trigger", trigger[self._channel])
        self.add_selection("lumi", lumi_mask)
        self.add_selection("metfilters", metfilters)
        self.add_selection("two_bjets", n_good_bjets >= 2)
        self.add_selection("one_lepton", one_lepton[self._channel])
        self.add_selection("deltaR", lep_bjet_dr > 0.4)
        self.add_selection("notaus", n_deep_tau_ele == 0)

        # output parquets
        output = {}
        fill_output = True
        # for data, only fill output for the dataset needed
        if not self.isMC and self.dataset_per_ch[self._channel] not in dataset:
            fill_output = False
        selection = self.selections.all(*self.selections.names)
        # only fill output for that channel if the selections yield any events
        if np.sum(selection) <= 0:
            fill_output = False
        if fill_output:
            out = {}
            for key in variables:
                for var, item in variables[key].items():
                    # pad all the variables that are not a cut with -1
                    pad_item = (
                        item if ("cut" in var or "weight" in var) else pad_val(item, -1)
                    )
                    # fill out dictionary
                    out[var] = item
            # fill the output dictionary after selections
            output = {key: value[selection] for (key, value) in out.items()}
        else:
            output = {}
        # convert arrays to pandas
        if not isinstance(output, pd.DataFrame):
            output = ak_to_pandas(output)
            
        # now save pandas dataframe
        save_output(
            events,
            dataset,
            output,
            self._year,
            self._channel,
            self._output_location,
            self._dir_name,
        )

        # return dictionary with cutflows
        return {
            dataset: {
                "mc": self.isMC,
                self._year: {
                    "sumgenweight": ak.sum(events.genWeight) if self.isMC else 0,
                    "cutflows": self.cutflows,
                },
            }
        }

    def postprocess(self, accumulator):
        return accumulator