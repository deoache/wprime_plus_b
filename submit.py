import json
import time
import dask
import pickle
import argparse
import datetime
import numpy as np
import wprime_plus_b.utils
from pathlib import Path
from coffea import processor
from utils import get_filesets
from dask.distributed import Client
from humanfriendly import format_timespan
from distributed.diagnostics.plugin import UploadDirectory
from wprime_plus_b.utils import paths
from wprime_plus_b.processors.trigger_efficiency_processor import (
    TriggerEfficiencyProcessor,
)
from wprime_plus_b.processors.btag_efficiency_processor import BTagEfficiencyProcessor
from wprime_plus_b.processors.ttbar_analysis import TtbarAnalysis
from wprime_plus_b.processors.ztoll_processor import ZToLLProcessor
from wprime_plus_b.processors.qcd_analysis import QcdAnalysis
from wprime_plus_b.selections.ttbar.config import (
    ttbar_electron_selection,
    ttbar_muon_selection,
    ttbar_jet_selection,
    ttbar_tau_selection
)
from wprime_plus_b.selections.ztoll.config import (
    ztoll_electron_selection,
    ztoll_muon_selection,
    ztoll_jet_selection,
)
from wprime_plus_b.selections.qcd.config import (
    qcd_electron_selection,
    qcd_muon_selection,
    qcd_jet_selection,
    qcd_tau_selection
)


def main(args):
    args = vars(args)
    # define processors and executors
    processors = {
        "ttbar": TtbarAnalysis,
        "ztoll": ZToLLProcessor,
        "qcd": QcdAnalysis,
        "btag_eff": BTagEfficiencyProcessor,
        "trigger_eff": TriggerEfficiencyProcessor,
    }
    processor_args = [
        "year",
        "yearmod",
        "channel",
        "lepton_flavor",
        "output_type",
        "syst",
    ]
    processor_kwargs = {k: args[k] for k in processor_args if args[k]}
    executors = {
        "iterative": processor.iterative_executor,
        "futures": processor.futures_executor,
        "dask": processor.dask_executor,
    }
    executor_args = {
        "schema": processor.NanoAODSchema,
    }
    if args["executor"] == "futures":
        executor_args.update({"workers": args["workers"]})
    if args["executor"] == "dask":
        client = Client("tls://localhost:8786")
        executor_args.update({"client": client})
        # upload local directory to dask workers
        try:
            client.register_worker_plugin(
                UploadDirectory(f"{Path.cwd()}", restart=True, update_path=True),
                nanny=True,
            )
            print(f"Uploaded {Path.cwd()} succesfully")
        except OSError:
            print("Failed to upload the directory")
        
    # get .json filesets for sample
    filesets = get_filesets(
        sample=args["sample"],
        year=args["year"] + args["yearmod"],
        facility=args["facility"],
    )
    for sample, fileset_path in filesets.items():
        if len(args["nsample"]) != 0:
            samples_keys = args["nsample"].split(",")
            if sample.split("_")[-1] not in samples_keys:
                continue
        print(f"Processing {sample}")
        fileset = {}
        with open(fileset_path, "r") as handle:
            data = json.load(handle)
        for root_file in data.values():
            if args["nfiles"] != -1:
                root_file = root_file[: args["nfiles"]]
                
        if sample.startswith("Signal"):
            fileset[sample] = [f"root://eoscms.cern.ch//eos/cms/" + file for file in root_file]
            
        elif args["facility"] == "coffea-casa":
            fileset[sample] = [f"root://xcache/" + file for file in root_file]
        else:
            fileset[sample] = root_file
            
        # run processor
        t0 = time.monotonic()
        out = processor.run_uproot_job(
            fileset,
            treename="Events",
            processor_instance=processors[args["processor"]](**processor_kwargs),
            executor=executors[args["executor"]],
            executor_args=executor_args,
        )
        exec_time = format_timespan(time.monotonic() - t0)

        # get metadata
        metadata = {"walltime": exec_time}
        metadata.update({"fileset": fileset[sample]})
        if "metadata" in out[sample]:
            output_metadata = out[sample]["metadata"]
            # save number of raw initial events
            metadata.update({"raw_initial_nevents": float(output_metadata["raw_initial_nevents"])})
            # save number of weighted initial events
            if args["processor"] == "qcd":
                if args["channel"] != "all":
                    metadata.update({"sumw": float(output_metadata[args["channel"]]["sumw"])})
                else:
                    sumws = {}
                    for r in ["A", "B", "C", "D"]:
                        sumws[r] = float(output_metadata[r]["sumw"])
                    metadata.update({"sumw": sumws})
            else:
                metadata.update({"sumw": float(output_metadata["sumw"])})
            # save qcd metadata
            if args["processor"] in ["qcd"]:
                metadata.update({"nevents": {}})
                region = args["channel"]
                if region != "all":
                    metadata["nevents"].update({region: {}})
                    metadata["nevents"][region]["raw_final_nevents"] = str(
                        output_metadata[region]["raw_final_nevents"]
                    )
                    metadata["nevents"][region]["weighted_final_nevents"] = str(
                        output_metadata[region]["weighted_final_nevents"]
                    )
                elif region == "all":
                    for r in ["A", "B", "C", "D"]:
                        metadata["nevents"].update({r: {}})
                        metadata["nevents"][r]["raw_final_nevents"] = str(
                            output_metadata[r]["raw_final_nevents"]
                        )
                        metadata["nevents"][r]["weighted_final_nevents"] = str(
                            output_metadata[r]["weighted_final_nevents"]
                        )
                        
            # save ttbar and ztoll metadata
            if args["processor"] in ["ttbar", "ztoll"]:
                # save raw and weighted number of events after selection
                if "raw_final_nevents" in output_metadata:
                    metadata.update(
                        {"raw_final_nevents": float(output_metadata["raw_final_nevents"])}
                    )
                    metadata.update(
                        {"weighted_final_nevents": float(output_metadata["weighted_final_nevents"])}
                    )
                else:
                    metadata.update(
                        {"raw_final_nevents": 0.}
                    )
                    metadata.update(
                        {"weighted_final_nevents": 0.}
                    )
                # save cutflow to metadata
                for cut_selection, nevents in output_metadata["cutflow"].items():
                    output_metadata["cutflow"][cut_selection] = str(nevents)
                metadata.update({"cutflow": output_metadata["cutflow"]})

                for weight, statistics in output_metadata["weight_statistics"].items():
                    output_metadata["weight_statistics"][weight] = str(statistics)
                metadata.update(
                    {"weight_statistics": output_metadata["weight_statistics"]}
                )
            # save selectios to metadata
            if args["processor"] == "ttbar": 
                selections = {
                    "electron_selection": ttbar_electron_selection[args["channel"]][
                        args["lepton_flavor"]
                    ],
                    "muon_selection": ttbar_muon_selection[args["channel"]][
                        args["lepton_flavor"]
                    ],
                    "jet_selection": ttbar_jet_selection[args["channel"]][
                        args["lepton_flavor"]
                    ],
                    "tau_selection": ttbar_tau_selection[args["channel"]][
                        args["lepton_flavor"]
                    ]
                }
                metadata.update({"selections": selections})
            elif args["processor"] == "ztoll":
                selections = {
                    "electron_selection": ztoll_electron_selection,
                    "muon_selection": ztoll_muon_selection,
                    "jet_selection": ztoll_jet_selection,
                }
                metadata.update({"selections": selections})
            elif args["processor"] == "qcd":  
                region = args["channel"]
                if region != "all":
                    selections = {
                        "electron_selection": qcd_electron_selection[region][args["lepton_flavor"]],
                        "muon_selection": qcd_muon_selection[region][args["lepton_flavor"]],
                        "jet_selection": qcd_jet_selection[region][args["lepton_flavor"]],
                        "tau_selection": qcd_tau_selection[region][args["lepton_flavor"]],
                    }
                    metadata.update({"selections": selections})
                elif region == "all":
                    selections = {}
                    for r in ["A", "B", "C", "D"]:
                        selections[r] = {
                            "electron_selection": qcd_electron_selection[r][args["lepton_flavor"]],
                            "muon_selection": qcd_muon_selection[r][args["lepton_flavor"]],
                            "jet_selection": qcd_jet_selection[r][args["lepton_flavor"]],
                            "tau_selection": qcd_tau_selection[r][args["lepton_flavor"]],
                        }
                        metadata.update({"selections": selections})
        # save args to metadata
        args_dict = args.copy()
        metadata.update(args_dict)
        del out[sample]["metadata"]
        # save output data and metadata
        with open(f"{args['output_path']}/metadata/{sample}_metadata.json", "w") as f:
            f.write(json.dumps(metadata))
        with open(f"{args['output_path']}/{sample}.pkl", "wb") as handle:
            pickle.dump(out, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--processor",
        dest="processor",
        type=str,
        default="",
        help="processor to be used {ttbar, ztoll, qcd, trigger_eff, btag_eff} (default ttbar)",
    )
    parser.add_argument(
        "--channel",
        dest="channel",
        type=str,
        default="",
        help="channel to be processed",
    )
    parser.add_argument(
        "--lepton_flavor",
        dest="lepton_flavor",
        type=str,
        default="",
        help="lepton flavor to be processed {'mu', 'ele'}",
    )
    parser.add_argument(
        "--sample",
        dest="sample",
        type=str,
        default="",
        help="sample key to be processed",
    )
    parser.add_argument(
        "--year",
        dest="year",
        type=str,
        default="",
        help="year of the data {2016, 2017, 2018} (default 2017)",
    )
    parser.add_argument(
        "--yearmod",
        dest="yearmod",
        type=str,
        default="",
        help="year modifier {'', 'APV'} (default '')",
    )
    parser.add_argument(
        "--executor",
        dest="executor",
        type=str,
        default="",
        help="executor to be used {iterative, futures, dask} (default iterative)",
    )
    parser.add_argument(
        "--workers",
        dest="workers",
        type=int,
        default=4,
        help="number of workers to use with futures executor (default 4)",
    )
    parser.add_argument(
        "--nfiles",
        dest="nfiles",
        type=int,
        default=1,
        help="number of .root files to be processed by sample. To run all files use -1 (default 1)",
    )
    parser.add_argument(
        "--nsample",
        dest="nsample",
        type=str,
        default="",
        help="partitions to run (--nsample 1,2,3 will only run partitions 1,2 and 3)",
    )
    parser.add_argument(
        "--chunksize",
        dest="chunksize",
        type=int,
        default=50000,
        help="number of chunks to process",
    )
    parser.add_argument(
        "--output_type",
        dest="output_type",
        type=str,
        default="",
        help="type of output {hist, array}",
    )
    parser.add_argument(
        "--syst",
        dest="syst",
        type=str,
        default="",
        help="systematic to apply {'nominal', 'jet', 'met', 'full'}",
    )
    parser.add_argument(
        "--facility",
        dest="facility",
        type=str,
        default="",
        help="facility to launch jobs {coffea-casa, lxplus}",
    )
    parser.add_argument(
        "--tag",
        dest="tag",
        type=str,
        default="",
        help="tag to reference output files directory",
    )
    parser.add_argument(
        "--output_path",
        dest="output_path",
        type=str,
        default="",
        help="output path directory",
    )
    args = parser.parse_args()
    main(args)