import os
import sys
import argparse

env_path = os.path.join(os.path.dirname(__file__), '..')
if env_path not in sys.path:
    sys.path.append(env_path)

from pytracking.evaluation.otbdataset import OTBDataset
from pytracking.evaluation.nfsdataset import NFSDataset
from pytracking.evaluation.uavdataset import UAVDataset
from pytracking.evaluation.votdataset import VOTDataset
from pytracking.evaluation.lasotdataset import LaSOTDataset
from pytracking.evaluation.combdataset import CombDataset
from pytracking.evaluation.trackingnetdataset import TrackingNetDataset
from pytracking.evaluation.got10kdataset import GOT10KDatasetTest, GOT10KDatasetVal, GOT10KDatasetLTRVal
from pytracking.evaluation.running import run_dataset, run_vot
from pytracking.evaluation import Tracker


def run_tracker(tracker_name, tracker_param, run_id=None, dataset_name='otb', sequence=None, debug=0, threads=0,
                visdom_info=None, delimiter=None, tracker_params=None):
    """Run tracker on sequence or dataset.
    args:
        tracker_name: Name of tracking method.
        tracker_param: Name of parameter file.
        run_id: The run id.
        dataset_name: Name of dataset (otb, nfs, uav, tpl, vot, tn, gott, gotv, lasot).
        sequence: Sequence number or name.
        debug: Debug level.
        threads: Number of threads.
        visdom_info: Dict optionally containing 'use_visdom', 'server' and 'port' for Visdom visualization.
    """

    visdom_info = {} if visdom_info is None else visdom_info

    if dataset_name == 'otb':
        dataset = OTBDataset()
    elif dataset_name == 'nfs':
        dataset = NFSDataset()
    elif dataset_name == 'uav':
        dataset = UAVDataset()
    elif dataset_name == 'vot':
        dataset = VOTDataset()
    elif dataset_name == 'tn':
        dataset = TrackingNetDataset()
    elif dataset_name == 'gott':
        dataset = GOT10KDatasetTest()
    elif dataset_name == 'gotv':
        dataset = GOT10KDatasetVal()
    elif dataset_name == 'gotlv':
        dataset = GOT10KDatasetLTRVal()
    elif dataset_name == 'lasot':
        dataset = LaSOTDataset()
    elif dataset_name == 'comb':
        dataset = CombDataset()
    else:
        raise ValueError('Unknown dataset name')

    if sequence is not None:
        dataset = [dataset[sequence]]

    trackers = [Tracker(tracker_name, tracker_param, run_id, tracker_params)]

    if dataset_name != 'vot':
        run_dataset(dataset, trackers, debug, threads, visdom_info=visdom_info, delimiter=delimiter)
    else:
        run_vot(dataset, trackers, debug, threads, visdom_info=visdom_info)

def main():
    parser = argparse.ArgumentParser(description='Run tracker on sequence or dataset.')
    parser.add_argument('tracker_name', type=str, help='Name of tracking method.')
    parser.add_argument('tracker_param', type=str, help='Name of parameter file.')
    parser.add_argument('--runid', type=str, default=None, help='The run id.')
    parser.add_argument('--dataset', type=str, default='otb', help='Name of dataset (otb, nfs, uav, tpl, vot, tn, gott, gotv, lasot).')
    parser.add_argument('--sequence', type=str, default=None, help='Sequence number or name.')
    parser.add_argument('--debug', type=int, default=0, help='Debug level.')
    parser.add_argument('--threads', type=int, default=0, help='Number of threads.')
    parser.add_argument('--use_visdom', type=bool, default=True, help='Flag to enable visdom')
    parser.add_argument('--visdom_server', type=str, default='127.0.0.1', help='Server for visdom')
    parser.add_argument('--visdom_port', type=int, default=8097, help='Port for visdom')
    parser.add_argument('--delimiter', type=str, default='\t', help='Type for saving tracking results.')
    parser.add_argument('--cuda_id', type=str, default='0', help='Specific cuda to use for running tracker.')

    # tracker params
    # main params for fcot-tracker
    parser.add_argument('--params__search_area_scale', type=float, default=None)
    parser.add_argument('--params__sample_memory_size', type=int, default=None)
    parser.add_argument('--params__learning_rate', type=float, default=None)
    parser.add_argument('--params__init_samples_minimum_weight', type=float, default=None)
    parser.add_argument('--params__train_skipping', type=int, default=None)
    parser.add_argument('--params__init_train_frames', type=int, default=None)
    parser.add_argument('--params__init_train_iter', type=int, default=None)
    parser.add_argument('--params__net_opt_iter', type=int, default=None)
    parser.add_argument('--params__net_opt_update_iter', type=int, default=None)
    parser.add_argument('--params__net_opt_hn_iter', type=int, default=None)
    parser.add_argument('--params__init_train_iter_72', type=int, default=None)
    parser.add_argument('--params__net_opt_iter_72', type=int, default=None)
    parser.add_argument('--params__net_opt_update_iter_72', type=int, default=None)
    parser.add_argument('--params__net_opt_hn_iter_72', type=int, default=None)
    parser.add_argument('--params__reg_init_train_iter', type=int, default=None)
    parser.add_argument('--params__reg_net_opt_iter', type=int, default=None)
    parser.add_argument('--params__reg_net_opt_hn_iter', type=int, default=None)
    parser.add_argument('--params__reg_net_opt_update_iter', type=int, default=None)
    parser.add_argument('--params__lamda_72', type=float, default=None)
    parser.add_argument('--params__lamda_18', type=float, default=None)
    parser.add_argument('--params__reg_lamda', type=float, default=None)
    parser.add_argument('--params__merge_rate_72', type=float, default=None)
    parser.add_argument('--params__merge_rate_18', type=float, default=None)
    parser.add_argument('--params__hard_negative_threshold', type=float, default=None)
    parser.add_argument('--params__hard_negative_learning_rate', type=float, default=None)
    parser.add_argument('--params__target_not_found_threshold', type=float, default=None)
    parser.add_argument('--params__reg_mem_size', type=float, default=None)
    parser.add_argument('--params__epoch', type=int, default=None)
    parser.add_argument('--params__han', type=float, default=None)
    parser.add_argument('--params__target_neighborhood_scale', type=float, default=None)
    parser.add_argument('--params__distractor_threshold', type=float, default=None)
    parser.add_argument('--params__lamda_cls', type=float, default=None)
    parser.add_argument('--params__reg_init_memory', type=int, default=None)
    parser.add_argument('--params__reg_train_skipping', type=int, default=None)

    args = parser.parse_args()

    tracker_params = {}
    for param in list(filter(lambda s: s.split('__')[0] == 'params' and getattr(args, s) != None, args.__dir__())):
        tracker_params[param.split('__')[1]] = getattr(args, param)
    print(tracker_params)

    # set cuda environment
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_id

    run_tracker(args.tracker_name, args.tracker_param, args.runid, args.dataset, args.sequence, args.debug, args.threads,
                {'use_visdom': args.use_visdom, 'server': args.visdom_server, 'port': args.visdom_port}, args.delimiter,
                tracker_params)


if __name__ == '__main__':
    main()
