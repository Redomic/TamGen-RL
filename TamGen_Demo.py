# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
import os
from glob import glob
import argparse
from torch.serialization import safe_globals
from fairseq.meters import AverageMeter, StopwatchMeter, TimeMeter
from fairseq.data import Dictionary

from fairseq import bleu, checkpoint_utils, options, progress_bar, tasks, utils
from fairseq.meters import StopwatchMeter, TimeMeter
from rdkit import Chem, RDLogger
from tqdm import tqdm

RDLogger.DisableLog('rdApp.*')

OVERRIDES = ['sample_beta', 'gen_coord_noise', 'gen_rot', 'gen_vae']


def prepare_pdb_data(pdb_id, ligand_inchi=None, DemoDataFolder="TamGen_Demo_Data", thr=10):
    out_split = pdb_id.lower()
    FF = glob(f"{DemoDataFolder}/*")
    for ff in FF:
        if f"gen_{out_split}" in ff:
            print(f"{pdb_id} is downloaded")
            return
    
    os.system(f"mkdir -p {DemoDataFolder}")
    if ligand_inchi is None:
        with open("tmp_pdb.csv", "w") as fw:
            print("pdb_id", file=fw)
            print(f"{pdb_id}", file=fw)
    else:
        with open("tmp_pdb.csv", "w") as fw:
            print("pdb_id,ligand_inchi", file=fw)
            print(f"{pdb_id},{ligand_inchi}", file=fw)
    
    os.system(f"python scripts/build_data/prepare_pdb_ids.py tmp_pdb.csv gen_{out_split} -o {DemoDataFolder} -t {thr}")
    os.system(r"rm tmp_pdb.csv")


def prepare_pdb_data_center(pdb_id, scaffold_file=None, DemoDataFolder="TamGen_Demo_Data", thr=10):
    out_split = pdb_id.lower()
    FF = glob(f"{DemoDataFolder}/*")
    for ff in FF:
        if f"gen_{out_split}" in ff:
            print(f"{pdb_id} is downloaded")
            return

    with open("tmp_pdb.csv", "w") as fw:
        print("pdb_id,ligand_inchi", file=fw)
        print(f"{pdb_id},{ligand_inchi}", file=fw)
    
    os.system(f"python scripts/build_data/prepare_pdb_ids.py tmp_pdb.csv gen_{out_split} -o {DemoDataFolder} -t {thr}")
    os.system(r"rm tmp_pdb.csv")


def filter_generated_cmpd(smi):
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    sssr = Chem.GetSymmSSSR(m)
    if len(sssr) <= 1:
        return None
    if len(sssr) >= 4:
        return None
    if smi.lower().count('p') > 3:
        return None
    s = Chem.MolToSmiles(m)
    return s, m


class TamGenDemo:
    def __init__(
            self, 
            ckpt="checkpoints/crossdock_pdb_A10/checkpoint_best.pt", 
            data="TamGent_Demo_Data",
            use_conditional=True
            ):
        
        input_args = [
            data,
            "-s", "tg", "-t", "m1", "--task", "translation_coord",
            "--path",  ckpt,
            "--gen-subset", "gen_8fln", "--beam", "20", "--nbest", "20",
            "--max-tokens", "1024",
            "--seed", "1", 
            "--sample-beta", "1",
            "--use-src-coord", 
        ]  
        if use_conditional:
            self.use_conditional = True
            input_args.append("--gen-vae")
        else:
            self.use_conditional = False
        self.use_conditional = use_conditional
        parser = options.get_generation_parser()
        args = options.parse_args_and_arch(parser, input_args)

        self.args = args

        assert args.path is not None, '--path required for generation!'
        assert not args.sampling or args.nbest == args.beam, \
            '--sampling requires --nbest to be equal to --beam'
        assert args.replace_unk is None or args.raw_text, \
            '--replace-unk requires a raw text dataset (--raw-text)'

        utils.import_user_module(args)

        if args.max_tokens is None and args.max_sentences is None:
            args.max_tokens = 12000
        print(args)

        use_cuda = torch.cuda.is_available() and not args.cpu
        # Load dataset splits
        self.task = tasks.setup_task(args)

        # Set dictionaries
        try:
            self.src_dict = getattr(self.task, 'source_dictionary', None)
        except NotImplementedError:
            self.src_dict = None
        self.tgt_dict = self.task.target_dictionary

        # Set override args
        overrides = eval(args.model_overrides)
        for name in OVERRIDES:
            overrides[name] = getattr(args, name, None)

        # Load ensemble (wrapped with safe unpickling)
        print('| loading model(s) from {}'.format(args.path))
        with safe_globals([
            argparse.Namespace,
            AverageMeter,
            StopwatchMeter,
            TimeMeter,
            Dictionary,
        ]):
            self.models, _model_args = checkpoint_utils.load_model_ensemble(
                args.path.split(':'),
                arg_overrides=overrides,
                task=self.task,
            )

        for model in self.models:
            model.make_generation_fast_(
                beamable_mm_beam_size=None if args.no_beamable_mm else args.beam,
                need_attn=args.print_alignment,
            )
            if args.fp16:
                model.half()
            if use_cuda:
                model.cuda()

        self.max_position = utils.resolve_max_positions(
                self.task.max_positions(),
                *[model.max_positions() for model in self.models]
        )

        self.generator = self.task.build_generator(args)
        self.has_target = True
        
    def reload_data(self, subset=None):
        if subset is None:
            dataset = self.args.gen_subset
        else:
            dataset = subset
            self.args.gen_subset = subset
        self.task.load_dataset(dataset)
        self.itr = self.task.get_batch_iterator(
            dataset=self.task.dataset(dataset),
            max_tokens=self.args.max_tokens,
            max_sentences=self.args.max_sentences,
            max_positions=self.max_position,
            ignore_invalid_inputs=self.args.skip_invalid_size_inputs_valid_test,
            required_batch_size_multiple=self.args.required_batch_size_multiple,
            num_shards=self.args.num_shards,
            shard_id=self.args.shard_id,
            num_workers=self.args.num_workers,
        ).next_epoch_itr(shuffle=False)

    def sample(self, m_sample=50, use_cuda=True, toolcompound=None, customer_filter_fn=None, maxseed=101):
        if toolcompound is not None:
            toolcompound = Chem.MolFromSmiles(toolcompound)
        results_set = {}
        for seed in tqdm(range(1, maxseed), total=maxseed):
            if len(results_set) > m_sample:
                break
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            with progress_bar.build_progress_bar(self.args, self.itr) as t:
                for sample in t:
                    sample = utils.move_to_cuda(sample) if use_cuda else sample
                    if 'net_input' not in sample:
                        continue

                    prefix_tokens = None
                    hypos = self.task.inference_step(self.generator, self.models, sample, prefix_tokens)
                    # Extract and log latent vectors for visualization
                    for model in self.models:
                        if hasattr(model, 'encoder'):
                            encoder_out = model.encoder.forward(
                                sample['net_input']['src_tokens'],
                                sample['net_input']['src_lengths'],
                                src_coord=sample['net_input'].get('src_coord', None),
                                tgt_tokens=sample.get('target', None),
                                tgt_coord=sample['net_input'].get('tgt_coord', None),
                            )

                            if 'latent_mean' in encoder_out and encoder_out['latent_mean'] is not None:
                                z = encoder_out['latent_mean']
                                z_np = z.detach().cpu().numpy()
                                print("🔥 Logging latent vectors, shape before reduction:", z_np.shape)
                                if z_np.ndim == 3:
                                    z_np = z_np.mean(axis=1)  # Reduce fragment dimension
                                    print("✅ Reduced latent vector shape:", z_np.shape)
                                assert z_np.ndim == 2, f"Expected 2D array, got {z_np.shape}"
                                os.makedirs("latent_logs", exist_ok=True)
                                with open("latent_logs/latent_vectors.tsv", "a") as f:
                                    for zi in z_np:
                                        zi = zi.flatten()
                                        f.write("\t".join(f"{float(x):.5f}" for x in zi.flatten()) + "\n")
                                        f.flush()

                    num_generated_tokens = sum(len(h[0]['tokens']) for h in hypos)

                    for i, sample_id in enumerate(sample['id'].tolist()):
                        has_target = sample['target'] is not None
                        src_tokens = utils.strip_pad(sample['net_input']['src_tokens'][i, :], self.tgt_dict.pad())
                        target_tokens = None
                        if has_target:
                            target_tokens = utils.strip_pad(sample['target'][i, :], self.tgt_dict.pad()).int().cpu()

                        src_str = self.src_dict.string(src_tokens, self.args.remove_bpe)
                        target_str = self.tgt_dict.string(target_tokens, self.args.remove_bpe, escape_unk=True)
                     
                        if toolcompound is None:
                            tmps = target_str.strip().replace(" ", "")
                            toolcompound = Chem.MolFromSmiles(tmps)

                        for j, hypo in enumerate(hypos[i][:self.args.nbest]):
                            hypo_tokens, hypo_str, alignment = utils.post_process_prediction(
                                hypo_tokens=hypo['tokens'].int().cpu(),
                                src_str=src_str,
                                alignment=hypo['alignment'].int().cpu() if hypo['alignment'] is not None else None,
                                align_dict=None,
                                tgt_dict=self.tgt_dict,
                                remove_bpe=self.args.remove_bpe,
                            )

                            curr_ret = filter_generated_cmpd(hypo_str.strip().replace(" ", ""))
                            if curr_ret is None:
                                continue
                            if customer_filter_fn is not None:
                                if not customer_filter_fn(*curr_ret):
                                    continue
                            results_set[curr_ret[0]] = curr_ret[1]
        
        os.makedirs("latent_logs", exist_ok=True)
        with open("latent_logs/generated_smiles.txt", "w") as f:
            for smi in results_set.keys():
                f.write(f"{smi}\n")
        

        return results_set, toolcompound
