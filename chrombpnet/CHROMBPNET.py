import chrombpnet.parsers as parsers
import os
from chrombpnet.data import DefaultDataFile, get_default_data_path
from chrombpnet.data import print_meme_motif_file
import chrombpnet.pipelines as pipelines
import copy
import subprocess
import pandas as pd
import logging
logging.getLogger('matplotlib.font_manager').disabled = True

MODEL_COMMANDS = ["pipeline", "train", "qc", "bias", "pred_bw", "contribs_bw", "footprints"]

def check_model_runtime(args):
	# fail before any output directory is created
	device = getattr(args, "device", None)
	if device is not None:
		from chrombpnet.training.runtime import assert_gpu_if_requested
		assert_gpu_if_requested(device)
	from chrombpnet.training.utils.model_io import require_jax_backend
	require_jax_backend()


# invoke pipeline modules based on command

def main():
	args = parsers.read_parser()

	if args.cmd in MODEL_COMMANDS:
		check_model_runtime(args)
	
	if args.cmd == "pipeline" or args.cmd == "train":
		os.makedirs(os.path.join(args.output_dir,"logs"), exist_ok=False)
		os.makedirs(os.path.join(args.output_dir,"auxiliary"), exist_ok=False)
		os.makedirs(os.path.join(args.output_dir,"models"), exist_ok=False)
		os.makedirs(os.path.join(args.output_dir,"evaluation"), exist_ok=False)

		pipelines.chrombpnet_train_pipeline(args)
	
	elif args.cmd == "qc":
		os.makedirs(os.path.join(args.output_dir,"auxiliary"), exist_ok=False)
		os.makedirs(os.path.join(args.output_dir,"evaluation"), exist_ok=False)
		
		pipelines.chrombpnet_qc(args)
		
	elif args.cmd == "bias":
		if args.cmd_bias == "pipeline" or args.cmd_bias == "train":
			os.makedirs(os.path.join(args.output_dir,"logs"), exist_ok=False)
			os.makedirs(os.path.join(args.output_dir,"auxiliary"), exist_ok=False)
			os.makedirs(os.path.join(args.output_dir,"models"), exist_ok=False)
			os.makedirs(os.path.join(args.output_dir,"evaluation"), exist_ok=False)

			pipelines.train_bias_pipeline(args)
		
		elif args.cmd_bias == "qc":
			os.makedirs(os.path.join(args.output_dir,"auxiliary"), exist_ok=False)
			os.makedirs(os.path.join(args.output_dir,"evaluation"), exist_ok=False)
			
			pipelines.bias_model_qc(args)
			
		else:
			print("Command not found")

			
	
	elif args.cmd == "pred_bw":
	
		assert (args.bias_model is None) + (args.chrombpnet_model is None) + (args.chrombpnet_model_nb is None) < 3, "No input model provided!"
		import chrombpnet.evaluation.make_bigwigs.predict_to_bigwig as predict_to_bigwig

		predict_to_bigwig.main(args)

	elif args.cmd == "contribs_bw":
	
		import chrombpnet.evaluation.interpret.interpret as interpret
		pipelines.interpret_args(args, args)
		interpret.main(args)
		import chrombpnet.evaluation.make_bigwigs.importance_hdf5_to_bigwig as importance_hdf5_to_bigwig
		if "counts" in  args.profile_or_counts:
			args_copy = copy.deepcopy(args)
			args_copy.hdf5 = args_copy.output_prefix + ".counts_scores.h5"
			args_copy.output_prefix = args.output_prefix + ".counts_scores"
			args_copy.regions =  args.output_prefix + ".interpreted_regions.bed"

			importance_hdf5_to_bigwig.main(args_copy)
		if "profile" in  args.profile_or_counts:
			args_copy = copy.deepcopy(args)
			args_copy.hdf5 = args_copy.output_prefix + ".profile_scores.h5"
			args_copy.output_prefix = args.output_prefix + ".profile_scores"
			args_copy.regions =  args.output_prefix + ".interpreted_regions.bed"
	
			importance_hdf5_to_bigwig.main(args_copy)
			
	elif args.cmd == "footprints":
	
		import chrombpnet.evaluation.marginal_footprints.marginal_footprinting as marginal_footprinting
		marginal_footprinting.main(args)

	elif args.cmd == "prep":
	
		if args.cmd_prep == "nonpeaks":

			assert(args.inputlen%2==0) # input length should be a multiple of 2
	
			os.makedirs(args.output_prefix+"_auxiliary/", exist_ok=False)
	
			from chrombpnet.helpers.make_gc_matched_negatives.get_genomewide_gc_buckets.get_genomewide_gc_bins import get_genomewide_gc
			get_genomewide_gc(args.genome,args.output_prefix+"_auxiliary/genomewide_gc.bed",args.inputlen, args.stride)
	
			# get gc content in peaks
			import chrombpnet.helpers.make_gc_matched_negatives.get_gc_content as get_gc_content
			args_copy = copy.deepcopy(args)
			args_copy.input_bed = args_copy.peaks
			args_copy.output_prefix = args.output_prefix+"_auxiliary/foreground.gc"
			get_gc_content.main(args_copy)
	
			# prepare candidate negatives
	
			exclude_bed = pd.read_csv(args.peaks, sep="\t", header=None)
			subprocess.run("bedtools slop -i {peaks} -g {chrom_sizes} -b {flank_size} > {output}".format(peaks=args.peaks,
												chrom_sizes=args.chrom_sizes,
												flank_size=args.inputlen//2,
												output=args.output_prefix+"_auxiliary/peaks_slop.bed"), shell=True, check=True)
			exclude_bed = pd.read_csv(args.output_prefix+"_auxiliary/peaks_slop.bed", sep="\t", header=None, usecols=[0,1,2])
	
			if args.blacklist_regions:
				subprocess.run("bedtools slop -i {blacklist} -g {chrom_sizes} -b {flank_size} > {output}".format(blacklist=args.blacklist_regions,
												chrom_sizes=args.chrom_sizes,
												flank_size=args.inputlen//2,
												output=args.output_prefix+"_auxiliary/blacklist_slop.bed"), shell=True, check=True)
										
				exclude_bed = pd.concat([exclude_bed,pd.read_csv(args.output_prefix+"_auxiliary/blacklist_slop.bed",sep="\t",header=None, usecols=[0,1,2])])

			exclude_bed.to_csv(args.output_prefix+"_auxiliary/exclude_unmerged.bed", sep="\t", header=False, index=False)
			subprocess.run("bedtools sort -i {inputb} | bedtools merge -i stdin > {output}".format(inputb=args.output_prefix+"_auxiliary/exclude_unmerged.bed",
																					output=args.output_prefix+"_auxiliary/exclude.bed"), shell=True, check=True)			
	
	

			bedtools_command = "bedtools intersect -v -a {genomewide_gc} -b {exclude_bed}  > {candidate_bed}".format(
															genomewide_gc=args.output_prefix+"_auxiliary/genomewide_gc.bed",
															exclude_bed=args.output_prefix+"_auxiliary/exclude.bed",
															candidate_bed=args.output_prefix+"_auxiliary/candidates.bed")
			subprocess.run(bedtools_command, shell=True, check=True)
													
			# get final negatives
			import chrombpnet.helpers.make_gc_matched_negatives.get_gc_matched_negatives as get_gc_matched_negatives
			args_copy = copy.deepcopy(args)
			args_copy.candidate_negatives = args.output_prefix+"_auxiliary/candidates.bed"
			args_copy.foreground_gc_bed = args.output_prefix+"_auxiliary/foreground.gc.bed"
			args_copy.output_prefix = 	args.output_prefix+"_auxiliary/negatives"
	
			get_gc_matched_negatives.main(args_copy)
	
			negatives = pd.read_csv(args.output_prefix+"_auxiliary/negatives.bed", sep="\t", header=None)
			negatives[3]="."
			negatives[4]="."
			negatives[5]="."
			negatives[6]="."
			negatives[7]="."
			negatives[8]="."
			negatives[9]=args.inputlen//2
			negatives.to_csv(args.output_prefix+"_negatives.bed", sep="\t", header=False, index=False)

		elif args.cmd_prep == "splits":
			import chrombpnet.helpers.make_chr_splits.splits as splits
			splits.main(args)
			
		else:
			print("Command not found")
		
	else:
		print("Command not found")


if __name__=="__main__":
	main()

    
        
