from datetime import datetime

path_sep="\\"
dataset_home=f".{path_sep}datasets"
project_name="commons-collections"
mutant_type="ABS"
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
tensorboard_log_dir=f".{path_sep}runs{path_sep}{project_name}_{timestamp}"

embeddings_home=dataset_home+f"{path_sep}embeddings"+f"{path_sep}"+project_name
change_embeddings_home=embeddings_home+f"{path_sep}"+mutant_type
mutant_home=dataset_home+f"{path_sep}mutations"+f"{path_sep}"+project_name+f"{path_sep}mutations{path_sep}main"+f"{path_sep}"+mutant_type
mutant_execrecord_home=mutant_home+f"{path_sep}exec"
model_state_home=f".{path_sep}state"+f"{path_sep}"+project_name

method_embedding_dir=embeddings_home+f"{path_sep}method_embeddings.pt"
athena_embedding_dir=embeddings_home+f"{path_sep}athena_embeddings.pt"
athena_st_embedding_dir=embeddings_home+f"{path_sep}athena_st_embeddings.pt"
method_filter_embedding_dir=embeddings_home+f"{path_sep}method_filter_embeddings.pt"
method_splitlins_dir=embeddings_home+f"{path_sep}method_splitlines.pt"
st_embedding_pt_dir=embeddings_home+f"{path_sep}st_embeddings.pt"
node_embedding_dir=change_embeddings_home+f"{path_sep}node_embeddings.pt"
change_embeddings_dir=change_embeddings_home+f"{path_sep}change_embeddings.pt"
callgraph_dir=dataset_home+f"{path_sep}graphs"+f"{path_sep}"+project_name+f"{path_sep}callgraph.xml"
callgraph_pt_dir=dataset_home+f"{path_sep}graphs"+f"{path_sep}"+project_name+f"{path_sep}callgraph.pt"
not_found_mapping_dir=dataset_home+f"{path_sep}graphs"+f"{path_sep}"+project_name+f"{path_sep}not_found_mapping.csv"
method_dir=dataset_home+f"{path_sep}graphs"+f"{path_sep}"+project_name+f"{path_sep}method.csv"
mutant_info_dir=mutant_home+f"{path_sep}mutations.xml"
mutant_record_dir=mutant_home+f"{path_sep}record.npy"
exec_record_dir=dataset_home+f"{path_sep}mutations"+f"{path_sep}"+project_name+f"{path_sep}smf.run.xml"
mutant_change_from_dir=change_embeddings_home+f"{path_sep}from_embeddings.pt"
mutant_change_to_dir=change_embeddings_home+f"{path_sep}to_embeddings.pt"

callpath_search_max_depth=20