import os
import pandas as pd
import argparse

def convert_raw_attention_map(raw_attention_file_path, output_attention_file_path):
    """
    Method to convert the raw attention file generated attention file

    raw_attention_file_path: file path for the input raw attention
    output_attention_file_path: file path for output attention
    """

    df_in = pd.read_csv(raw_attention_file_path)

    df_out_dict_list = []
    for index, row in df_in.iterrows():
        word_num = 1
        while "Word_"+str(word_num)+"_Left_score" in row and not(pd.isna(row["Word_"+str(word_num)+"_Left_score"])):
            df = {}
            df["Image"] = row["Image"]
            df["Label"] = row["Label"]
            df["Prediction"] = row["Prediction"]
            df["Word_num"] = word_num
            df["Left_score"] = row["Word_"+str(word_num)+"_Left_score"]
            df["Right_score"] = row["Word_"+str(word_num)+"_Left_score"]

            df_out_dict_list.append(df)
            word_num+=1

    df_out = pd.DataFrame(df_out_dict_list)
    df_out.to_csv(output_attention_file_path)


##### Parsing arguments #####
parser = argparse.ArgumentParser(description='Process command line optional inputs')
parser.add_argument('--input_raw_attention_folder', '-i', help="Input raw attention folder path")
parser.add_argument('-out_folder',  '-o', help="Output directory for final computed attention")
parser.add_argument('-num_models',  '-n', type=int, help="Number of models")
parser.add_argument('-type',  '-t', help="Type of data ie. new or old")


args = parser.parse_args()


if __name__ == '__main__':

    for n in range(1, args.num_models+1):
        cur_model = "model_{}".format(n)
        curr_file_path = os.path.join(args.input_raw_attention_folder, "res_{}.csv".format(cur_model))
        output_file_path = os.path.join(args.out_folder, "att_{}_{}.csv".format(args.type, cur_model))
        convert_raw_attention_map(curr_file_path, output_file_path)
