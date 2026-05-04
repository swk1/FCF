import argparse
import pickle

from baselines.CC2Vec.cc2vecTransfor import read_args, set_seed
from baselines.CC2Vec.jit_padding import padding_message, clean_and_reformat_code, padding_commit_code, mapping_dict_msg, mapping_dict_code, convert_msg_to_label
from baselines.CC2Vec.jit_cc2ftr_train import train_model
from baselines.CC2Vec.jit_cc2ftr_extracted import extracted_cc2ftr


if __name__ == '__main__':
    params = read_args().parse_args()    
    set_seed(seed=42)
    if params.train is True:
        train_data = pickle.load(open(params.train_data, 'rb'))
        ids, labels, msgs, codes = train_data    
        
        dictionary = pickle.load(open(params.dictionary_data, 'rb'))
        dict_msg, dict_code = dictionary  

        pad_msg = padding_message(data=msgs, max_length=params.msg_length)
        added_code, removed_code = clean_and_reformat_code(codes)
        pad_added_code = padding_commit_code(data=added_code, max_file=params.code_file, max_line=params.code_line, max_length=params.code_length)
        pad_removed_code = padding_commit_code(data=removed_code, max_file=params.code_file, max_line=params.code_line, max_length=params.code_length)

        pad_msg = mapping_dict_msg(pad_msg=pad_msg, dict_msg=dict_msg)
        pad_added_code = mapping_dict_code(pad_code=pad_added_code, dict_code=dict_code)
        pad_removed_code = mapping_dict_code(pad_code=pad_removed_code, dict_code=dict_code)
        pad_msg_labels = convert_msg_to_label(pad_msg=pad_msg, dict_msg=dict_msg)

        data = (pad_added_code, pad_removed_code, pad_msg_labels, dict_msg, dict_code) 

        # params.save_dir = os.path.join(params.save_dir,params.project,datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
        params.vocab_code = len(dict_code)


        if len(pad_msg_labels.shape) == 1:
            params.class_num = 1
        else:
            params.class_num = pad_msg_labels.shape[1]
        
#         train_model(data=data, params=params)

        # just to measure training time...
        # start = time.time()
        train_model(data=data, params=params)
        # end = time.time()

        # train_time_sec = end-start
            
        print('--------------------------------------------------------------------------------')
        print('--------------------------Finish the training process---------------------------')
        print('--------------------------------------------------------------------------------')
        exit()
    
    elif params.predict is True:
        data = pickle.load(open(params.predict_data, 'rb'))
        ids, labels, msgs, codes = data 

        dictionary = pickle.load(open(params.dictionary_data, 'rb'))   
        dict_msg, dict_code = dictionary  

        pad_msg = padding_message(data=msgs, max_length=params.msg_length)
        added_code, removed_code = clean_and_reformat_code(codes)
        pad_added_code = padding_commit_code(data=added_code, max_file=params.code_file, max_line=params.code_line, max_length=params.code_length)
        pad_removed_code = padding_commit_code(data=removed_code, max_file=params.code_file, max_line=params.code_line, max_length=params.code_length)

        pad_msg = mapping_dict_msg(pad_msg=pad_msg, dict_msg=dict_msg)
        pad_added_code = mapping_dict_code(pad_code=pad_added_code, dict_code=dict_code)
        pad_removed_code = mapping_dict_code(pad_code=pad_removed_code, dict_code=dict_code)
        pad_msg_labels = convert_msg_to_label(pad_msg=pad_msg, dict_msg=dict_msg)
        
        data = (pad_added_code, pad_removed_code, pad_msg_labels, dict_msg, dict_code)
        params.batch_size = 1
        extracted_cc2ftr(data=data, params=params)
        print('--------------------------------------------------------------------------------')
        print('--------------------------Finish the extracting process-------------------------')
        print('--------------------------------------------------------------------------------')
        exit()
