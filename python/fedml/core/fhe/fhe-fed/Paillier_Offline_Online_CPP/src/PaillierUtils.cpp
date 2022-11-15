#include "PaillierUtils.h"


void PaillierUtils::load_keys(string keys_path){

	char* pub_key;
	char* priv_key;

	string pub_path = keys_path;
	pub_path+="public_key-paillier.txt";

	string priv_path = keys_path;
	priv_path+="private_key-paillier.txt";

	FILE* file;
	long fsize;
	char* key_loaded;
	size_t result;


	file = fopen(pub_path.c_str(), "rb");



	if (file!=NULL){

		fseek(file, 0, SEEK_END);
		fsize = ftell(file);
		fseek(file, 0, SEEK_SET);

		key_loaded = (char *)malloc(fsize + 1);
		result = fread(key_loaded, fsize, 1, file);

		fclose(file);
		key_loaded[fsize] = 0;

		public_key = paillier_pubkey_from_hex(key_loaded);
		delete[] key_loaded;
		 	

	}

	else{

		cout<<"Error reading public key"<<endl;

	}


	file = fopen(priv_path.c_str(), "rb");


	if (file!=NULL){

		fseek(file, 0, SEEK_END);
		fsize = ftell(file);
		fseek(file, 0, SEEK_SET);

		key_loaded = (char *)malloc(fsize + 1);
		result = fread(key_loaded, fsize, 1, file);

		fclose(file);

		key_loaded[fsize] = 0;

		if(fsize!=0 && public_key!=nullptr){

			private_key = paillier_prvkey_from_hex(key_loaded, public_key);

		}
		

	}

	else {

		cout<<"Error reading private key"<<endl;

	}


}


void PaillierUtils::genKeys(string keys_path){

	paillier_keygen(this->modulusbits, &public_key, &private_key, paillier_get_rand_devurandom);

	if(public_key == nullptr || private_key == nullptr){
		cout<<"keys not initialized"<<endl;
		
	} 


	char* pub_key;
	char* priv_key;

	pub_key = paillier_pubkey_to_hex( public_key );
	priv_key = paillier_prvkey_to_hex( private_key );


	string pub_path = keys_path;
	pub_path+= "public_key-paillier.txt";



	FILE *pub_file = fopen(pub_path.c_str(), "w");

	int results_pub = fputs(pub_key, pub_file);
	if (results_pub == EOF) {
	    cout<<"Failed to write public key"<<endl;
	}
	fclose(pub_file);

	string priv_path = keys_path;
	priv_path+= "private_key-paillier.txt";


	FILE *priv_file = fopen(priv_path.c_str(), "w");

	int results_priv = fputs(priv_key, priv_file);
	if (results_priv == EOF) {
	    cout<<"Failed to write private key"<<endl;
	}
	fclose(priv_file);


}





void PaillierUtils::scaleUpParams(const vector<double>& params, vector<unsigned long int>& scaled_params){

	unsigned long int scale = pow(2, precision_bits);
	unsigned long int threshold = pow(2,num_rep_bits-1); 

	scaled_params.reserve(params.size());


	//scale up to keep precision bits
	for(int i=0; i<params.size(); i++){

		scaled_params.push_back((unsigned long int) (params[i] * scale) );

	}


	clip(scaled_params, threshold);


	for(int i=0; i<params.size(); i++){

		//changes all numbers to positive in the range 0-2^(num_rep_bits) - 1
		scaled_params[i] = scaled_params[i]+threshold;

	}


}


void PaillierUtils::scaleDownParams(vector<unsigned long int>& scaled_params, vector<double>& params){

	params.reserve(scaled_params.size());

	double threshold = pow(2,num_rep_bits-1);
	double scale = pow(2, precision_bits);


	for(int i=0; i<scaled_params.size(); i++){

		params.push_back(scaled_params[i]);

		params[i] = (params[i] - (totalLearners*threshold))/scale;


	}


}



void PaillierUtils::pack_params(const std::vector<unsigned long int>& params, std::vector<std::string>& packed_params){


	int pad_zeros = totalLearners - 1;

	

	const unsigned int params_size =  params.size();

	int bytes_to_rep_num = PAILLIER_BITS_TO_BYTES(num_rep_bits); 

	//determining how many zero bytes to pad to account for overflow during sum
	int extra_padding_bits = pad_zeros - ((bytes_to_rep_num*8) - num_rep_bits);
	int extra_bytes_to_pad = 0;
	if(extra_padding_bits > 0){

		extra_bytes_to_pad = PAILLIER_BITS_TO_BYTES(extra_padding_bits);
	} 


	int total_size_num = bytes_to_rep_num + extra_bytes_to_pad;
	int nums_to_pack = (modulusbits/8)/total_size_num;
	int packed_params_size = ceil((float)params_size/(float)nums_to_pack);
	packed_params.resize(packed_params_size, "");


	int count_params = 0;

	//starting byte index to copy the number from
	int num_start_byte = bytes_to_rep_num -1;
	if(num_start_byte> sizeof(unsigned long int)-1){
		cout<<"Error: Number representation greater than unsigned long int."<<endl;
		return;
	}

	unsigned long int zero_rep = 0;
	char* num_ptr;


	for(int i=0; i<packed_params_size; i++){

		packed_params[i].reserve(modulusbits/8);

		for(int j=0; j<nums_to_pack; j++){


			for(int k=0; k<extra_bytes_to_pad; k++){

				packed_params[i]+= (char) 0;
				
			}


			if(count_params<params.size()){

				num_ptr = (char*)& params[count_params++];

			}
			else{

				num_ptr = (char*)& zero_rep;
			}


			int start_index = num_start_byte;

			while(start_index>=0){

				packed_params[i]+= num_ptr[start_index--];
			}


		}


	}


}


void PaillierUtils::unpack_params(std::vector<std::string>& packed_params, std::vector<unsigned long int>& params){


	int pad_zeros = totalLearners - 1;


	int packed_params_size = packed_params.size();

	int bytes_to_rep_num = PAILLIER_BITS_TO_BYTES(num_rep_bits); 


	//determining how many zero bytes to pad to account for overflow during sum
	int extra_padding_bits = pad_zeros - ((bytes_to_rep_num*8) - num_rep_bits);
	int extra_bytes_to_pad = 0;
	if(extra_padding_bits > 0){

		extra_bytes_to_pad = PAILLIER_BITS_TO_BYTES(extra_padding_bits);
	} 

	int total_size_num = bytes_to_rep_num + extra_bytes_to_pad;

	if(total_size_num > sizeof(unsigned long int)){
		cout<<"Error: Number representation greater than unsigned long int."<<endl;
		return;
	}


	


	for(int i=0; i<packed_params_size; i++){

		for(int j=0; j<packed_params[i].size(); j+=total_size_num){

			if(j+total_size_num > packed_params[i].size()){

				break;
			}

			unsigned int long num = 0;
			char* a_ptr = (char*)& num;


			for(int k=0; k<total_size_num; k++)
			{

				a_ptr[total_size_num-1-k] = packed_params[i][j+k];

			}

			params.push_back(num);

			
		}


	}


}




void PaillierUtils::clip(std::vector<unsigned long int>& params, unsigned long int threshold){

	int param_size = params.size();
	//2^63
	unsigned long int negativeNumCheck = 9223372036854775808U;

	for(int i=0; i<param_size; i++){

		//if number is negative and ..

		if(((params[i] & negativeNumCheck) != 0) &&  (params[i] < ((unsigned long int) (-1*threshold)))   ){

			params[i] = (unsigned long int) (-1*threshold);

		}

		//if number is positive and ..

		else if( ((params[i] & negativeNumCheck) == 0)  &&  (params[i] > (threshold-1)) ){

			params[i] = threshold-1;

		}

	}


}



void PaillierUtils::encryptParams(const std::vector<unsigned long int>& params, string& result){
	
	vector<string> packed_params;
	pack_params(params, packed_params);

	result="";
	result.reserve(PAILLIER_BITS_TO_BYTES(public_key->bits) * 2 * packed_params.size());


	//#pragma omp parallel for
	for(int i=0; i<packed_params.size(); i++){


		CryptoPP::RDRAND prng;

		
		paillier_plaintext_t* pt1 = paillier_plaintext_from_bytes((void*)packed_params[i].c_str(), PAILLIER_BITS_TO_BYTES(public_key->bits));
		paillier_ciphertext_t* ct1 = paillier_enc(NULL, public_key, pt1, get_rand);



	    char* byteCt1 = (char*)paillier_ciphertext_to_bytes(PAILLIER_BITS_TO_BYTES(public_key->bits)*2, ct1);

	    result.append(byteCt1, PAILLIER_BITS_TO_BYTES(public_key->bits)*2);

    	paillier_freeplaintext(pt1);
    	paillier_freeciphertext(ct1);
    	free(byteCt1);


	}


}


void PaillierUtils::decryptParams(string& ciphertext_arr, int total_params, std::vector<unsigned long int>& result){



	const char* ct_arr = ciphertext_arr.c_str();

	vector<string> packed_params;

	packed_params.reserve(ciphertext_arr.size()/(PAILLIER_BITS_TO_BYTES(public_key->bits*2)));


	//#pragma omp parallel for
	for(int i=0; i<ciphertext_arr.size(); i+=(PAILLIER_BITS_TO_BYTES(public_key->bits))*2){



	    paillier_ciphertext_t* ct1 = paillier_ciphertext_from_bytes((void*)(ct_arr+i), PAILLIER_BITS_TO_BYTES(public_key->bits)*2);

    	paillier_plaintext_t* dec_pt = paillier_dec(NULL, public_key, private_key, ct1); 

		char* res_plain_ans = (char*) paillier_plaintext_to_bytes( PAILLIER_BITS_TO_BYTES(public_key->bits), dec_pt );


		/*cout<<"Dec"<<endl;



			for(int j=0; j<PAILLIER_BITS_TO_BYTES(public_key->bits); j++){

				std::bitset<8> b(res_plain_ans[j]);
				cout<<b; 
			}

			cout<<endl;*/



		packed_params.push_back(string(res_plain_ans, PAILLIER_BITS_TO_BYTES(public_key->bits)));

		paillier_freeplaintext(dec_pt);
    	paillier_freeciphertext(ct1);
    	free(res_plain_ans);

	}


	result.reserve(total_params);
	unpack_params(packed_params,  result);

	result.resize(total_params);




}


void PaillierUtils::calculate_homomorphic_sum(std::vector<string>& learner_params, string& result){

	
	result.resize(learner_params[0].size());
	//const char* result_str = result.c_str();


	//#pragma omp parallel for 
	for(int i=0; i<learner_params[0].size(); i+= (PAILLIER_BITS_TO_BYTES(public_key->bits*2))){


		const char* L1 = learner_params[0].c_str();
	    paillier_ciphertext_t* ct1 = paillier_ciphertext_from_bytes((void*)(L1+i), PAILLIER_BITS_TO_BYTES(public_key->bits)*2);



		for(int j=1; j<learner_params.size(); j++){

			const char* Ln = learner_params[j].c_str();
	    	paillier_ciphertext_t* ctn = paillier_ciphertext_from_bytes((void*)(Ln+i), PAILLIER_BITS_TO_BYTES(public_key->bits)*2);
			paillier_mul(public_key, ct1, ct1, ctn);


			paillier_freeciphertext(ctn);

		}

	    char* byteCt = (char*)paillier_ciphertext_to_bytes(PAILLIER_BITS_TO_BYTES(public_key->bits)*2, ct1);

		int count_ct=0;


		for(int k =i; k<i+PAILLIER_BITS_TO_BYTES(public_key->bits)*2; k++){


			result[k] = byteCt[count_ct++];
		}



		paillier_freeciphertext(ct1);
		free(byteCt);

	}


	

}






string PaillierUtils::maskParams(std::vector<double>& params, string path, unsigned int iteration){


	string rand_path = path + to_string(iteration) + "/" + "learner_rand";
	ifstream random_file(rand_path, std::ifstream::in);

	if(!random_file.is_open()){

		cout<<"Error opening randomness file"<<endl;
		exit(EXIT_FAILURE);
	}

	unsigned long int N = pow(2,num_rep_bits);
	unsigned long int scale = pow(2, precision_bits);
	unsigned long int threshold = pow(2,num_rep_bits-1);

	std::vector<unsigned long int> scaled_params; 
	scaled_params.reserve(params.size());


	//scale up to keep precision bits
	for(unsigned long int i=0; i<params.size(); i++){

		scaled_params.push_back((unsigned long int) (params[i] * scale) );

	}


	clip(scaled_params, threshold);

	string value;

	for(unsigned long int i=0; i<scaled_params.size(); i++){

		getline(random_file, value, ' ');

		scaled_params[i] = scaled_params[i] - atoll(value.c_str());
		scaled_params[i] = scaled_params[i]%N;

	}
	

	string result = "";
	result.reserve(scaled_params.size()*10);

	for(unsigned long int i=0; i<scaled_params.size(); i++){

		result.append(to_string(scaled_params[i]));
		result.append(";");

	}


	return result;

}



string PaillierUtils::sumMaskedParams(std::vector<string>& learner_params, unsigned long int params){

	std::vector<unsigned long int> result_params;
	result_params.resize(params,0);

	char delim = ';';
	

    for(unsigned long int i=0; i<learner_params.size(); i++){

    	int start_index = 0;
    	int end_index = 0;
    	int count = 0;

    	for(unsigned long int j=0; j<learner_params[i].size(); j++){

    		if(learner_params[i].at(end_index) == delim){

    			string val_substr = learner_params[i].substr(start_index, end_index-start_index);
    			result_params[count] = result_params[count] + atoll(val_substr.c_str());
    			
    			count++;
    			end_index++;
    			start_index = end_index;


    		}

    		else{

    			end_index++;
    		}

    	}


    }


    unsigned long int N = pow(2,num_rep_bits);

	
	for(unsigned long int i=0; i<params; i++){

		result_params[i] = result_params[i]%N;

    }


    string result = "";

	result.reserve(result_params.size()*10);

	for(unsigned long int i=0; i<result_params.size(); i++){

		result.append(to_string(result_params[i]));
		result.append(";");

	}


	return result;



}




std::vector<double> PaillierUtils::unmaskParams(string& learner_params, int params, string sum_random_path, unsigned int iteration){

	unsigned long int N = pow(2,num_rep_bits);

	string rand_path = sum_random_path + to_string(iteration) + "/" + "learner_rand_sum";
	ifstream random_file(rand_path, std::ifstream::in);

	if(!random_file.is_open()){

		cout<<"Error opening randomness file"<<endl;
		exit(EXIT_FAILURE);
	}


	std::vector<unsigned long int> scaled_params;
	scaled_params.reserve(params);

	std::vector<double> result;
	result.reserve(params);
	

	char delim = ';';
	int start_index = 0;
	int end_index = 0;

	for(unsigned long int j=0; j<learner_params.size(); j++){

		if(learner_params.at(end_index) == delim){

			string val_substr = learner_params.substr(start_index, end_index-start_index);
			scaled_params.push_back(atoll(val_substr.c_str())); 
			
			end_index++;
			start_index = end_index;

		}

		else{

			end_index++;
		}

	}

	
	string value;

	for(unsigned long int i=0; i<params; i++){

		getline(random_file, value, ' ');

		double decimalValue;

		scaled_params[i] = scaled_params[i] + atoll(value.c_str());
		scaled_params[i] = scaled_params[i]%N;


		if(isNumNegative(scaled_params[i])){
                
            decimalValue = calculateTwosCompliment(scaled_params[i]);
            decimalValue = -1 * decimalValue/pow(2,precision_bits);

        }
        else{
            
            decimalValue = scaled_params[i];
            decimalValue = decimalValue/pow(2,precision_bits);

        }


		result.push_back(decimalValue/totalLearners);


	}

	random_file.close();



	return result; 
	


}



string PaillierUtils::getEncryptedRandomness(string path, unsigned long int params, unsigned int iteration){


	unsigned long int N = pow(2,num_rep_bits);
	CryptoPP::AutoSeededRandomPool rng;
	std::vector<unsigned long int> rand_nums;
	rand_nums.reserve(params);


	struct stat st;

	if(stat(path.c_str(),&st) != 0){
        	if (mkdir(path.c_str(), 0777) == -1)
        		cerr << "Error creating directory for output:  " << strerror(errno) << endl;
    }


	
	string path_folder = path+to_string(iteration)+"/";


	if(stat(path_folder.c_str(),&st) != 0){
        	if (mkdir(path_folder.c_str(), 0777) == -1)
        		cerr << "Error creating directory for output:  " << strerror(errno) << endl;
    }


    string path_name = path_folder + "learner_rand";

    FILE *rand_file = fopen(path_name.c_str(), "w"); 

    string rand_nums_str = "";
    rand_nums_str.reserve(params * 10);


	for(unsigned long int i=0; i<params; i++){

		CryptoPP::Integer rand_num_int = CryptoPP::Integer(rng, num_rep_bits);
		unsigned long int rand_num = (unsigned long int) rand_num_int.ConvertToLong();
		rand_num = rand_num%N;

		rand_nums.push_back(rand_num);

		rand_nums_str.append(to_string(rand_num));
		rand_nums_str.append(" ");


	}


	fputs(rand_nums_str.c_str(), rand_file);
	fclose(rand_file);


	string encrypted_rand_nums;
	encryptParams(rand_nums, encrypted_rand_nums);
	return encrypted_rand_nums;


}




string PaillierUtils::addEncryptedRandomness(std::vector<string>& encrypted_rand_learners){


	string result;
	calculate_homomorphic_sum(encrypted_rand_learners, result);
	return result;

}


void PaillierUtils::decryptRandomnessSum(string& enc_rand_sum, string path, unsigned long int params, unsigned int iteration){

	std::vector<unsigned long int> result;
	unsigned long int N = pow(2,num_rep_bits);

	decryptParams(enc_rand_sum, params, result);


	string path_folder = path+to_string(iteration) +"/";
	string sum_rand_path = path_folder + "learner_rand_sum";


	struct stat st;
	
	if(stat(path_folder.c_str(),&st) != 0){
        	if (mkdir(path_folder.c_str(), 0777) == -1)
        		cerr << "Error creating directory for output:  " << strerror(errno) << endl;
    }


	FILE* rand_sum_file = fopen(sum_rand_path.c_str(), "w");


	for(unsigned int i=0; i<result.size(); i++){

		result[i] = result[i]%N;

		fputs(to_string(result[i]).c_str(), rand_sum_file);
		fputs(" ", rand_sum_file);	

	}

	fclose(rand_sum_file);


}

