#pragma once

#include<iostream>
#include <fstream> 
#include "gmp.h"
#include "paillier.h"
#include <string>
#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <math.h> 
#include <limits.h>
#include <vector>
#include <stdio.h>
#include <string.h>
#include <sstream>

#include <sys/stat.h>
#include <sys/types.h>

#include "cryptopp/cryptlib.h"
#include "cryptopp/osrng.h"
#include <cryptopp/rdrand.h>

using namespace std;


class PaillierUtils{

private:

	int modulusbits;
	int num_rep_bits;
	int precision_bits;

	int totalLearners;

	const paillier_get_rand_t get_rand = &paillier_get_rand_devrandom;

	paillier_pubkey_t* public_key;
	paillier_prvkey_t* private_key; 

	void scaleUpParams(const vector<double>& params, vector<unsigned long int>& scaled_params);
	void scaleDownParams(vector<unsigned long int>& scaled_params, vector<double>& params);
	void clip(std::vector<unsigned long int>& params, unsigned long int threshold);
	void pack_params(const std::vector<unsigned long int>& params, std::vector<std::string>& packed_params);
	void unpack_params(std::vector<std::string>& packed_params, std::vector<unsigned long int>& params);

	void load_keys(string keys_path);

	inline bool isNumNegative(unsigned long int n)
	{
	    //2^(num_rep_bits-1)
	    unsigned long int negativeNumCheck = pow(2,num_rep_bits-1);

	    if((n & negativeNumCheck) == 0)
	        return false;
	    else
	        return true;
	}

	inline unsigned long int calculateTwosCompliment(unsigned long int num){
	    
	    unsigned long int N = pow(2,num_rep_bits);
	    unsigned long int num_magnitude = N - num;
	    return num_magnitude;
	    
	}






public:

	PaillierUtils(int learners, string keys_path = "", int mod_bits = 2048, int num_bits = 17, int prec_bits = 13){

		totalLearners = learners;
		modulusbits = mod_bits;
		num_rep_bits = num_bits;
		precision_bits = prec_bits;

		load_keys(keys_path);

	}

	void genKeys(string keys_path);

	void encryptParams(const std::vector<unsigned long int>& params, string& result);
	void decryptParams(string& ciphertext_arr, int params, std::vector<unsigned long int>& result);
	void calculate_homomorphic_sum(std::vector<string>& learner_params, string& result);



	//Offline Phase
	string getEncryptedRandomness(string path, unsigned long int params, unsigned int iteration);
	string addEncryptedRandomness(std::vector<string>& encrypted_rand_learners);
	void decryptRandomnessSum(string& enc_rand_sum, string path, unsigned long int params, unsigned int iteration);




	//Online Phase
	string maskParams(std::vector<double>& params, string path, unsigned int iteration);
	string sumMaskedParams(std::vector<string>& learner_params, unsigned long int params);
	std::vector<double> unmaskParams(string& learner_params, int params, string sum_random_path, unsigned int iteration);



};
