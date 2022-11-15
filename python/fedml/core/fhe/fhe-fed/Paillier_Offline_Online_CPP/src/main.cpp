
#include "PaillierUtils.h"
#include <time.h>
#include <omp.h>
#include <string>
#include <string.h>
#include <chrono>


using namespace std;



int main(){

	printf("Start\n");


	vector<double> params1;
	vector<double> params2;
	vector<double> params3;
	vector<double> params4;
	vector<double> params5;
	vector<double> params6;
	vector<double> params7;
	vector<double> params8;



	int total_learners = 3;
	unsigned long int total_params = 100;
	


	
	vector<string> learner_params;


	PaillierUtils* utils = new PaillierUtils(total_learners);

	//utils->genKeys("");

	//utils->generateRandomnessOffline("randomness/",total_params, 2);



	//init_params(5);



	for(int i=0; i<total_params; i++){


		float r1 = -3.0 + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/((3.0)-(-3.0))));
		float r2 = -3.0 + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/((3.0)-(-3.0))));
		float r3 = -3.0 + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/((3.0)-(-3.0))));
		float r4 = -3.0 + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/((3.0)-(-3.0))));
		float r5 = -3.0 + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/((3.0)-(-3.0))));
		float r6 = -3.0 + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/((3.0)-(-3.0))));
		float r7 = -3.0 + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/((3.0)-(-3.0))));
		float r8 = -3.0 + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/((3.0)-(-3.0))));


		params1.push_back(r1);
		params2.push_back(r2);
		params3.push_back(r3);
		params4.push_back(r4);
		params5.push_back(r5);
		params6.push_back(r6);
		params7.push_back(r7);
		params8.push_back(r8);

		cout<<r1<<" "<<r2<<" "<<r3<<endl;


	}

	string enc_rand_1 = utils->getEncryptedRandomness("randomness1/", total_params, 0);
	string enc_rand_2 = utils->getEncryptedRandomness("randomness2/", total_params, 0);
	string enc_rand_3 = utils->getEncryptedRandomness("randomness3/", total_params, 0);


	vector<string> encrypted_rands;

	encrypted_rands.push_back(enc_rand_1);
	encrypted_rands.push_back(enc_rand_2);
	encrypted_rands.push_back(enc_rand_3);

	string result_add_enc_rand = utils->addEncryptedRandomness(encrypted_rands);

	utils->decryptRandomnessSum(result_add_enc_rand, "randomness/", total_params, 0);



	string result1 = utils->maskParams(params1, "randomness1/", 0);
	string result2 = utils->maskParams(params2, "randomness2/", 0);
	string result3 = utils->maskParams(params3, "randomness3/", 0);



	vector<string> masked_params;

	masked_params.push_back(result1);
	masked_params.push_back(result2);
	masked_params.push_back(result3);


	string sum_masked = utils->sumMaskedParams(masked_params, total_params);



	std::vector<double> sum_result = utils->unmaskParams(sum_masked, total_params, "randomness/", 0);



	for(int i=0; i<sum_result.size(); i++){
		cout<<sum_result[i]<<endl;
	}










	return 1;





	/*params1.push_back(-1.23);
	params1.push_back(3.8);
	params1.push_back(-0.98);
	params1.push_back(-1.008);
	params1.push_back(3.008);

	params2.push_back(0.009);
	params2.push_back(1.1);
	params2.push_back(2.5);
	params2.push_back(-0.23);
	params2.push_back(-2.9);

	params3.push_back(-2.88);
	params3.push_back(1.56);
	params3.push_back(-0.77);
	params3.push_back(-4.23);
	params3.push_back(2.1);*/




	/*for(int i=0; i<total_params; i++){

		//float random_val = -20.0 + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/( (20.0) -(-20.0))));

		float random_val1 = -2.2134345345;
		float random_val2 = -0.1;
		float random_val3 = -0.3;
		float random_val4 = 0.2;
		float random_val5 = 0.5;


		params1.push_back(random_val1);

		//random_val = -20.0 + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/( (20.0) -(-20.0))));

		//params2.push_back(random_val2);

		//random_val = -20.0 + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/( (20.0) -(-20.0))));

		//params3.push_back(random_val3);

		//params4.push_back(random_val4);

		//params5.push_back(random_val5);

	}*/


/*
	string res0 =  utils->maskParams(params1, "learner_rand_0");
	string res1 =  utils->maskParams(params2, "learner_rand_1");
	string res2 =  utils->maskParams(params3, "learner_rand_2");

	


	std::vector<string> l_params;

	l_params.push_back(res0);
	l_params.push_back(res1);
	l_params.push_back(res2);


	string sum_result = utils->sumMaskedParams(l_params, total_params);



	std::vector<double> ress = utils->unmaskParams(sum_result, total_params, "learner_rand_sum");
*/


	//cout<<sum_result<<endl;

	//for(int i=0; i<ress.size(); i++){

	//	cout<<ress[i];
	//	cout<<" ";
	//}


	//return 1;

	



	



}
