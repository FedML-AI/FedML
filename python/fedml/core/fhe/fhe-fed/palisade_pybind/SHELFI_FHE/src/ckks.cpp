#include "ckks.h"



CKKS::CKKS(string scheme, uint batchSize, uint scaleFactorBits, string cryptodir) : Scheme(scheme) {
  this->batchSize = batchSize;
  this->scaleFactorBits = scaleFactorBits;
  this->cryptodir = cryptodir;
}

void CKKS::loadCryptoParams() {
	    if (!Serial::DeserializeFromFile(cryptodir + "cryptocontext.txt", cc, SerType::BINARY)) {
      	std::cout << "Could not read serialization from " << cryptodir + "cryptocontext.txt" << std::endl;
    	}

    	if (!Serial::DeserializeFromFile(cryptodir + "key-public.txt", pk, SerType::BINARY)) {
    		std::cout << "Could not read public key" << std::endl;
    	}

    	if (Serial::DeserializeFromFile(cryptodir + "key-private.txt", sk, SerType::BINARY) == false) {
    		std::cerr << "Could not read secret key" << std::endl;
    	}
}

int CKKS::genCryptoContextAndKeyGen() {
      usint multDepth = 1;

      CryptoContext<DCRTPoly> cryptoContext = CryptoContextFactory<DCRTPoly>::genCryptoContextCKKS(multDepth, scaleFactorBits, batchSize);
      

      // enable features that you wish to use
      cryptoContext->Enable(ENCRYPTION);
      cryptoContext->Enable(SHE);
      // cryptoContext->Enable(LEVELEDSHE);

      string cryptocontext_file = "cryptocontext.txt";
      string public_key_file = "key-public.txt";
      string private_key_file = "key-private.txt";


      if (!Serial::SerializeToFile(cryptodir + cryptocontext_file, cryptoContext, SerType::BINARY)) {
        std::cerr << "Error writing serialization of the crypto context to "<< cryptocontext_file << std::endl;
        return 0;
      }

      LPKeyPair<DCRTPoly> keyPair = cryptoContext->KeyGen();

      if (!Serial::SerializeToFile(cryptodir + public_key_file, keyPair.publicKey, SerType::BINARY)) {
        std::cerr << "Error writing serialization of public key to "<< public_key_file << std::endl;
        return 0;
      }

      if (!Serial::SerializeToFile(cryptodir + private_key_file, keyPair.secretKey, SerType::BINARY)) {
        std::cerr<< "Error writing serialization of private key to "<< private_key_file << std::endl;
        return 0;
      }

      return 1;
}

	py::bytes CKKS::encrypt(py::array_t<double> data_array) {
        unsigned long int size = data_array.size();
        auto learner_Data = data_array.data();

        int cipherSize = ceil(size / (double)batchSize);
        vector<Ciphertext<DCRTPoly>> ciphertext_data(cipherSize);
        if (size > (unsigned long int) batchSize) {

          //int j = 0;
          #pragma omp parallel for
          for (unsigned long int i = 0; i < size; i += batchSize) {
            unsigned long int last = std::min((long)size, (long)i + batchSize);
            vector<double> batch;
            batch.reserve(last - i + 1);

            for (unsigned long int j = i; j < last; j++) {
              batch.push_back(learner_Data[j]);
            }

            Plaintext plaintext_data = cc->MakeCKKSPackedPlaintext(batch);
            ciphertext_data[(int)(i/batchSize)] = cc->Encrypt(pk, plaintext_data);
            batch.clear();
          }
        } else {

          vector<double> batch;
          batch.reserve(size);

          for (unsigned long int i = 0; i < size; i++) {
            // float dat = py::float_(learner_Data[i]);
            batch.push_back(py::float_(learner_Data[i]));
          }

          Plaintext plaintext_data = cc->MakeCKKSPackedPlaintext(batch);
          ciphertext_data[0] = cc->Encrypt(pk, plaintext_data);
        }

        stringstream s;
        const SerType::SERBINARY st;
        Serial::Serialize(ciphertext_data, s, st);
        py::bytes res(s.str());

        return res;
	}


  string CKKS::encrypt_cpp(vector<double> learner_Data)  {


    unsigned long int size = learner_Data.size();
    int cipherSize = ceil(size / (double)batchSize);
    vector<Ciphertext<DCRTPoly>> ciphertext_data(cipherSize);

    

    if (size > (unsigned long int)batchSize) {


        #pragma omp parallel for
        for (unsigned long int i = 0; i < size; i += batchSize) {

          unsigned long int last = std::min((long)size, (long)i + batchSize);

          vector<double> batch;
          batch.reserve(last - i + 1);

          for (unsigned long int j = i; j < last; j++) {

            batch.push_back(learner_Data[j]);
          }


          Plaintext plaintext_data = cc->MakeCKKSPackedPlaintext(batch);
          ciphertext_data[(int)(i/batchSize)] = cc->Encrypt(pk, plaintext_data);

        }

      }

      else {

        vector<double> batch;
        batch.reserve(size);

        for (unsigned long int i = 0; i < size; i++) {

          batch.push_back(learner_Data[i]);
        }

        Plaintext plaintext_data = cc->MakeCKKSPackedPlaintext(batch);
        ciphertext_data[0] = cc->Encrypt(pk, plaintext_data);
      }



    


    stringstream s;
    const SerType::SERBINARY st;
    Serial::Serialize(ciphertext_data, s, st);


    return s.str();

   
}


	py::array_t<double> CKKS::decrypt(string learner_Data, unsigned long int data_dimesions) {
		    //auto start_deserialize = std::chrono::system_clock::now();
      	const SerType::SERBINARY st;
      	stringstream ss(learner_Data);

     	  vector<Ciphertext<DCRTPoly>> learner_ciphertext;
      	Serial::Deserialize(learner_ciphertext, ss, st);

      	//auto end_deserialize = std::chrono::system_clock::now();
      	//elapsed_time_deserialize+=std::chrono::duration_cast<std::chrono::milliseconds>(end_deserialize
        //- start_deserialize).count(); 
      	auto result = py::array_t<double>(data_dimesions);
		    py::buffer_info buf3 = result.request();
		    double *ptr3 = static_cast<double *>(buf3.ptr);
		    //auto start_decrypt = omp_get_wtime();

      	#pragma omp parallel for
      	for (unsigned long int i = 0; i < learner_ciphertext.size(); i++) {
          Plaintext pt;
          cc->Decrypt(sk, learner_ciphertext[i], &pt);
        	int length;

        	if (i == learner_ciphertext.size() - 1) {
            length = data_dimesions - (i)*batchSize;
        	} else {
            length = batchSize;
          }

        	pt->SetLength(length);
        	vector<double> layer_data = pt->GetRealPackedValue();
          int m = i*batchSize;

        	for (unsigned long int j = 0; j < layer_data.size(); j++) {
            ptr3[m++] = layer_data[j];
        	}
      	}

      	//auto end_decrypt = omp_get_wtime();
      	//elapsed_time_decrypt+=end_decrypt - start_decrypt; 
      	//std::cout <<"Deserialization time: "<< elapsed_time_deserialize<< " milliseconds"<<'\n';
      	//std::cout <<"Decryption Time: "<< elapsed_time_decrypt *1000<< " milliseconds"<<'\n';
      	learner_ciphertext.clear();
      	return result;
    } 
	


  vector<double> CKKS::decrypt_cpp(string learner_Data, unsigned long int data_dimesions) {

    const SerType::SERBINARY st;
    stringstream ss(learner_Data);

    vector<Ciphertext<DCRTPoly>> learner_ciphertext;
    Serial::Deserialize(learner_ciphertext, ss, st);

    vector<double> result(data_dimesions); 


    #pragma omp parallel for
    for (unsigned long int i = 0; i < learner_ciphertext.size(); i++) {

      Plaintext pt;
      cc->Decrypt(sk, learner_ciphertext[i], &pt);
      int length;

      if (i == learner_ciphertext.size() - 1) {

        length = data_dimesions - (i)*batchSize;
      }

      else {

        length = batchSize;
      }

      pt->SetLength(length);
      vector<double> layer_data = pt->GetRealPackedValue();
      int m = i*batchSize;

      for (unsigned long int j = 0; j < layer_data.size(); j++) {

        result[m++] = layer_data[j];
      }

      
    }


    return result;

}



	py::bytes CKKS::computeWeightedAverage(py::list learner_data, py::list scaling_factors) {
        if (learner_data.size() != scaling_factors.size()) {
      		cout << "Error: learner_data and scaling_factors size mismatch" << endl;
        	return "";
        }

        const SerType::SERBINARY st;
        vector<Ciphertext<DCRTPoly>> result_ciphertext;

        for (unsigned long int i = 0; i < learner_data.size(); i++) {

        	//auto start_deserialize = omp_get_wtime();
        	string dat = std::string(py::str(learner_data[i]));

        	stringstream ss(dat);
        	vector<Ciphertext<DCRTPoly>> learner_ciphertext;

        	Serial::Deserialize(learner_ciphertext, ss, st);
          //auto end_deserialize = omp_get_wtime();
        	//elapsed_time_deserialize+=end_deserialize - start_deserialize; 
          //auto start_pwa = omp_get_wtime();

        	for (unsigned long int j = 0; j < learner_ciphertext.size(); j++) {
          		float sc = py::float_(scaling_factors[i]);
          		learner_ciphertext[j] = cc->EvalMult(learner_ciphertext[j], sc);
        	}

        	if (result_ciphertext.size() == 0) {
          		result_ciphertext = learner_ciphertext;
        	} else {
        		for (unsigned long int j = 0; j < learner_ciphertext.size(); j++) {
            		result_ciphertext[j] =
                cc->EvalAdd(result_ciphertext[j], learner_ciphertext[j]);
          		}
        	}

        	//auto end_pwa = omp_get_wtime();
        	//elapsed_time_pwa+=end_pwa - start_pwa; 

        	learner_ciphertext.clear();
      	}

      	//auto start_serialize = std::chrono::system_clock::now();

      	stringstream ss;
      	Serial::Serialize(result_ciphertext, ss, st);
      	py::bytes res(ss.str());

      	//auto end_serialize = std::chrono::system_clock::now();
		    //elapsed_time_serialize+=std::chrono::duration_cast<std::chrono::milliseconds>(end_serialize
        //- start_serialize).count(); 
      	//std::cout <<"Deserialization time: "<< elapsed_time_deserialize*1000 << " milliseconds"<<'\n';
      	//std::cout <<"PWA Time: "<< elapsed_time_pwa*1000 << " milliseconds"<<'\n';
      	//std::cout <<"Serialization Time: "<< elapsed_time_serialize << " milliseconds"<<'\n';
      	result_ciphertext.clear();
      	return res;
	}


  string CKKS::computeWeightedAverage_cpp(vector<string> learners_Data, vector<float> scalingFactors) {

  if (learners_Data.size() != scalingFactors.size()) {
      cout << "Error: learners_Data and scalingFactors size mismatch" << endl;
      return "";
  }

  

  const SerType::SERBINARY st;
      vector<Ciphertext<DCRTPoly>> result_ciphertext;

      for (unsigned long int i = 0; i < learners_Data.size(); i++) {

        stringstream ss(learners_Data[i]);
        vector<Ciphertext<DCRTPoly>> learner_ciphertext;

        Serial::Deserialize(learner_ciphertext, ss, st);

        for (unsigned long int j = 0; j < learner_ciphertext.size(); j++) {

          float sc = scalingFactors[i];
          learner_ciphertext[j] = cc->EvalMult(learner_ciphertext[j], sc);
        }

        if (result_ciphertext.size() == 0) {

          result_ciphertext = learner_ciphertext;
        }

        else {

          for (unsigned long int j = 0; j < learner_ciphertext.size(); j++) {

            result_ciphertext[j] = cc->EvalAdd(result_ciphertext[j], learner_ciphertext[j]);
          }
        }


      }


      stringstream ss;
      Serial::Serialize(result_ciphertext, ss, st);
      return ss.str();


    
}



