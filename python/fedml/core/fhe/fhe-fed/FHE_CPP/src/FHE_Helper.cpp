#include "FHE_Helper.hpp"


FHE_Helper::FHE_Helper(string scheme, usint batchSize, usint scaleFactorBits, string cryptodir) {

    this->scheme = scheme;
    this->batchSize = batchSize;
    this->scaleFactorBits = scaleFactorBits;
    this->cryptodir = cryptodir;
}



void FHE_Helper::load_crypto_params() {

    if (!Serial::DeserializeFromFile(cryptodir + "/cryptocontext.txt", cc, SerType::BINARY)) {

        std::cout << "Could not read cryptocontext"<< std::endl;
    }

    if (!Serial::DeserializeFromFile(cryptodir + "/key-public.txt", pk, SerType::BINARY)) {
        
        std::cout << "Could not read public key" << std::endl;
    }

    if (!Serial::DeserializeFromFile(cryptodir + "/key-private.txt", sk, SerType::BINARY)) {
        
        std::cout << "Could not read secret key" << std::endl;
    }

}


int FHE_Helper::genCryptoContextAndKeys() {

    CryptoContext<DCRTPoly> cryptoContext;

    if (scheme == "bgvrns") {
      int plaintextModulus = 65537;
      double sigma = 3.2;
      uint32_t depth = 2;
      SecurityLevel securityLevel = HEStd_128_classic;
      

      cryptoContext = CryptoContextFactory<DCRTPoly>::genCryptoContextBGVrns(
                      depth, plaintextModulus, securityLevel, sigma, depth, OPTIMIZED, BV,
                      0, 0, 0, 0, 0, batchSize);


    } 

    else if (scheme == "ckks") {

      usint multDepth = 2;

      cryptoContext = CryptoContextFactory<DCRTPoly>::genCryptoContextCKKS(
          multDepth, scaleFactorBits, batchSize);
    }

    cryptoContext->Enable(ENCRYPTION);
    cryptoContext->Enable(SHE);
    // cryptoContext->Enable(LEVELEDSHE);


    if (!Serial::SerializeToFile(cryptodir + "/cryptocontext.txt",
                                 cryptoContext, SerType::BINARY)) {
      std::cerr << "Error writing serialization of the crypto context"<< std::endl;
      return 0;
    }

    
    LPKeyPair<DCRTPoly> keyPair;
    keyPair = cryptoContext->KeyGen();


    if (!Serial::SerializeToFile(cryptodir + "/key-public.txt",
                                 keyPair.publicKey, SerType::BINARY)) {
      std::cerr << "Error writing serialization of public key"<< std::endl;
      return 0;
    }


    if (!Serial::SerializeToFile(cryptodir + "/key-private.txt",
                                 keyPair.secretKey, SerType::BINARY)) {
      
      std::cerr<< "Error writing serialization of private key"<< std::endl;
      return 0;
    }


    // Generate the relinearization key
    cryptoContext->EvalMultKeyGen(keyPair.secretKey);


    std::ofstream emkeyfile(cryptodir + "/" + "key-eval-mult.txt",
                            std::ios::out | std::ios::binary);
    if(emkeyfile.is_open()) {
      if(cryptoContext->SerializeEvalMultKey(emkeyfile, SerType::BINARY) == false) {

        std::cerr << "Error writing serialization of the eval mult keys"<< std::endl;
        return 0;
      }

      emkeyfile.close();

    } 
    
    else {
      std::cerr << "Error serializing eval mult keys" << std::endl;
      return 0;
    }

    return 1;

}


string FHE_Helper::encrypt(vector<double> learner_Data) {


    unsigned long int size = learner_Data.size();
    vector<Ciphertext<DCRTPoly>> ciphertext_data((int)((size + batchSize) / batchSize));

    if(scheme == "ckks"){

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



    }

    else{

      std::cout << "Not supported!" << std::endl;
      return "";

    }


    stringstream s;
    const SerType::SERBINARY st;
    Serial::Serialize(ciphertext_data, s, st);


    return s.str();

   
}


string FHE_Helper::computeWeightedAverage(vector<string> learners_Data, vector<float> scalingFactors){

  if (learners_Data.size() != scalingFactors.size()) {
      cout << "Error: learners_Data and scalingFactors size mismatch" << endl;
      return "";
  }

  if(scheme == "ckks"){

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

  else {

    std::cout << "Not supported!" << std::endl;
    return "";

  }


    
}



vector<double> FHE_Helper::decrypt(string learner_Data, unsigned long int data_dimesions){

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