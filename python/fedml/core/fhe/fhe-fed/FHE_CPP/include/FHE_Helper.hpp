
#include "palisade.h"
#include <random>
#include <string>

#include "ciphertext-ser.h"
#include "cryptocontext-ser.h"
#include "pubkeylp-ser.h"
#include "scheme/bgvrns/bgvrns-ser.h"
#include "scheme/ckks/ckks-ser.h"
#include <string>

#include <omp.h>

using namespace std;
using namespace lbcrypto;
using namespace std::chrono;



class FHE_Helper{

    private:

      string scheme;
      usint batchSize;
      usint scaleFactorBits;
      std::string cryptodir;

      CryptoContext<DCRTPoly> cc;
      LPPublicKey<DCRTPoly> pk;
      LPPrivateKey<DCRTPoly> sk;


    public:

      FHE_Helper(string scheme, usint batchSize, usint scaleFactorBits, string cryptodir);

      int genCryptoContextAndKeys();
      void load_crypto_params();

      string encrypt(vector<double> data_array);
      string computeWeightedAverage(vector<string> learners_Data, vector<float> scalingFactors);
      vector<double> decrypt(string learner_Data, unsigned long int data_dimesions);


};

